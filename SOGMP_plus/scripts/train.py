#!/usr/bin/env python
#
# file: $ISIP_EXP/SOGMP/scripts/train.py
#
# revision history: xzt
#  20220824 (TE): first version
#
# usage:
#  python train.py mdir trian_data val_data
#
# arguments:
#  mdir: the directory where the output model is stored
#  trian_data: the directory of training data
#  val_data: the directory of valiation data
#
# This script trains a SOGMP++ model
#------------------------------------------------------------------------------

# import pytorch modules
#
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import time

# visualize:
from tensorboardX import SummaryWriter
import numpy as np
# import wandb
# wandb.init(project="ogm_pred_dataset")

# import the model and all of its variables/functions
#
from model import *
from local_occ_grid_map import LocalMap
from dtaci_grid import DtACIGrid
from onpolicy.utils.mpe_runner_util import fuse_ogm

# import modules
#
import sys
import os


#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# general global values
#
# model_dir = 'SOGMP_plus/models'  # the path of model storage 

MODEL_NAME = 'models_library_holo_no_pretrained'
DATASET_NAME = 'datasets_library_holo_no_pretrained'
NUM_ARGS = 3
NUM_EPOCHS = 20 #100
BATCH_SIZE = 10 #512 #64
# BATCH_SIZE = 1
LEARNING_RATE = "lr"
BETAS = "betas"
EPS = "eps"
WEIGHT_DECAY = "weight_decay"

# Constants
NUM_INPUT_CHANNELS = 1
NUM_LATENT_DIM = 512 # 16*16*2 
NUM_OUTPUT_CHANNELS = FUTURE_STEP
BETA = 0.01

# Init map parameters
P_prior = 0.5	# Prior occupancy probability
P_occ = 0.7	    # Probability that cell is occupied with total confidence
P_free = 0.3	# Probability that cell is free with total confidence 
MAP_X_LIMIT = [0, 6.4]      # Map limits on the x-axis
MAP_Y_LIMIT = [-3.2, 3.2]   # Map limits on the y-axis
TRESHOLD_P_OCC = 0.8    # Occupancy threshold

VIS_SEQ = 5
NUM_SAMPES = 8
TEST_INTERVAL = 5
SAVE_INTERVAL = 5
coop_mode = 'early'
# coop_mode = 'no_coop'
# for reproducibility, we seed the rng
#
set_seed(SEED1)       

# adjust_learning_rate

def adjust_learning_rate(optimizer, epoch):
    lr = 1e-4
    if epoch > 30000:
        lr = 3e-4
    if epoch > 50000:
        lr = 2e-5
    if epoch > 48000:
       # lr = 5e-8
       lr = lr * (0.1 ** (epoch // 110000))
    #  if epoch > 8300:
    #      lr = 1e-9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def log_odds(p):
    """
    Log odds ratio of p(x):

                p(x)
    l(x) = log ----------
                1 - p(x)

    """
    p = torch.tensor(p)

    return torch.log(p / (1 - p))

def retrieve_p(log_map):
    """
    Retrieve p(x) from log odds ratio:

                    1
    p(x) = 1 - ---------------
                1 + exp(l(x))

    """
    prob_map = 1 - 1 / (1 + torch.exp(log_map))

    return prob_map
    
def calc_MLE(prob_map, threshold_p_occ):
    """
    Calculate Maximum Likelihood estimate of the map (binary map)
    """
    prob_map[prob_map >= threshold_p_occ] = 1
    prob_map[prob_map < threshold_p_occ] = 0

    return prob_map

def to_prob_occ_map(occ_map, threshold_p_occ):
    """
    Transformation to GRAYSCALE image format
    """
    log_map = torch.sum(occ_map, dim=1)  # sum of all timestep maps
    prob_map = retrieve_p(log_map)
    prob_map = calc_MLE(prob_map, threshold_p_occ)

    return prob_map

def shannon_entropy(ogm):
    """
    Compute the Shannon entropy for the predicted OGM.
    
    Args:
        ogm (torch.Tensor): Predicted occupancy grid map, shape (samples, batch_size, vis_seq, 64, 64)

    Returns:
        entropy (torch.Tensor): Entropy value for each sample in the batch.
    """
    # eps = 1e-8
    # N = ogm.shape[-2] * ogm.shape[-1]

    # # Compute entropy using the Shannon entropy formula
    # entropy = -torch.sum(ogm * torch.log(ogm + eps) + (1 - ogm) * torch.log(1 - ogm + eps), dim=(-2, -1)) / N
    # entropy=torch.mean(entropy)
    eps = 1e-8  # Small constant to avoid log(0)

    # Compute entropy per pixel using the Shannon entropy formula
    entropy_map = - (ogm * torch.log(ogm + eps) + (1 - ogm) * torch.log(1 - ogm + eps))

    entropy_map = entropy_map.mean(dim=0)
    
    return entropy_map

def train(model, dataloader, dataset, device, optimizer, criterion, epoch, epochs, coop_mode='no_coop'):
    
    # set model to training mode:
    model.train()
    # for each batch in increments of batch size:
    running_loss = 0.0
    # kl_divergence:
    kl_avg_loss = 0.0
    # CE loss:
    ce_avg_loss = 0.0
    
    total_wmse_per_step = torch.zeros(FUTURE_STEP)
    total_ssim_per_step = torch.zeros(FUTURE_STEP)
    total_iou_per_step = torch.zeros(FUTURE_STEP)
      
    criterion_wmse = WeightedMSELoss()

    counter = 0
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(dataset)/dataloader.batch_size)

    time_cost = 0
    
    for i, batch in tqdm(enumerate(dataloader), total=num_batches):
    
        counter += 1
        # collect the samples as a batch:
        ego_agent_id = batch['ego_agent_id'][0]

        ogm = batch['ego_ogm']
        ogm = ogm.to(device) # b seq_len map_size map_size
        
        if coop_mode == 'early':
            fused_ogm = batch['fused_ogm']
            fused_ogm = fused_ogm.to(device) # b 2 seq_len map_size map_size
            vis_ogm = fused_ogm[:,:VIS_SEQ,:]
            target_ogm = fused_ogm[:,VIS_SEQ:,:]
        else:
            vis_ogm = ogm[:,:VIS_SEQ,:]
            target_ogm = ogm[:,VIS_SEQ:,:]
        
        batch_size, seq_len, H, W = ogm.shape
        # predict multiple steps per forward pass
        occ_map = torch.full((batch_size, VIS_SEQ, H, W), fill_value = log_odds(P_prior)).to(device)
        occ_map[vis_ogm==0] += log_odds(P_free)
        occ_map[vis_ogm==1] += log_odds(P_occ)
        occ_map = to_prob_occ_map(occ_map, TRESHOLD_P_OCC) # b, map_size, map_size
                
        optimizer.zero_grad()
        start_time = time.time()
        prediction, kl_loss = model(vis_ogm, occ_map)
        time_cost += time.time() - start_time

        ce_loss = criterion(prediction, target_ogm).div(batch_size)
        # beta-vae:

        # perform back propagation:
        loss = ce_loss + BETA*kl_loss
        loss.backward(torch.ones_like(loss))
            
        optimizer.step()
        
        # multiple GPUs:
        if torch.cuda.device_count() > 1:
            loss = loss.mean()  
            ce_loss = ce_loss.mean()
            kl_loss = kl_loss.mean()

        running_loss += loss.item()
        # kl_divergence:
        kl_avg_loss += kl_loss.item()
        # CE loss:
        ce_avg_loss += ce_loss.item()

        for step in range(FUTURE_STEP):
            step_pred = prediction[:, step, :, :].unsqueeze(1)  # [B, 1, H, W]
            step_target = target_ogm[:, step, :, :].unsqueeze(1)  # [B, 1, H, W]

            step_wmse = criterion_wmse(step_pred, step_target, calculate_weights(step_target))
            step_ssim = calculate_ssim(step_pred, step_target)
            step_iou = calculate_iou(step_pred, step_target)

            total_wmse_per_step[step] += step_wmse.item()
            total_ssim_per_step[step] += step_ssim.item()
            total_iou_per_step[step] += step_iou
        
        # display informational message:
        # if(i % 64 == 0):
        #     print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, CE_Loss: {:.4f}, KL_Loss: {:.4f}'
        #             .format(epoch, epochs, i + 1, num_batches, loss.item(), ce_loss.item(), kl_loss.item()))
    train_loss = running_loss / counter 
    train_kl_loss = kl_avg_loss / counter
    train_ce_loss = ce_avg_loss / counter
    avg_wmse_per_step = (total_wmse_per_step / counter).tolist()
    avg_ssim_per_step = (total_ssim_per_step / counter).tolist()
    avg_iou_per_step = (total_iou_per_step / counter).tolist()

    fps = counter / time_cost

    return train_loss, train_kl_loss, train_ce_loss, avg_wmse_per_step, avg_ssim_per_step, avg_iou_per_step, fps

def calculate_occupied_grid_rate(grid_map):
    # Assuming grid_map is a binary map (0 for free space, 1 for occupied)
    occupied_cells = torch.sum(grid_map > 0).item()
    total_cells = grid_map.numel()
    occupied_rate = occupied_cells / total_cells
    return occupied_rate

def validate(model, dataloader, dataset, device, criterion, coop_mode='no_coop'): 
    
    model.eval()
    
    running_loss = 0.0
    kl_avg_loss = 0.0
    ce_avg_loss = 0.0
    total_wmse_per_step = torch.zeros(FUTURE_STEP)
    total_ssim_per_step = torch.zeros(FUTURE_STEP)
    total_iou_per_step = torch.zeros(FUTURE_STEP)
    # total_wmse = 0
    # total_ssim = 0  # Initialize total SSIM

    criterion_wmse = WeightedMSELoss()

    counter = 0
    num_batches = int(len(dataset) / dataloader.batch_size)

    grid_height, grid_width = MAP_SIZE, MAP_SIZE
    dtaci_estimator = DtACIGrid(grid_height, grid_width, alpha=0.1)

    time_cost = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=num_batches):
            
            counter += 1
            # collect the samples as a batch:
            ego_agent_id = batch['ego_agent_id'][0]

            ogm = batch['ego_ogm']
            ogm = ogm.to(device) # b seq_len map_size map_size

            fused_ogm = batch['fused_ogm']
            fused_ogm = fused_ogm.to(device) # b seq_len map_size map_size
            
            if coop_mode == 'late':
                rob_ogms = batch['rob_ogms']
                rob_ogms = rob_ogms.to(device) # b seq_len map_size map_size
                rob_translation_2d = batch['rob_translation_2d']
            
            batch_size, seq_len, H, W = ogm.shape
            
            if coop_mode == 'early':
                vis_ogm = fused_ogm[:,:VIS_SEQ,:]
            else:
                vis_ogm = ogm[:,:VIS_SEQ,:]
                
            target_ogm = fused_ogm[:,VIS_SEQ:,:]

            occ_map = torch.full((batch_size, VIS_SEQ, H, W), fill_value = log_odds(P_prior)).to(device)
            occ_map[vis_ogm==0] += log_odds(P_free)
            occ_map[vis_ogm==1] += log_odds(P_occ)
            occ_map = to_prob_occ_map(occ_map, TRESHOLD_P_OCC)

            vis_ogm_sample = vis_ogm.repeat(NUM_SAMPES, 1, 1, 1, 1)
            occ_map_sample = occ_map.repeat(NUM_SAMPES, 1, 1, 1)

            start_time = time.time()
            prediction_samples, kl_loss = model(vis_ogm_sample, occ_map_sample)
            time_cost += time.time() - start_time
            prediction_samples = prediction_samples.reshape(NUM_SAMPES, -1, FUTURE_STEP, H, W)
            prediction = prediction_samples.mean(dim=0)

            pred_entropy = shannon_entropy(prediction_samples).detach().cpu().numpy() # samples, batch, t, map_size, map_size

            if coop_mode == 'late':
                prediction = prediction.detach().cpu().numpy()
                for sourround_agent_id in range(3):
                    if sourround_agent_id != ego_agent_id:
                        rob_ogm = rob_ogms[:,sourround_agent_id,:VIS_SEQ,:,:]
                        rob_occ_map = torch.full((batch_size, VIS_SEQ, H, W), fill_value = log_odds(P_prior)).to(device)
                        rob_occ_map[rob_ogm==0] += log_odds(P_free)
                        rob_occ_map[rob_ogm==1] += log_odds(P_occ)
                        rob_occ_map = to_prob_occ_map(rob_occ_map, TRESHOLD_P_OCC)

                        rob_ogm_sample = rob_ogm.repeat(NUM_SAMPES, 1, 1, 1, 1)
                        rob_occ_map_sample = rob_occ_map.repeat(NUM_SAMPES, 1, 1, 1)

                        rob_prediction_samples, kl_loss = model(rob_ogm_sample, rob_occ_map_sample)
                        rob_prediction_samples = rob_prediction_samples.reshape(NUM_SAMPES, -1, FUTURE_STEP, H, W)
                        rob_prediction = rob_prediction_samples.mean(dim=0).detach().cpu().numpy()
                        start_time = time.time()
                        prediction = fuse_ogm(prediction, rob_prediction, rob_translation_2d[:, sourround_agent_id], is_prediction=True)
                        time_cost += time.time() - start_time
                
                prediction = torch.from_numpy(prediction).to(device)
                    
            ce_loss = criterion(prediction, target_ogm).div(batch_size)
            # beta-vae:
            loss = ce_loss + BETA*kl_loss

            # multiple GPUs:
            if torch.cuda.device_count() > 1:
                loss = loss.mean()  
                ce_loss = ce_loss.mean()
                kl_loss = kl_loss.mean()

            running_loss += loss.item()
            # kl_divergence:
            kl_avg_loss += kl_loss.item()
            # CE loss:
            ce_avg_loss += ce_loss.item()

            # compute metrics step by step:
            for step in range(FUTURE_STEP):
                step_pred = prediction[:, step, :, :].unsqueeze(1)  # [B, 1, H, W]
                step_target = target_ogm[:, step, :, :].unsqueeze(1)  # [B, 1, H, W]

                step_wmse = criterion_wmse(step_pred, step_target, calculate_weights(step_target))
                step_ssim = calculate_ssim(step_pred, step_target)
                step_iou = calculate_iou(step_pred, step_target)

                total_wmse_per_step[step] += step_wmse.item()
                total_ssim_per_step[step] += step_ssim.item()
                total_iou_per_step[step] += step_iou

            # plot the vis_ogm, target_ogm, and prediction:
            # fig, axs = plt.subplots(nrows=4, ncols=VIS_STEP+FUTURE_STEP, figsize=(4 * (FUTURE_STEP + VIS_STEP), 21))
            # # Loop over each time step
            # for t in range(VIS_STEP+FUTURE_STEP):

            #     if t < VIS_SEQ:
            #     # First row: vis_ogm
            #         axs[0, t].imshow(fused_ogm[0, t].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
            #         axs[0, t].set_title(f"ego_ogm t={t}")
            #         axs[0, t].axis("off")

            #     else:
            #         # Second row: target_ogm
            #         # Make sure that target_ogm has enough frames at index t (if seq_len = VIS_STEP + future steps)
            #         axs[1, t].imshow(fused_ogm[0, t].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
            #         axs[1, t].set_title(f"ego_target_ogm t={t}")
            #         axs[1, t].axis("off")

            #         # Third row: prediction
            #         axs[2, t].imshow(prediction[0, t-VIS_SEQ].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
            #         axs[2, t].set_title(f"ego_prediction t={t}")
            #         axs[2, t].axis("off")

            #         # axs[3, t].imshow(pred_entropy[0, t-VIS_SEQ], cmap='gray', interpolation='nearest')
            #         # axs[3, t].set_title(f"ego_entropy map t={t}")
            #         # axs[3, t].axis("off")
            #         im = axs[3, t].imshow(pred_entropy[0, t-VIS_SEQ], cmap='viridis', interpolation='nearest') # Or another suitable cmap
            #         axs[3, t].set_title(f"ego_entropy map t={t}")
            #         axs[3, t].axis("off")

            #         # Add a colorbar (important for interpreting the heatmap)
            #         cbar = fig.colorbar(im, ax=axs[3, t])
            #         cbar.set_label("Entropy")
    
            #     # axs[4, t].imshow(rob_ogms[0, 0, t].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
            #     # axs[4, t].set_title(f"rob 1 map t={t}")
            #     # axs[4, t].axis("off")

            #     # axs[5, t].imshow(rob_ogms[0, 1, t].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
            #     # axs[5, t].set_title(f"rob 2 map t={t}")
            #     # axs[5, t].axis("off")
                
            #     # axs[6, t].imshow(fused_ogm[0, t].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
            #     # axs[6, t].set_title(f"fused map t={t}")
            #     # axs[6, t].axis("off")

            # plt.tight_layout()
            # plt.show()

            # multi_step_loss = 0.0     
            # current_vis_ogm = vis_ogm.clone()  # shape: (b, seq_len, H, W)
            # current_occ_map = occ_map.clone()

            # multi_step_loss = 0.0
            # multi_step_kl   = 0.0
            # multi_step_ce   = 0.0
            
            # for t in range(FUTURE_STEP):
                
            #     # Forward pass
            #     prediction, kl_loss = model(current_vis_ogm, current_occ_map)  

            #     target_ogm = ogm[:,VIS_SEQ+t,:].unsqueeze(1)
                
            #     # Compute cross-entropy (or BCE/MSE) per step
            #     ce_loss = criterion(prediction, target_ogm)
                
            #     # Weighted MSE metric (for logging, not included in final training loss)
            #     wmse_step = criterion_wmse(prediction, target_ogm, calculate_weights(target_ogm))
            #     ssim_step = calculate_ssim(prediction, target_ogm)
                
            #     # Accumulate step losses
            #     multi_step_ce += ce_loss
            #     multi_step_kl += kl_loss
            #     total_wmse += wmse_step.item()
            #     total_ssim += ssim_step.item()
                
            #     # If you want to combine CE + Beta * KL for your training objective:
            #     multi_step_loss += ce_loss + BETA * kl_loss

            #     current_vis_ogm = torch.cat([current_vis_ogm[:, 1:], prediction], dim=1) 

            #     current_occ_map = torch.full((batch_size, VIS_SEQ, H, W), fill_value = log_odds(P_prior)).to(device)
            #     current_occ_map[current_vis_ogm==0] += log_odds(P_free)
            #     current_occ_map[current_vis_ogm==1] += log_odds(P_occ)
            #     current_occ_map = to_prob_occ_map(current_occ_map, TRESHOLD_P_OCC) # t, map_size, map_size
                
            #     # Plot past ogm, gt ogm and predicted ogm in the same figure
            #     # fig, axs = plt.subplots(nrows=1, ncols=VIS_STEP+2, figsize=(3*VIS_STEP+6, 3))
            #     # axs = axs.flatten()  # Flatten the array of axes if necessary

            #     # for i, ax in enumerate(axs):
            #     #     if i < len(vis_ogm[0]):
            #     #         ax.imshow(vis_ogm[0][i].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
            #     #     elif i == len(vis_ogm[0]):
            #     #         # ax.axis('off')  # Ensure no empty subplots have visible axes
            #     #         ax.imshow(target_ogm[0][0].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
            #     #     else:
            #     #         ax.imshow(prediction[0][0].detach().cpu().numpy(), cmap='gray', interpolation='nearest')

            #     # plt.tight_layout()
            #     # plt.show()
                
            # if torch.cuda.device_count() > 1:
            #     multi_step_loss = multi_step_loss.mean()
            #     multi_step_ce   = multi_step_ce.mean()
            #     multi_step_kl   = multi_step_kl.mean()

            # running_loss += multi_step_loss.item()
            # kl_avg_loss += multi_step_kl.item()
            # ce_avg_loss += multi_step_ce.item()
                
    val_loss = running_loss / counter
    val_kl_loss = kl_avg_loss / counter
    val_ce_loss = ce_avg_loss / counter

    avg_wmse_per_step = (total_wmse_per_step / counter).tolist()
    avg_ssim_per_step = (total_ssim_per_step / counter).tolist()
    avg_iou_per_step = (total_iou_per_step / counter).tolist()

    fps = counter / time_cost

    return val_loss, val_kl_loss, val_ce_loss, avg_wmse_per_step, avg_ssim_per_step, avg_iou_per_step, fps

def calculate_ssim(pred_ogm, gt_ogm, C1=1e-4, C2=9e-4):
    """
    Calculate the Structural Similarity Index Measure (SSIM) from predicted and ground truth OGMs.
    
    Parameters:
    pred_ogm (torch.Tensor): Predicted occupancy grid map tensor.
    gt_ogm (torch.Tensor): Ground truth occupancy grid map tensor.
    C1 (float): Constant to avoid instability, default is 1e-4.
    C2 (float): Constant to avoid instability, default is 9e-4.
    
    Returns:
    torch.Tensor: SSIM value.
    """
    mu_pred = torch.mean(pred_ogm)
    mu_gt = torch.mean(gt_ogm)
    
    delta_pred = torch.var(pred_ogm)
    delta_gt = torch.var(gt_ogm)
    
    delta_pred_gt = torch.mean((pred_ogm - mu_pred) * (gt_ogm - mu_gt))
    
    numerator = (2 * mu_pred * mu_gt + C1) * (2 * delta_pred_gt + C2)
    denominator = (mu_pred**2 + mu_gt**2 + C1) * (delta_pred + delta_gt + C2)
    ssim = numerator / denominator
    
    return ssim

def calculate_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    intersection = (pred_bin * target_bin).sum(dim=[1, 2, 3])
    union = ((pred_bin + target_bin) > 0).float().sum(dim=[1, 2, 3])
    iou = (intersection / (union + 1e-6)).mean().item()
    return iou

class WeightedMSELoss(nn.Module):
    def forward(self, input, target, weights):
        squared_diffs = (input - target) ** 2
        weighted_squared_diffs = weights * squared_diffs
        wmse = weighted_squared_diffs.sum() / weights.sum()
        return wmse

def calculate_weights(ground_truth):
    # Assuming ground_truth is a binary tensor where 1 represents occupied and 0 represents free
    total_cells = ground_truth.numel()
    occupied_count = ground_truth.sum()
    free_count = total_cells - occupied_count

    # Frequencies
    freq_occupied = occupied_count / total_cells
    freq_free = free_count / total_cells

    # Median frequency
    median_freq = torch.median(torch.tensor([freq_occupied, freq_free]))

    # Weights
    weight_occupied = median_freq / freq_occupied if freq_occupied > 0 else 0
    weight_free = median_freq / freq_free if freq_free > 0 else 0

    # Create weight map based on ground truth
    weights = torch.where(ground_truth == 1, weight_occupied, weight_free)
    return weights
#------------------------------------------------------------------------------
#
# the main program starts here
#
#------------------------------------------------------------------------------

# function: main
#
# arguments: none
#
# return: none
#
# This method is the main function.
#
def main():
    # # ensure we have the correct amount of arguments:
    # #global cur_batch_win
    # if(len(argv) != NUM_ARGS):
    #     print("usage: python train.py [MDL_PATH] [TRAIN_PATH] [VAL_PATH]")
    #     exit(-1)

    # # define local variables:
    # mdl_path = argv[0]
    # pTrain = argv[1]
    # pDev = argv[2]

    # get the output directory name:
    if coop_mode == 'late':
        # odir = f'SOGMP_plus/models_0.2s_no_coop_{FUTURE_STEP}'
        odir = f'SOGMP_plus/{MODEL_NAME}_no_coop_{FUTURE_STEP}'
    else:
        # odir = f'SOGMP_plus/models_0.2s_{coop_mode}_{FUTURE_STEP}'
        odir = f'SOGMP_plus/{MODEL_NAME}_{coop_mode}_{FUTURE_STEP}'

    ckpts_epoch = 80

    # if the odir doesn't exits, we make it:
    if not os.path.exists(odir):
        os.makedirs(odir)

    # set the device to use GPU if available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('...Start reading data...')

    if coop_mode == 'no_coop':
        train_dataset = VaeDatasetNoFusion(f'SOGMP_plus/{DATASET_NAME}')
        dev_dataset = VaeTestDatasetNoFusion(f'SOGMP_plus/{DATASET_NAME}', is_train=False)
    elif coop_mode == 'early':
        train_dataset = VaeDatasetEarlyFusion(f'SOGMP_plus/{DATASET_NAME}')
        dev_dataset = VaeDatasetEarlyFusion(f'SOGMP_plus/{DATASET_NAME}', is_train=False)
    else:
        train_dataset = VaeDatasetNoFusion(f'SOGMP_plus/{DATASET_NAME}')
        dev_dataset = VaeTestDatasetLateFusion(f'SOGMP_plus/{DATASET_NAME}', is_train=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, \
                                                   shuffle=True, drop_last=True, pin_memory=True)

    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=2, \
                                                 shuffle=True, drop_last=True, pin_memory=True)

    print("Training set size: ", len(train_dataset))
    print("Validation set size: ", len(dev_dataset))

    # # instantiate a model:
    model = RVAEP(input_channels=NUM_INPUT_CHANNELS,
                  latent_dim=NUM_LATENT_DIM,
                  output_channels=NUM_OUTPUT_CHANNELS)
    # #moves the model to device (cpu in our case so no change):

    model.to(device)
    
    # set the adam optimizer parameters:
    opt_params = { LEARNING_RATE: 0.001,
                   BETAS: (.9,0.999),
                   EPS: 1e-08,
                   WEIGHT_DECAY: .001 }
    # set the loss criterion and optimizer:
    criterion = nn.BCELoss(reduction='sum') #, weight=class_weights)
    criterion.to(device)
    # create an optimizer, and pass the model params to it:
    #all_params = list(encoder.parameters())
    optimizer = Adam(model.parameters(), **opt_params)
    
    # get the number of epochs to train on:
    epochs = NUM_EPOCHS
    
    if os.path.exists(os.path.join(odir, 'model{}.pth'.format(ckpts_epoch))):
        checkpoint = torch.load(os.path.join(odir, 'model{}.pth'.format(ckpts_epoch)))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from epoch {}'.format(start_epoch))
    else:
        start_epoch = 0
        print('No trained models, restart training')
        
    # multiple GPUs:
    if torch.cuda.device_count() > 1:
        print("Let's use 2 of total", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model) #, device_ids=[0, 1])
    # moves the model to device (cpu in our case so no change):
    model.to(device)

    # tensorboard writer:
    writer = SummaryWriter(f'SOGMP_plus/{DATASET_NAME}_{MODEL_NAME}_{coop_mode}_{FUTURE_STEP}_runs')

    epoch_num = 0
    for epoch in range(start_epoch+1, epochs):
        # adjust learning rate:
        adjust_learning_rate(optimizer, epoch)
        ################################## Train #####################################
        # for each batch in increments of batch size
        #

        train_epoch_loss, train_kl_epoch_loss, train_ce_epoch_loss, wmse_train, ssim_train, iou_train, fps_train = train(
            model, train_dataloader, train_dataset, device, optimizer, criterion, epoch, epochs, coop_mode
        )

        # log the epoch loss
        writer.add_scalar('training loss',
                        train_epoch_loss,
                        epoch)
        writer.add_scalar('training kl loss',
                        train_kl_epoch_loss,
                        epoch)
        writer.add_scalar('training ce loss',
                train_ce_epoch_loss,
                epoch)

        print('Epoch [{}/{}], Train Loss: {:.4f}, train WMSE: {}, train SSIM: {}, train IOU: {}, train FPS: {}'.format(
            epoch, epochs, train_epoch_loss, wmse_train, ssim_train, iou_train, fps_train))

        if epoch % TEST_INTERVAL == 0:
            valid_epoch_loss, valid_kl_epoch_loss, valid_ce_epoch_loss, wmse, ssim, iou, fps = validate(
                model, dev_dataloader, dev_dataset, device, criterion, coop_mode
            )
            writer.add_scalar('validation loss',
                            valid_epoch_loss,
                            epoch)
            writer.add_scalar('validation kl loss',
                            valid_kl_epoch_loss,
                            epoch)
            writer.add_scalar('validation ce loss',
                            valid_ce_epoch_loss,
                            epoch)
            
            print('Epoch [{}/{}], Val Loss: {:.4f}, val WMSE: {}, val SSIM: {}, val IOU: {}, val FPS: {}'.format(
                epoch, epochs, valid_epoch_loss, wmse, ssim, iou, fps))
            
         #Log metrics to wandb
        # wandb.log({"Train Loss": train_epoch_loss, "Train KL Loss": train_kl_epoch_loss, "Train CE Loss": train_ce_epoch_loss,"Train_WMSE": wmse_train/3, "Train_SSIM": ssim_train/3,
        #         "Validation Loss": valid_epoch_loss, "Validation KL Loss": valid_kl_epoch_loss, "Validation CE Loss": valid_ce_epoch_loss,
        #         "Val_WMSE": wmse/3, "Val_SSIM": ssim/3})
        #"Train RESET": RESET_COUNT_train,"Val RESET": RESET_COUNT     
                
        # save the model:
        if epoch % SAVE_INTERVAL == 0:
            if torch.cuda.device_count() > 1: # multiple GPUS: 
                state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            else:
                state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            # path = f'SOGMP_plus/models_0.2s_{coop_mode}_{FUTURE_STEP}/model' + str(epoch) +'.pth'
            path = os.path.join(odir, 'model{}.pth'.format(epoch))
            torch.save(state, path)
            
        epoch_num = epoch   

    # exit gracefully
    #

    return True
#
# end of function


# begin gracefully
#
if __name__ == '__main__':
    main()
#
# end of file
