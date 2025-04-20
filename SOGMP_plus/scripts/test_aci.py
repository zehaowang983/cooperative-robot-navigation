import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# visualize:
from tensorboardX import SummaryWriter
import numpy as np
# import wandb
# wandb.init(project="ogm_pred_dataset")

# import the model and all of its variables/functions
#
from model import *
from train import *
from local_occ_grid_map import LocalMap
from dtaci_grid import DtACIGrid

# import modules
#
import sys
import os

coop_mode = 'no_coop'

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
    
    odir = f'SOGMP_plus/models_0.2s_{coop_mode}_{FUTURE_STEP}'
    data_path = f'SOGMP_plus/datasets_new_0.2s'
    ckpts_epoch = 15

    # set the device to use GPU if available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    # multiple GPUs:
    if torch.cuda.device_count() > 1:
        print("Let's use 2 of total", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model) #, device_ids=[0, 1])
    # moves the model to device (cpu in our case so no change):

    if os.path.exists(os.path.join(odir, 'model{}.pth'.format(ckpts_epoch))):
        checkpoint = torch.load(os.path.join(odir, 'model{}.pth'.format(ckpts_epoch)))
        # model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from epoch {}'.format(start_epoch))
    else:
        start_epoch = 0
        print('No trained models')
        exit(-1)
    
    print('...Start reading data...')
        
    for agent_id in range(3):
        all_lidar_data = np.load(os.path.join(data_path, f"all_lidar_data_{agent_id}_val.npy"))
        all_pose_data = np.load(os.path.join(data_path, f"all_pos_data_{agent_id}_val.npy"))
        all_traj_length = np.load(os.path.join(data_path, f"all_traj_length_{agent_id}_val.npy"))
        all_traj_end_index = np.cumsum(all_traj_length)
        for i in range(len(all_traj_end_index)):
            
            if i == 0:
                start_idx = 0
            else:
                start_idx = all_traj_end_index[i-1]
            end_idx = all_traj_end_index[i]

            grid_height, grid_width = MAP_SIZE, MAP_SIZE
            dtaci_estimator = [DtACIGrid(grid_height, grid_width, alpha=0.1, initial_pred=0.1) for _ in range(FUTURE_STEP)]
            
            # save whole ogm sequence
            for t in range(start_idx, end_idx - TOTAL_LEN + 1):
                scans = all_lidar_data[t:t+TOTAL_LEN]
                poses = all_pose_data[t:t+TOTAL_LEN]

                ego_motion_translation_2d = np.stack([poses[VIS_STEP-1, 1] - poses[:, 1], poses[VIS_STEP-1, 0] - poses[:, 0]], axis=-1) # t, 2
                angles = np.linspace(0, 2 * np.pi, 1080, endpoint=False)
                transformed_lidar = np.zeros((TOTAL_LEN, 1080))
                
                for i in range(TOTAL_LEN):

                    tx, ty = ego_motion_translation_2d[i]
                    
                    robot_points_x = scans[i] * np.cos(angles)
                    robot_points_y = scans[i] * np.sin(angles)

                    # Mask valid points
                    valid_points_mask = (scans[i] > 0)
                    valid_robot_points_x = robot_points_x[valid_points_mask]
                    valid_robot_points_y = robot_points_y[valid_points_mask]
                    valid_angles = angles[valid_points_mask]

                    # Transform to ego's coordinate system
                    ego_points_x = valid_robot_points_x - tx
                    ego_points_y = valid_robot_points_y - ty
                    
                    ego_points_r = np.sqrt(ego_points_x**2 + ego_points_y**2)
                    ego_points_angles = np.arctan2(ego_points_y, ego_points_x)

                    # Apply valid range filter (0.5m < distance < 5m)
                    # valid_ego_points_mask = (0.5 < ego_points_r) & (ego_points_r <= 5)
                    valid_ego_points_mask = (0.3 < ego_points_r) & (ego_points_r <= 7)
                    ego_points_r = ego_points_r[valid_ego_points_mask]
                    ego_points_angles = ego_points_angles[valid_ego_points_mask]

                    # Normalize angles to match lidar bins
                    ego_points_angles_normalized = np.mod(ego_points_angles, 2 * np.pi)
                    ego_points_indices = np.floor(ego_points_angles_normalized / (2 * np.pi / 1080)).astype(int)
                    ego_points_indices = np.mod(ego_points_indices, 1080)  # Ensure indices are within bounds
                    
                    for j, index in enumerate(ego_points_indices):
                        transformed_lidar[i][index] = ego_points_r[j]
                    
                ego_transformed_ogm = transform_lidar_to_ogm(transformed_lidar, map_size=MAP_SIZE)
                ego_transformed_ogm = np.expand_dims(ego_transformed_ogm, axis=0)
                
                vis_ogm = torch.FloatTensor(ego_transformed_ogm[:, :VIS_SEQ]).to(device)
                
                batch, _, H, W = vis_ogm.shape
                occ_map = torch.full((batch, VIS_SEQ, H, W), fill_value = log_odds(P_prior)).to(device)
                occ_map[vis_ogm==0] += log_odds(P_free)
                occ_map[vis_ogm==1] += log_odds(P_occ)
                occ_map = to_prob_occ_map(occ_map, TRESHOLD_P_OCC)

                vis_ogm_sample = vis_ogm.repeat(NUM_SAMPES, 1, 1, 1, 1)
                occ_map_sample = occ_map.repeat(NUM_SAMPES, 1, 1, 1)

                prediction_samples, kl_loss = model(vis_ogm_sample, occ_map_sample)
                prediction_samples = prediction_samples.reshape(NUM_SAMPES, -1, FUTURE_STEP, H, W)
                prediction = prediction_samples.mean(dim=0)

                gt_ogm = ego_transformed_ogm[:, VIS_SEQ:]

                for i in range(FUTURE_STEP):
                    nonconformity_score = np.linalg.norm(gt_ogm[:, i] - prediction[:, i].detach().cpu().numpy(), axis=0)
                    dtaci_estimator[i].update_true_grid(nonconformity_score)
                    aci_predicted_conformity_scores = np.clip(dtaci_estimator[i].make_prediction(), 0, 1)
                    # mask = aci_predicted_conformity_scores > 0.3
                    # aci_predicted_conformity_scores[~mask] = 0
                    # plt.imshow(aci_predicted_conformity_scores, cmap='gray')
                    # plt.show()
                    plt.imsave(f'SOGMP_plus/dtaci/agent_{agent_id}_pred_{t}_{i}.png', aci_predicted_conformity_scores, cmap='gray')

                # dtaci_pred_grid = dtaci_estimator.get_prediction_grid()
                # dtaci_error_grid = dtaci_estimator.get_error_grid()
                # dtaci_coverage_grid = dtaci_estimator.get_coverage_grid()

                # plt.imsave(f'SOGMP_plus/dtaci/agent_{agent_id}_ogm_{i}_{t}.png', ogm[0, VIS_SEQ-1], cmap='viridis')
                # plt.imsave(f'SOGMP_plus/dtaci/agent_{agent_id}_pred_{i}_{t}.png', prediction[0, 0].detach().cpu().numpy(), cmap='viridis')
            break
        break

    return True
    
if __name__ == '__main__':
    main()