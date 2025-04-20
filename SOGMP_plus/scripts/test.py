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
    if coop_mode == 'late':
        odir = f'SOGMP_plus/models_0.2s_no_coop_{FUTURE_STEP}'
    else:
        odir = f'SOGMP_plus/models_0.2s_{coop_mode}_{FUTURE_STEP}'
        
    ckpts_epoch = 30

    # set the device to use GPU if available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('...Start reading data...')

    if coop_mode == 'no_coop':
        dev_dataset = VaeTestDatasetNoFusion('SOGMP_plus/datasets_new_0.2s', is_train=False)
    elif coop_mode == 'early':
        dev_dataset = VaeDatasetEarlyFusion('SOGMP_plus/datasets_new_0.2s', is_train=False)
    else:
        dev_dataset = VaeTestDatasetLateFusion('SOGMP_plus/datasets_new_0.2s', is_train=False)
        
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=2, \
                                                 shuffle=False, drop_last=True, pin_memory=True)

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
    
    # multiple GPUs:
    if torch.cuda.device_count() > 1:
        print("Let's use 2 of total", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model) #, device_ids=[0, 1])
    # moves the model to device (cpu in our case so no change):
    model.to(device)

    if os.path.exists(os.path.join(odir, 'model{}.pth'.format(ckpts_epoch))):
        checkpoint = torch.load(os.path.join(odir, 'model{}.pth'.format(ckpts_epoch)))
        # model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from epoch {}'.format(start_epoch))
    else:
        start_epoch = 0
        print('No trained models')
        exit(-1)
        
    valid_epoch_loss, valid_kl_epoch_loss, valid_ce_epoch_loss, wmse, ssim, iou, fps = validate(
        model, dev_dataloader, dev_dataset, device, criterion, coop_mode
    )
            
    print('Val Loss: {:.4f}, val WMSE: {}, val SSIM: {}, val IOU: {}, val FPS: {}'.format(valid_epoch_loss, wmse, ssim, iou, fps))
            
    return True
    
if __name__ == '__main__':
    main()