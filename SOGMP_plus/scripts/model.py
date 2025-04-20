#!/usr/bin/env python
#
# file: $ISIP_EXP/SOGMP/scripts/model.py
#
# revision history: xzt
#  20220824 (TE): first version
#
# usage:
#
# This script hold the model architecture
#------------------------------------------------------------------------------

# import pytorch modules
#
from __future__ import print_function
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from SOGMP_plus.scripts.convlstm import ConvLSTMCell

import os
import random
import glob

from onpolicy.utils.mpe_runner_util import transform_point_clouds
from onpolicy.utils.mpe_runner_util import transform_lidar_to_ogm as transform_fused_lidar_to_ogm

# for reproducibility, we seed the rng
SEED1 = 1337
NEW_LINE = "\n"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------------------------------
#
# helper functions are listed here
#
#-----------------------------------------------------------------------------

# function: set_seed
#
# arguments: seed - the seed for all the rng
#
# returns: none
#
# this method seeds all the random number generators and makes
# the results deterministic
#
def set_seed(seed):
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
#
# end of method


# function: get_data
#
# arguments: img_path - file pointer
#            file_name - the name of data file
#
# returns: data - the signals/features
#
# this method takes in a fp and returns the data and labels
POINTS = 1080   # the number of lidar points
MAP_SIZE = 64
RESOLUTION = 0.05
VIS_STEP = 5
FUTURE_STEP = 5
# FUTURE_STEP = 10
TOTAL_LEN = VIS_STEP + FUTURE_STEP

def transform_lidar_to_ogm(lidar_scan, map_size=100):
    """
    Converts lidar distance measurements into an occupancy grid map (OGM) using only NumPy.

    Parameters:
    - ego_fused_lidar: np.ndarray of shape (batch_size, t, num_agents, num_ray)
      Lidar distance readings for each batch, time step, agent, and ray.
    - map_size: int, size of the square occupancy grid (default: 100).

    Returns:
    - ogm: np.ndarray of shape (batch_size, t, num_agents, map_size, map_size)
      The computed occupancy grid map.
    """
    batch_size, num_ray = lidar_scan.shape

    # Generate angles for each lidar ray (assuming a 360-degree scan)
    angles = np.linspace(0, 2 * np.pi, num_ray)  # Shape: (num_ray,)

    # Initialize the occupancy grid map with -1 (unknown)
    ogm = np.full((batch_size, map_size, map_size), -1, dtype=np.float32)

    # Grid cell resolution
    cell_length = RESOLUTION
    center_index = map_size // 2  # Ego vehicle is at the center of the grid
    
    for b in range(batch_size):
        # Get lidar distances for current agent at this timestep
        rob_lidar = lidar_scan[b]  # Shape: (num_ray,)

        # Compute x and y displacements using trigonometry
        distance_x = rob_lidar * np.cos(angles)  # Shape: (num_ray,)
        distance_y = rob_lidar * np.sin(angles)  # Shape: (num_ray,)

        # Filter out points with very small distances (avoid noise)
        # valid_mask = rob_lidar > 0.5
        valid_mask = rob_lidar > 0

        # Convert distances to grid indices
        x_indices = (distance_x[valid_mask] / cell_length).astype(int) + center_index
        y_indices = (distance_y[valid_mask] / cell_length).astype(int) + center_index

        # Ensure indices are within grid boundaries
        valid_x = (x_indices >= 0) & (x_indices < map_size)
        valid_y = (y_indices >= 0) & (y_indices < map_size)

        valid_indices = valid_x & valid_y  # Combine masks

        # Set occupied cells (1) in the occupancy grid
        ogm[b, x_indices[valid_indices], y_indices[valid_indices]] = 1
        # ogm[b, center_index, center_index] = 1

    # Convert remaining -1 values to 0 (free space)
    ogm[ogm == -1] = 0
    
    return ogm
    
def ego_motion_transform_lidar_data(scans, positions):
    """
    Transform lidar data into the ego-agent's coordinate system.
    """
    transformed_lidar = np.zeros_like(scans)
    _, n_points = scans.shape

    # Compute translation to ego's frame
    translation_2d = np.stack([
        positions[VIS_STEP - 1, 1] - positions[:, 1],  # y translation
        positions[VIS_STEP - 1, 0] - positions[:, 0]   # x translation
    ], axis=-1)

    for i in range(TOTAL_LEN):

        tx, ty = translation_2d[i]

        # Extract valid points
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        robot_points_x = scans[i] * np.cos(angles)
        robot_points_y = scans[i] * np.sin(angles)
        valid_mask = scans[i] > 0

        # Transform valid points
        valid_robot_points_x = robot_points_x[valid_mask]
        valid_robot_points_y = robot_points_y[valid_mask]
        ego_points_x = valid_robot_points_x - tx
        ego_points_y = valid_robot_points_y - ty

        # Compute polar coordinates in ego frame
        ego_points_r = np.sqrt(ego_points_x**2 + ego_points_y**2)
        ego_points_angles = np.arctan2(ego_points_y, ego_points_x)

        # Apply range and angle normalization
        valid_range_mask = (0.3 < ego_points_r) & (ego_points_r <= 7)
        ego_points_r = ego_points_r[valid_range_mask]
        ego_points_angles = ego_points_angles[valid_range_mask]
        ego_points_indices = np.floor(
            np.mod(ego_points_angles, 2 * np.pi) / (2 * np.pi / n_points)
        ).astype(int)

        # Assign transformed points to lidar bins
        for j, index in enumerate(ego_points_indices):
            transformed_lidar[i, index] = ego_points_r[j]

    return transformed_lidar

def transform_surrounding_lidar(ego_positions, surround_positions, surround_scans):
    """
    Transform the surrounding agent's lidar data into the ego-agent's coordinate system.
    """
    transformed_lidar = np.zeros_like(surround_scans)
    _, n_points = surround_scans.shape

    translation_2d = np.stack([
        ego_positions[VIS_STEP - 1, 1] - surround_positions[VIS_STEP - 1, 1],  # y translation
        ego_positions[VIS_STEP - 1, 0] - surround_positions[VIS_STEP - 1, 0]   # x translation
    ], axis=-1)

    for i in range(TOTAL_LEN):
        # Compute translation to ego's frame
        tx, ty = translation_2d

        # Extract valid points
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        robot_points_x = surround_scans[i] * np.cos(angles)
        robot_points_y = surround_scans[i] * np.sin(angles)
        valid_mask = surround_scans[i] > 0

        # Transform valid points
        valid_robot_points_x = robot_points_x[valid_mask]
        valid_robot_points_y = robot_points_y[valid_mask]
        ego_points_x = valid_robot_points_x - tx
        ego_points_y = valid_robot_points_y - ty

        # Compute polar coordinates in ego frame
        ego_points_r = np.sqrt(ego_points_x**2 + ego_points_y**2)
        ego_points_angles = np.arctan2(ego_points_y, ego_points_x)

        # Apply range and angle normalization
        valid_range_mask = (0.3 < ego_points_r) & (ego_points_r <= 7)
        ego_points_r = ego_points_r[valid_range_mask]
        ego_points_angles = ego_points_angles[valid_range_mask]
        ego_points_indices = np.floor(
            np.mod(ego_points_angles, 2 * np.pi) / (2 * np.pi / n_points)
        ).astype(int)

        # Assign transformed points to lidar bins
        for j, index in enumerate(ego_points_indices):
            transformed_lidar[i, index] = ego_points_r[j]

    return transformed_lidar

class VaeDatasetNoFusion(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train=True):
        # initialize the data and labels

        self.all_lidar_data = {}
        self.all_pose_data = {}
        self.all_traj_length = {}
        self.index_map = []
        
        data_mode = "" if is_train else "_val"
        
        for agent_id in range(3):
            lidar_list, pose_list, traj_list = [], [], []

            # Find all chunk files dynamically
            lidar_paths = sorted(glob.glob(os.path.join(
                data_path, f"all_lidar_data_agent_{agent_id}{data_mode}_chunk_*.npy"
            )))
            pose_paths = sorted(glob.glob(os.path.join(
                data_path, f"all_pos_data_agent_{agent_id}{data_mode}_chunk_*.npy"
            )))
            traj_paths = sorted(glob.glob(os.path.join(
                data_path, f"all_traj_length_agent_{agent_id}{data_mode}_chunk_*.npy"
            )))

            # Safety check
            assert len(lidar_paths) == len(pose_paths) == len(traj_paths), \
                f"Mismatched chunk numbers for agent {agent_id}"

            for l_path, p_path, t_path in zip(lidar_paths, pose_paths, traj_paths):
                lidar_list.append(np.load(l_path))
                pose_list.append(np.load(p_path))
                traj_list.append(np.load(t_path))

            self.all_lidar_data[agent_id] = np.concatenate(lidar_list, axis=0)
            self.all_pose_data[agent_id] = np.concatenate(pose_list, axis=0)
            self.all_traj_length[agent_id] = np.concatenate(traj_list, axis=0)

            start_idx = 0
            for length in self.all_traj_length[agent_id]:
                # Only add indices if there are enough points for a full sequence
                for i in range(start_idx, start_idx + length - TOTAL_LEN + 1):
                    self.index_map.append((agent_id, i))
                start_idx += length

        self.length = len(self.index_map)

    def __len__(self):
        return self.length

    def __getitem__(self, idx): # where is idx?
        # get the index of start point:
        # Validate idx
        if idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_map)}.")

        # Deconstruct index mapping
        ego_agent_id, local_idx = self.index_map[idx]
        end_idx = local_idx + TOTAL_LEN

        # Fetch ego-agent data
        ego_scans = self.all_lidar_data[ego_agent_id][local_idx:end_idx, :]
        ego_positions = self.all_pose_data[ego_agent_id][local_idx:end_idx]

        # Initialize transformed lidar array
        _, n_points = ego_scans.shape
        ego_transformed_lidar = np.zeros((TOTAL_LEN, n_points))

        # Transform ego-agent lidar scans
        ego_transformed_lidar = ego_motion_transform_lidar_data(ego_scans, ego_positions)

        # Generate ego-agent occupancy grid map (OGM)
        ego_transformed_ogm = transform_lidar_to_ogm(ego_transformed_lidar, map_size=MAP_SIZE)
        # transfer to pytorch tensor:
        ego_ogm_tensor = torch.FloatTensor(ego_transformed_ogm)
        
        data = {
            'ego_agent_id': ego_agent_id,
            'ego_ogm': ego_ogm_tensor
        }
        
        return data

class VaeTestDatasetNoFusion(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train=True):
        # initialize the data and labels

        self.all_lidar_data = {}
        self.all_pose_data = {}
        self.all_traj_length = {}
        self.index_map = []
        
        data_mode = "" if is_train else "_val"  

        for agent_id in range(3):
            lidar_list, pose_list, traj_list = [], [], []

            # Find all chunk files dynamically
            lidar_paths = sorted(glob.glob(os.path.join(
                data_path, f"all_lidar_data_agent_{agent_id}{data_mode}_chunk_*.npy"
            )))
            pose_paths = sorted(glob.glob(os.path.join(
                data_path, f"all_pos_data_agent_{agent_id}{data_mode}_chunk_*.npy"
            )))
            traj_paths = sorted(glob.glob(os.path.join(
                data_path, f"all_traj_length_agent_{agent_id}{data_mode}_chunk_*.npy"
            )))

            # Safety check
            assert len(lidar_paths) == len(pose_paths) == len(traj_paths), \
                f"Mismatched chunk numbers for agent {agent_id}"

            for l_path, p_path, t_path in zip(lidar_paths, pose_paths, traj_paths):
                lidar_list.append(np.load(l_path))
                pose_list.append(np.load(p_path))
                traj_list.append(np.load(t_path))

            self.all_lidar_data[agent_id] = np.concatenate(lidar_list, axis=0)
            self.all_pose_data[agent_id] = np.concatenate(pose_list, axis=0)
            self.all_traj_length[agent_id] = np.concatenate(traj_list, axis=0)

            start_idx = 0
            for length in self.all_traj_length[agent_id]:
                # Only add indices if there are enough points for a full sequence
                for i in range(start_idx, start_idx + length - TOTAL_LEN + 1):
                    self.index_map.append((agent_id, i))
                start_idx += length

        self.length = len(self.index_map)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Fetch and process data for the given index `idx`.
        """
        # Validate idx
        if idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_map)}.")

        # Deconstruct index mapping
        ego_agent_id, local_idx = self.index_map[idx]
        end_idx = local_idx + TOTAL_LEN

        # Fetch ego-agent data
        ego_scans = self.all_lidar_data[ego_agent_id][local_idx:end_idx, :]
        ego_positions = self.all_pose_data[ego_agent_id][local_idx:end_idx]

        # Initialize transformed lidar array
        _, n_points = ego_scans.shape
        ego_transformed_lidar = np.zeros((TOTAL_LEN, n_points))

        # Transform ego-agent lidar scans
        ego_transformed_lidar = ego_motion_transform_lidar_data(ego_scans, ego_positions)

        # Generate ego-agent occupancy grid map (OGM)
        ego_transformed_ogm = transform_lidar_to_ogm(ego_transformed_lidar, map_size=MAP_SIZE)

        # Initialize fused lidar data
        fused_lidar = np.zeros((1, TOTAL_LEN, n_points, 3))
        fused_lidar[0, :, :, ego_agent_id] = ego_transformed_lidar

        # Process surrounding agents
        for surround_agent_id in range(3):
            if surround_agent_id == ego_agent_id:
                continue

            # Fetch surrounding agent data
            surround_scans = self.all_lidar_data[surround_agent_id][local_idx:end_idx, :]
            surround_positions = self.all_pose_data[surround_agent_id][local_idx:end_idx]

            surround_transformed_lidar = ego_motion_transform_lidar_data(surround_scans, surround_positions)

            # Transform surrounding agent lidar data
            surround_transformed_lidar = transform_surrounding_lidar(
                ego_positions, surround_positions, surround_transformed_lidar
            )

            # Add to fused lidar
            fused_lidar[0, :, :, surround_agent_id] = surround_transformed_lidar

        # Convert fused lidar data to occupancy grid map
        transformed_ogm_fused = transform_fused_lidar_to_ogm(fused_lidar, map_size=MAP_SIZE)
        transformed_ogm_fused = np.squeeze(transformed_ogm_fused, axis=0)

        # Convert data to PyTorch tensors
        data = {
            'ego_agent_id': ego_agent_id,
            'ego_ogm': torch.FloatTensor(ego_transformed_ogm),
            'fused_ogm': torch.FloatTensor(transformed_ogm_fused)
        }

        return data
        
class VaeDatasetEarlyFusion(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train=True):
        # initialize the data and labels

        self.all_lidar_data = {}
        self.all_pose_data = {}
        self.all_traj_length = {}
        self.index_map = []
        
        data_mode = "" if is_train else "_val"

        for agent_id in range(3):
            lidar_list, pose_list, traj_list = [], [], []

            # Find all chunk files dynamically
            lidar_paths = sorted(glob.glob(os.path.join(
                data_path, f"all_lidar_data_agent_{agent_id}{data_mode}_chunk_*.npy"
            )))
            pose_paths = sorted(glob.glob(os.path.join(
                data_path, f"all_pos_data_agent_{agent_id}{data_mode}_chunk_*.npy"
            )))
            traj_paths = sorted(glob.glob(os.path.join(
                data_path, f"all_traj_length_agent_{agent_id}{data_mode}_chunk_*.npy"
            )))

            # Safety check
            assert len(lidar_paths) == len(pose_paths) == len(traj_paths), \
                f"Mismatched chunk numbers for agent {agent_id}"

            for l_path, p_path, t_path in zip(lidar_paths, pose_paths, traj_paths):
                lidar_list.append(np.load(l_path))
                pose_list.append(np.load(p_path))
                traj_list.append(np.load(t_path))

            self.all_lidar_data[agent_id] = np.concatenate(lidar_list, axis=0)
            self.all_pose_data[agent_id] = np.concatenate(pose_list, axis=0)
            self.all_traj_length[agent_id] = np.concatenate(traj_list, axis=0)

            start_idx = 0
            for length in self.all_traj_length[agent_id]:
                # Only add indices if there are enough points for a full sequence
                for i in range(start_idx, start_idx + length - TOTAL_LEN + 1):
                    self.index_map.append((agent_id, i))
                start_idx += length

        self.length = len(self.index_map)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Fetch and process data for the given index `idx`.
        """
        # Validate idx
        if idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_map)}.")

        # Deconstruct index mapping
        ego_agent_id, local_idx = self.index_map[idx]
        end_idx = local_idx + TOTAL_LEN

        # Fetch ego-agent data
        ego_scans = self.all_lidar_data[ego_agent_id][local_idx:end_idx, :]
        ego_positions = self.all_pose_data[ego_agent_id][local_idx:end_idx]

        # Initialize transformed lidar array
        _, n_points = ego_scans.shape
        ego_transformed_lidar = np.zeros((TOTAL_LEN, n_points))

        # Transform ego-agent lidar scans
        ego_transformed_lidar = ego_motion_transform_lidar_data(ego_scans, ego_positions)

        # Generate ego-agent occupancy grid map (OGM)
        ego_transformed_ogm = transform_lidar_to_ogm(ego_transformed_lidar, map_size=MAP_SIZE)

        # Initialize fused lidar data
        fused_lidar = np.zeros((1, TOTAL_LEN, n_points, 3))
        fused_lidar[0, :, :, ego_agent_id] = ego_transformed_lidar

        # Process surrounding agents
        for surround_agent_id in range(3):
            if surround_agent_id == ego_agent_id:
                continue

            # Fetch surrounding agent data
            surround_scans = self.all_lidar_data[surround_agent_id][local_idx:end_idx, :]
            surround_positions = self.all_pose_data[surround_agent_id][local_idx:end_idx]

            surround_transformed_lidar = ego_motion_transform_lidar_data(surround_scans, surround_positions)

            # Transform surrounding agent lidar data
            surround_transformed_lidar = transform_surrounding_lidar(
                ego_positions, surround_positions, surround_transformed_lidar
            )

            # Add to fused lidar
            fused_lidar[0, :, :, surround_agent_id] = surround_transformed_lidar

        # Convert fused lidar data to occupancy grid map
        transformed_ogm_fused = transform_fused_lidar_to_ogm(fused_lidar, map_size=MAP_SIZE)
        transformed_ogm_fused = np.squeeze(transformed_ogm_fused, axis=0)

        # Convert data to PyTorch tensors
        data = {
            'ego_agent_id': ego_agent_id,
            'ego_ogm': torch.FloatTensor(ego_transformed_ogm),
            'fused_ogm': torch.FloatTensor(transformed_ogm_fused)
        }

        return data

class VaeTestDatasetLateFusion(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train=True):
        # initialize the data and labels

        self.all_lidar_data = {}
        self.all_pose_data = {}
        self.all_traj_length = {}
        self.index_map = []

        data_mode = "" if is_train else "_val"
        
        for agent_id in range(3):
            lidar_list, pose_list, traj_list = [], [], []

            # Find all chunk files dynamically
            lidar_paths = sorted(glob.glob(os.path.join(
                data_path, f"all_lidar_data_agent_{agent_id}{data_mode}_chunk_*.npy"
            )))
            pose_paths = sorted(glob.glob(os.path.join(
                data_path, f"all_pos_data_agent_{agent_id}{data_mode}_chunk_*.npy"
            )))
            traj_paths = sorted(glob.glob(os.path.join(
                data_path, f"all_traj_length_agent_{agent_id}{data_mode}_chunk_*.npy"
            )))

            # Safety check
            assert len(lidar_paths) == len(pose_paths) == len(traj_paths), \
                f"Mismatched chunk numbers for agent {agent_id}"

            for l_path, p_path, t_path in zip(lidar_paths, pose_paths, traj_paths):
                lidar_list.append(np.load(l_path))
                pose_list.append(np.load(p_path))
                traj_list.append(np.load(t_path))

            self.all_lidar_data[agent_id] = np.concatenate(lidar_list, axis=0)
            self.all_pose_data[agent_id] = np.concatenate(pose_list, axis=0)
            self.all_traj_length[agent_id] = np.concatenate(traj_list, axis=0)

            start_idx = 0
            for length in self.all_traj_length[agent_id]:
                # Only add indices if there are enough points for a full sequence
                for i in range(start_idx, start_idx + length - TOTAL_LEN + 1):
                    self.index_map.append((agent_id, i))
                start_idx += length

        self.length = len(self.index_map)

    def __len__(self):
        return self.length

    def __getitem__(self, idx): # where is idx?
        """
        Fetch and process data for the given index `idx`.
        """
        # Validate idx
        if idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_map)}.")

        # Deconstruct index mapping
        ego_agent_id, local_idx = self.index_map[idx]
        end_idx = local_idx + TOTAL_LEN

        # Fetch ego-agent data
        ego_scans = self.all_lidar_data[ego_agent_id][local_idx:end_idx, :]
        ego_positions = self.all_pose_data[ego_agent_id][local_idx:end_idx]

        # Initialize transformed lidar array
        _, n_points = ego_scans.shape
        ego_transformed_lidar = np.zeros((TOTAL_LEN, n_points))

        # Transform ego-agent lidar scans
        ego_transformed_lidar = ego_motion_transform_lidar_data(ego_scans, ego_positions)

        # Generate ego-agent occupancy grid map (OGM)
        ego_transformed_ogm = transform_lidar_to_ogm(ego_transformed_lidar, map_size=MAP_SIZE)
        sourround_transformed_ogm = np.zeros((3, TOTAL_LEN, MAP_SIZE, MAP_SIZE))
        rob_translation_2d = np.zeros((3, 2))

        # Initialize fused lidar data
        fused_lidar = np.zeros((1, TOTAL_LEN, n_points, 3))
        fused_lidar[0, :, :, ego_agent_id] = ego_transformed_lidar

        # Process surrounding agents
        for surround_agent_id in range(3):
            if surround_agent_id == ego_agent_id:
                continue

            # Fetch surrounding agent data
            surround_scans = self.all_lidar_data[surround_agent_id][local_idx:end_idx, :]
            surround_positions = self.all_pose_data[surround_agent_id][local_idx:end_idx]

            surround_transformed_lidar = ego_motion_transform_lidar_data(surround_scans, surround_positions)
            sourround_transformed_ogm[surround_agent_id] = transform_lidar_to_ogm(surround_transformed_lidar, map_size=MAP_SIZE)
            rob_translation_2d[surround_agent_id] = np.stack([
                (surround_positions[VIS_STEP - 1, 1] - ego_positions[VIS_STEP - 1, 1]) / RESOLUTION,  # y translation
                (surround_positions[VIS_STEP - 1, 0] - ego_positions[VIS_STEP - 1, 0]) / RESOLUTION   # x translation
            ], axis=-1)

            # Transform surrounding agent lidar data 
            surround_transformed_lidar = transform_surrounding_lidar(
                ego_positions, surround_positions, surround_transformed_lidar
            )

            # Add to fused lidar
            fused_lidar[0, :, :, surround_agent_id] = surround_transformed_lidar

        # Convert fused lidar data to occupancy grid map
        transformed_ogm_fused = transform_fused_lidar_to_ogm(fused_lidar, map_size=MAP_SIZE)
        transformed_ogm_fused = np.squeeze(transformed_ogm_fused, axis=0)

        # Convert data to PyTorch tensors
        data = {
            'ego_agent_id': ego_agent_id,
            'ego_ogm': torch.FloatTensor(ego_transformed_ogm),
            'fused_ogm': torch.FloatTensor(transformed_ogm_fused),
            'rob_ogms': torch.FloatTensor(sourround_transformed_ogm),
            'rob_translation_2d': rob_translation_2d
        }

        return data

#
# end of function


#------------------------------------------------------------------------------
#
# the model is defined here
#
#------------------------------------------------------------------------------

# define the PyTorch VAE model
#
# define a VAE
# Residual blocks: 
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_hiddens)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

# Encoder & Decoder Architecture:
# Encoder:
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=in_channels,
                                                  out_channels=num_hiddens//2,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens//2),
                                        nn.ReLU()
                                    ])
        self._conv_2 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=num_hiddens//2,
                                                  out_channels=num_hiddens,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens)
                                        #nn.ReLU(True)
                                    ])
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._conv_2(x)
        x = self._residual_stack(x)
        return x

# Decoder:
class Decoder(nn.Module):
    def __init__(self, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_2 = nn.Sequential(*[
                                            nn.ReLU(),
                                            nn.ConvTranspose2d(in_channels=num_hiddens,
                                                              out_channels=num_hiddens//2,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1),
                                            nn.BatchNorm2d(num_hiddens//2),
                                            nn.ReLU()
                                        ])

        self._conv_trans_1 = nn.Sequential(*[
                                            nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                              out_channels=num_hiddens//2,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1),
                                            nn.BatchNorm2d(num_hiddens//2),
                                            nn.ReLU(),                  
                                            nn.Conv2d(in_channels=num_hiddens//2,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1),
                                            nn.Sigmoid()
                                        ])
        


    def forward(self, inputs):
        x = self._residual_stack(inputs)
        x = self._conv_trans_2(x)
        x = self._conv_trans_1(x)
        return x

class VAE_Encoder(nn.Module):
    def __init__(self, input_channel):
        super(VAE_Encoder, self).__init__()
        # parameters:
        self.input_channels = input_channel
        # Constants
        num_hiddens = 128 #128
        num_residual_hiddens = 64 #32
        num_residual_layers = 2
        embedding_dim = 2 #64

        # encoder:
        in_channels = input_channel
        self._encoder = Encoder(in_channels, 
                                num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)

        # z latent variable: 
        self._encoder_z_mu = nn.Conv2d(in_channels=num_hiddens, 
                                    out_channels=embedding_dim,
                                    kernel_size=1, 
                                    stride=1)
        self._encoder_z_log_sd = nn.Conv2d(in_channels=num_hiddens, 
                                    out_channels=embedding_dim,
                                    kernel_size=1, 
                                    stride=1)  
        
    def forward(self, x):
        # input reshape:
        x = x.reshape(-1, self.input_channels, MAP_SIZE, MAP_SIZE)
        # Encoder:
        encoder_out = self._encoder(x)
        # get `mu` and `log_var`:
        z_mu = self._encoder_z_mu(encoder_out)
        z_log_sd = self._encoder_z_log_sd(encoder_out)
        return z_mu, z_log_sd

# our proposed model: SOGMP++
class RVAEP(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels):
        super(RVAEP, self).__init__()
        # parameters:
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.z_w = int(np.sqrt(latent_dim//2))
        
        # Constants
        num_hiddens = 128 #64
        num_residual_hiddens = 64 #64
        num_residual_layers = 2
        embedding_dim = 2 
        
        # prediction encoder:
        self._convlstm = ConvLSTMCell(input_dim=self.input_channels,
                                    hidden_dim=num_hiddens//4,
                                    kernel_size=(3, 3),
                                    bias=True)
        
        self._encoder = VAE_Encoder((num_hiddens//4 + self.input_channels),) # num_hiddens//4 + self.input_channels

        # decoder:
        self._decoder_z_mu = nn.ConvTranspose2d(in_channels=embedding_dim, 
                                    out_channels=num_hiddens,
                                    kernel_size=1, 
                                    stride=1)
        self._decoder = Decoder(self.output_channels,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)
                                
    def vae_reparameterize(self, z_mu, z_log_sd):
        """
        :param mu: mean from the encoder's latent space
        :param log_sd: log standard deviation from the encoder's latent space
        :output: reparameterized latent variable z, Monte carlo KL divergence
        """
        # reshape:
        z_mu = z_mu.reshape(-1, self.latent_dim, 1)
        z_log_sd = z_log_sd.reshape(-1, self.latent_dim, 1)
        # define the z probabilities (in this case Normal for both)
        # p(z): N(z|0,I)
        pz = torch.distributions.Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_log_sd))
        # q(z|x,phi): N(z|mu, z_var)
        qz_x = torch.distributions.Normal(loc=z_mu, scale=torch.exp(z_log_sd))

        # repameterization trick: z = z_mu + xi (*) z_log_var, xi~N(xi|0,I)
        z = qz_x.rsample()
        # Monte Carlo KL divergence: MCKL(p(z)||q(z|x,phi)) = log(p(z)) - log(q(z|x,phi))
        # sum over weight dim, leaves the batch dim 
        kl_divergence = (pz.log_prob(z) - qz_x.log_prob(z)).sum(dim=1)
        kl_loss = -kl_divergence.mean()

        return z, kl_loss

    def forward(self, x, x_map, pos=None, ego_index=None, fusion='no'):
        
        """
        Forward pass `input_img` through the network
        """
        # reconstruction: 
        # encode:
        # input reshape:

        x = x.reshape(-1, VIS_STEP, 1, MAP_SIZE, MAP_SIZE)
        x_map = x_map.reshape(-1, 1, MAP_SIZE, MAP_SIZE)

        # find size of different input dimensions
        b, seq_len, c, h, w = x.size()
            
        # encode: 
        # initialize hidden states
        h_enc, enc_state = self._convlstm.init_hidden(batch_size=b, image_size=(h, w))
        for t in range(seq_len): 
            x_in = x[:,t]
            h_enc, enc_state = self._convlstm(input_tensor=x_in,
                                            cur_state=[h_enc, enc_state])
        
        enc_in = torch.cat([h_enc, x_map], dim=1)
        
        z_mu, z_log_sd = self._encoder(enc_in)

        # get the latent vector through reparameterization:
        z, kl_loss = self.vae_reparameterize(z_mu, z_log_sd)
        
        # decode:
        # reshape:
        z = z.reshape(-1, 2, self.z_w, self.z_w)

        x_d = self._decoder_z_mu(z)
        
        prediction = self._decoder(x_d)

        return prediction, kl_loss
            
# end of class

# end of file
