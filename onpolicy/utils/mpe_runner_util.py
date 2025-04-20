import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import io
import imageio
    
def ego_motion_transform(ego_ogm, translation_2d):
    """
    Transform occupancy grid maps (OGM) according to the ego-motion between frames.
    
    Parameters:
    - ego_ogm: numpy array of shape (batch_size, t, H, W) - OGM for each time step
    - translation_2d: numpy array of shape (batch_size, t, 2) - translations (dx, dy)
    
    Returns:
    - rob_ogm_transformed: numpy array of shape (batch_size, t, H, W) - transformed OGM
    """
    batch_size, t, H, W = ego_ogm.shape
    
    # Create a meshgrid for all pixels
    x = np.arange(H)
    y = np.arange(W)
    yy, xx = np.meshgrid(y, x)  # Shape: (H, W)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    
    # Create array to store transformed OGMs
    rob_ogm_transformed = np.zeros_like(ego_ogm)
    
    for b in range(batch_size):
        for time_step in range(t):
            # Extract translations for this batch and time
            dx, dy = translation_2d[b, time_step]
            
            # Calculate new coordinates after translation
            x_transformed = xx_flat - dx
            y_transformed = yy_flat - dy
            
            # Reshape and convert coordinates
            x_transformed = x_transformed.reshape(H, W).round().astype(int)
            y_transformed = y_transformed.reshape(H, W).round().astype(int)
            
            # Ensure transformed coordinates are within valid bounds
            valid_mask = (x_transformed >= 0) & (x_transformed < H) & (y_transformed >= 0) & (y_transformed < W)
            
            # Extract valid coordinates
            valid_x = x_transformed[valid_mask]
            valid_y = y_transformed[valid_mask]
            src_x = xx[valid_mask]
            src_y = yy[valid_mask]
            
            # Map valid source coordinates to target coordinates in the transformed OGM
            rob_ogm_transformed[b, time_step, valid_x, valid_y] = ego_ogm[b, time_step, src_x, src_y]

    return rob_ogm_transformed

def transform_lidar_to_ogm(ego_fused_lidar, map_size=100, resolution=0.1):
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
    batch_size, t, num_ray, num_agents = ego_fused_lidar.shape

    # Generate angles for each lidar ray (assuming a 360-degree scan)
    angles = np.linspace(0, 2 * np.pi, num_ray)  # Shape: (num_ray,)

    # Initialize the occupancy grid map with -1 (unknown)
    ogm = np.full((batch_size, t, map_size, map_size), -1, dtype=np.float32)

    # Grid cell resolution
    cell_length = resolution
    center_index = map_size // 2  # Ego vehicle is at the center of the grid
    
    for b in range(batch_size):
        for a in range(t):
            for i in range(num_agents):
                # Get lidar distances for current agent at this timestep
                rob_lidar = ego_fused_lidar[b, a, :, i]  # Shape: (num_ray,)

                # Compute x and y displacements using trigonometry
                distance_x = rob_lidar * np.cos(angles)  # Shape: (num_ray,)
                distance_y = rob_lidar * np.sin(angles)  # Shape: (num_ray,)

                # Filter out points with very small distances (avoid noise)
                valid_mask = rob_lidar > 0
                # valid_mask = rob_lidar > 0.2

                # Convert distances to grid indices
                x_indices = (distance_x[valid_mask] / cell_length).astype(int) + center_index
                y_indices = (distance_y[valid_mask] / cell_length).astype(int) + center_index

                # Ensure indices are within grid boundaries
                valid_x = (x_indices >= 0) & (x_indices < map_size)
                valid_y = (y_indices >= 0) & (y_indices < map_size)

                valid_indices = valid_x & valid_y  # Combine masks

                # Set occupied cells (1) in the occupancy grid
                ogm[b, a, x_indices[valid_indices], y_indices[valid_indices]] = 1

    # Convert remaining -1 values to 0 (free space)
    ogm[ogm == -1] = 0
    
    return ogm
    
def transform_point_clouds(other_lidar, translation_2d, is_current=False):
    """
    transforming the second cloud to the first agent's reference frame.
    
    Parameters:
    - other_lidar: numpy array of shape (n_rollout_threads, t, 360) - lidar readings from other agent
    - translation_2d: numpy array of shape (n_rollout_threads, t, 2) - translation vectors from ego to other
    
    Returns:
    - transformed_lidar: list-based structure of shape (n_rollout_threads, t, 360, variable_length)
      where each angle index stores multiple points dynamically.
    """
    n_rollout_threads, t, n_angles = other_lidar.shape

    transformed_lidar = np.zeros((n_rollout_threads, t, n_angles))

    # Convert lidar readings to Cartesian coordinates
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    for thread_idx in range(n_rollout_threads):
        for time_idx in range(t):
            if is_current:
                tx, ty = translation_2d[thread_idx, t-1]
            else:
                tx, ty = translation_2d[thread_idx, time_idx]  # Vector from ego to other

            # Convert other agent's lidar readings to Cartesian coordinates
            robot_points_x = other_lidar[thread_idx, time_idx] * np.cos(angles)
            robot_points_y = other_lidar[thread_idx, time_idx] * np.sin(angles)

            # Mask valid points
            valid_points_mask = (other_lidar[thread_idx, time_idx] > 0)
            valid_robot_points_x = robot_points_x[valid_points_mask]
            valid_robot_points_y = robot_points_y[valid_points_mask]
            valid_angles = angles[valid_points_mask]

            # Transform to ego's coordinate system
            ego_points_x = valid_robot_points_x - tx
            ego_points_y = valid_robot_points_y - ty

            # Convert to polar coordinates
            ego_points_r = np.sqrt(ego_points_x**2 + ego_points_y**2)
            ego_points_angles = np.arctan2(ego_points_y, ego_points_x)

            # Apply valid range filter (0.5m < distance < 5m)
            # valid_ego_points_mask = (0.5 < ego_points_r) & (ego_points_r <= 5)
            # valid_ego_points_mask = (0.2 < ego_points_r) & (ego_points_r <= 7)
            # ego_points_r = ego_points_r[valid_ego_points_mask]
            # ego_points_angles = ego_points_angles[valid_ego_points_mask]

            # Normalize angles to match lidar bins
            ego_points_angles_normalized = np.mod(ego_points_angles, 2 * np.pi)
            ego_points_indices = np.floor(ego_points_angles_normalized / (2 * np.pi / n_angles)).astype(int)
            ego_points_indices = np.mod(ego_points_indices, n_angles)  # Ensure indices are within bounds
            
            # Append valid points to the fused lidar structure
            for i, index in enumerate(ego_points_indices):
                transformed_lidar[thread_idx][time_idx][index] = ego_points_r[i]

    return transformed_lidar

def fuse_ogm(ego_ogm, rob_ogm, translation_2d, is_prediction=False):
    """
    Perform batched transformation and fusion of occupancy grids.
    """
    batch_size, t, H, W = ego_ogm.shape
    
    # Create a meshgrid for all pixels
    x = np.arange(H)
    y = np.arange(W)
    yy, xx = np.meshgrid(y, x)  # Shape: (H, W)

    # Flatten the grid
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    
    # Create homogeneous coordinates
    coords = np.stack([xx_flat, yy_flat, np.ones_like(xx_flat)], axis=0)  # Shape: (3, H*W)
    coords = np.expand_dims(coords, axis=0)  # Shape: (1, 3, H*W)
    coords = np.repeat(coords, batch_size, axis=0)  # Shape: (B, 3, H*W)

    # Apply batch-wise translation
    translation_homo = np.eye(3)  # Identity matrix (3x3)
    translation_homo = np.expand_dims(translation_homo, axis=0) 
    translation_homo = np.repeat(translation_homo, batch_size, axis=0)
    translation_homo[:, :2, 2] = translation_2d  # Add translation

    transformed_coords = translation_homo @ coords  # Shape: (B, 3, H*W)
    x_transformed = transformed_coords[:, 0, :].reshape(batch_size, H, W).round().astype(int)
    y_transformed = transformed_coords[:, 1, :].reshape(batch_size, H, W).round().astype(int)

    # Ensure transformed coordinates are within valid bounds
    valid_mask = (x_transformed >= 0) & (x_transformed < H) & (y_transformed >= 0) & (y_transformed < W)

    rob_ogm_transformed = np.zeros_like(ego_ogm)  # Shape: (batch, t, H, W)

    for b in range(batch_size):
        for time_step in range(t):  # Handle multiple time steps
            valid_x = x_transformed[b][valid_mask[b]]
            valid_y = y_transformed[b][valid_mask[b]]

            # Extract original indices correctly using flattening
            valid_indices = valid_mask[b].flatten()
            src_x = xx_flat[valid_indices]
            src_y = yy_flat[valid_indices]
            
            rob_ogm_transformed[b, time_step, valid_x, valid_y] = rob_ogm[b, time_step, src_x, src_y]

            # if not is_prediction:
            #     # rob_ogm_transformed[b, time_step, 30:33, 30:33] = 0  # Set the center region to 0
            #     rob_ogm_transformed[b, time_step, 29:35, 29:35] = 0  # Set the center region to 0
    
    # if is_prediction:
    #     pass

    fused_ogm = np.maximum(ego_ogm, rob_ogm_transformed)
    # if not is_prediction:
    # fused_ogm[:, :, 29:35, 29:35] = 0  # Set the center region to 0
    return fused_ogm  # Element-wise sum

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

def create_frame_in_memory(robot_vec, human_vec, robot_num=1, human_num=12):

    # 解析机器人
    robots_data = []
    for i in range(robot_num):
        # offset = i * 9
        offset = i * 6
        x_pos = robot_vec[offset + 0]
        y_pos = robot_vec[offset + 2]
        # theta = robot_vec[offset + 4]
        # x_goal = robot_vec[offset + 6]
        x_goal = robot_vec[offset + 3]
        # y_goal = robot_vec[offset + 8]
        y_goal = robot_vec[offset + 5]
        robots_data.append({
            "pos_x": x_pos,
            "pos_y": y_pos,
            # "theta": theta,
            "goal_x": x_goal,
            "goal_y": y_goal
        })

    # 解析8个Human(只用 x,y)
    humans_data = []
    for i in range(human_num):
        offset = i * 9
        # offset = i * 6
        hx = human_vec[offset + 0]
        hy = human_vec[offset + 2]
        theta = human_vec[offset + 4]
        humans_data.append({"pos_x": hx, "pos_y": hy, "theta": theta})

    # 创建图
    fig, ax = plt.subplots(figsize=(5, 5), dpi=60)
    ax.set_xlim(5, 18.5)
    ax.set_ylim(-6, 13)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(5, 18.5, 2))
    ax.set_yticks(np.arange(-6, 13, 2))

    robot_colors = ["#ffd079", "#96cbfd", "#ff0000"]
    # robot_colors = ["#ffd079", "#96cbfd", "#ff0000", "#ff6600", "#073f93", "#f56200", "#55ff55", "#000000"]
    arrow_colors = ["#ff6600", "#073f93", "#f56200"]
    # arrow_colors = ["#ff6600", "#073f93", "#f56200", "#55ff55", "#000000", "#ff0000", "#96cbfd", "#ffd079"]
    human_color = "#55ff55"
    human_arrow_color = "#ff0099"
    # goal_color = "#ff0000"
    goal_colors = ["#ffd079", "#96cbfd", "#ff0000"]
    # goal_colors = ["#ffd079", "#96cbfd", "#ff0000", "#ff6600", "#073f93", "#f56200", "#55ff55", "#000000"]
    arrow_style = patches.ArrowStyle("simple", head_length=6, head_width=5)

    # 从指定地址读取背景图
    # bg_img = plt.imread("./library_bg_no_obs.png")
    bg_img = plt.imread("./library_bg.png")
    ax.imshow(bg_img, extent=[5, 18.5, -6, 13])


    # 机器人
    for i, rinfo in enumerate(robots_data):
        rx, ry = rinfo["pos_x"], rinfo["pos_y"]
        # rtheta = rinfo["theta"]
        gx, gy = rinfo["goal_x"], rinfo["goal_y"]

        circle = plt.Circle((rx, ry), radius=0.2, color=robot_colors[i], alpha=1)
        ax.add_patch(circle)

        # 朝向箭头
        # arrow_length = 0.5
        # rtheta = np.radians(rtheta)
        # arrow_dx = arrow_length * np.sin(rtheta)
        # arrow_dy = arrow_length * np.cos(rtheta)
        # arrow_patch = patches.FancyArrowPatch(
        #     (rx, ry),
        #     (rx + arrow_dx, ry + arrow_dy),
        #     color=arrow_colors[i],
        #     arrowstyle=arrow_style
        # )
        # ax.add_patch(arrow_patch)

        # 目标点
        goal_marker = mlines.Line2D([gx], [gy], color=goal_colors[i], marker='*',
                                    linestyle='None', markersize=10)
        ax.add_line(goal_marker)

        # ax.text(rx + 0.2, ry + 0.2, f"R{i}", fontsize=8, color="black")

    # Human
    for i, hinfo in enumerate(humans_data):
        hx = hinfo["pos_x"]
        hy = hinfo["pos_y"]
        htheta = hinfo["theta"]

        arrow_length = 0.7
        htheta = np.radians(htheta)
        arrow_dx = arrow_length * np.sin(htheta)
        arrow_dy = arrow_length * np.cos(htheta)
        arrow_patch = patches.FancyArrowPatch(
            (hx, hy),
            (hx + arrow_dx, hy + arrow_dy),
            color=human_arrow_color,
            arrowstyle=arrow_style
        )
        ax.add_patch(arrow_patch)

        h_circle = plt.Circle((hx, hy), radius=0.25, color=human_color, alpha=0.9)
        ax.add_patch(h_circle)
        # ax.text(hx + 0.15, hy + 0.15, f"H{i}", fontsize=7, color="black")

    # 把图保存到内存
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    # 读回 np.array
    frame_img = imageio.imread(buf)
    return frame_img
    
def to_prob_occ_map(ego_ogm, device):
    
    batch_size, t, H, W = ego_ogm.shape

    threshold_p_occ = 0.8
    P_prior = 0.5
    P_free = 0.3
    P_occ = 0.7

    occ_map = torch.full((batch_size, t, H, W), fill_value = log_odds(P_prior)).to(device)
    occ_map[ego_ogm==0] += log_odds(P_free)
    occ_map[ego_ogm==1] += log_odds(P_occ)

    log_map = torch.sum(occ_map, dim=1)  # sum of all timestep maps
    prob_map = retrieve_p(log_map)
    prob_map = calc_MLE(prob_map, threshold_p_occ)

    return prob_map


def expand_uncertain_areas(ego_ogm_pred, ego_ogm_pred_entropy, thresholds=(0.5, 0.8), kernel_sizes=(1, 1)):
    
    t1, t2 = thresholds
    mask_tierA = (ego_ogm_pred_entropy >= t1) & (ego_ogm_pred_entropy < t2)  # medium
    mask_tierB = (ego_ogm_pred_entropy >= t2)                         #  high

    # Helper for morphological dilation via max-pooling
    def batch_dilate(mask_bool_torch, radius: int):
        """
        Morphologically dilate a boolean mask of shape (N, T, H, W) 
        using 2D max-pooling in a batched manner. 
        radius => kernel_size = 2*radius + 1
        """
        if radius <= 0:
            return mask_bool_torch
        
        k_size = 2 * radius + 1
        pool = nn.MaxPool2d(kernel_size=k_size, stride=1, padding=radius)

        N, T, H, W = mask_bool_torch.shape
        # Flatten (N,T) => (N*T) batch, because PyTorch pooling expects [B, C, H, W]
        x = mask_bool_torch.view(N*T, 1, H, W).float()
        
        # Max pooling => morphological dilation for binary inputs
        x_dilated = pool(x)
        
        # Reshape back to (N, T, H, W) and convert to boolean
        x_dilated = x_dilated.view(N, T, H, W)
        return x_dilated > 0.5

    # 3. Dilation for each tier
    #    (C is the highest tier, B is next, A is lowest).
    #    We do them separately so we can *overlay* them with priorities.
    expanded_mask_tierA = batch_dilate(mask_tierA, radius=kernel_sizes[0])  # Medium
    expanded_mask_tierB = batch_dilate(mask_tierB, radius=kernel_sizes[1])  # High

    # 4. Mark expanded regions by tier in the final occupancy
    #    Highest tier should overwrite lower ones in case of overlap.
    #    We do it in descending order of severity:
    #
    #    => Tier B (0.8) has highest priority
    ego_ogm_pred[expanded_mask_tierB] = 0.8

    #    => Tier A (0.5) applies only where Tier C didn't apply
    ego_ogm_pred[expanded_mask_tierA & ~expanded_mask_tierB] = 0.5

    # 5. Convert back to NumPy
    updated_ogm_pred = ego_ogm_pred.detach().cpu().numpy()

    return updated_ogm_pred