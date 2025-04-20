import os
import time
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import imageio
from datetime import datetime
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.utils.mpe_runner_util import *
from SOGMP_plus.scripts.dtaci_grid import DtACIGrid

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def run(self):
        self.warmup()   
        
        total_start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        if self.collect_data:
            all_lidar_data = [[] for _ in range(self.num_agents)]
            all_pos_data = [[] for _ in range(self.num_agents)]
            all_traj_length = [[] for _ in range(self.num_agents)]

            lidar_data = [[[] for _ in range(self.num_agents)] for _ in range(self.n_rollout_threads)]
            pos_data = [[[] for _ in range(self.num_agents)] for _ in range(self.n_rollout_threads)]
            chunk = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            all_ogm_pred_frames = {i: [] for i in range(self.num_agents)}
            all_ogm_gt_frames = {i: [] for i in range(self.num_agents)}
            env_frames = []
            for step in range(self.episode_length):
                # step_start = time.time()

                # Sample actions
                # collect_start = time.time()
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # collect_time = time.time() - collect_start

                # env_step_start = time.time()
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                # env_step_time = time.time() - env_step_start

                # prepare_obs_start = time.time()
                vis_obs, ogm_obs, depth, lidar, camera_intrinsics, camera_extrinsics, vec_obs, human_obs, ogm_cropped = self.prepare_obs(obs)

                # prepare_obs_time = time.time() - prepare_obs_start

                if self.collect_data:
                    for t_id in range(self.n_rollout_threads):
                        for a_id in range(self.num_agents):  # num_agents
                            lidar_data[t_id][a_id].append(lidar[t_id, a_id, -1, :])
                            # pos_data[t_id][a_id].append(vec_obs[t_id, a_id, -1, [0, 2, 4]])
                            pos_data[t_id][a_id].append(vec_obs[t_id, a_id, -1, [0, 2]])

                    if dones.any():
                        thread_ids, agent_ids = np.where(dones)
                        # for t_id, a_id in zip(thread_ids, agent_ids):
                        #     traj_length = len(lidar_data[t_id][a_id])
                        #     if traj_length >= 50:  # Only store valid trajectories
                        #         all_lidar_data[a_id].extend(lidar_data[t_id][a_id])
                        #         all_pos_data[a_id].extend(pos_data[t_id][a_id])
                        #         all_traj_length[a_id].append(traj_length)

                        #     # Reset storage for new trajectory
                        #     lidar_data[t_id][a_id].clear()
                        #     pos_data[t_id][a_id].clear()

                        for t_id in thread_ids:
                            for a_id in range(self.num_agents):
                                traj_length = len(lidar_data[t_id][a_id])
                                if traj_length >= 50:  # Only store valid trajectories
                                    all_lidar_data[a_id].extend(lidar_data[t_id][a_id])
                                    all_pos_data[a_id].extend(pos_data[t_id][a_id])
                                    all_traj_length[a_id].append(traj_length)

                                # Reset storage for new trajectory
                                lidar_data[t_id][a_id] = []
                                pos_data[t_id][a_id] = []

                # if dones.any():
                #     thread_ids, agent_ids = np.where(dones)
                #     for t_id in thread_ids:
                #         for a_id in range(self.num_agents):
                #             self.last_episode_step_train[t_id, a_id] = step
                #             self.dtaci_estimator_train[t_id][a_id] = [DtACIGrid(self.aci_map_size, self.aci_map_size, alpha=0.1, gammas=np.array([0.05, 0.1, 0.2]), sigma=1/1000, eta=2.72, initial_pred=0.1) for _ in range(self.vis_seq)]
                # prediction_start = time.time()
                if self.use_gt_ogm:
                    obs = np.concatenate((ogm_obs.reshape(self.n_rollout_threads, self.num_agents, self.vis_seq, -1), vec_obs), axis=-1).reshape(self.n_rollout_threads, self.num_agents, -1)
                elif self.use_traj:
                    obs = np.concatenate((human_obs.reshape(self.n_rollout_threads, self.num_agents, -1), vec_obs.reshape(self.n_rollout_threads, self.num_agents, -1)), axis=-1)
                else:
                    # ogm_pred = self.cooperative_prediction(lidar, vec_obs, step=step)
                    ogm_pred, costs = self.cooperative_prediction(ogm_obs, lidar, vec_obs, step=step)
                    obs = np.concatenate((ogm_pred.reshape(self.n_rollout_threads, self.num_agents, -1), vec_obs.reshape(self.n_rollout_threads, self.num_agents, -1)), axis=-1)

                # visualize the episode
                if self.visualization:
                    env_frames.append(create_frame_in_memory(vec_obs[0,0,-1,:], human_obs[0,0,-1,:], robot_num=self.num_agents, human_num=self.human_num))
                    for agent_id in range(self.num_agents):
                        ego_pred_ogm = ogm_pred[0, agent_id] # [self.vis_seq+1, 64, 64]
                        ego_pred_ogm = np.flip(ego_pred_ogm, axis=1)  # Flip height to match (0,0) at bottom-left
                        ego_pred_ogm = (ego_pred_ogm * 255).astype(np.uint8) 
                        if self.use_prediction:
                            all_ogm_pred_frames[agent_id].append(ego_pred_ogm[0])
                        else:
                            all_ogm_pred_frames[agent_id].append(ego_pred_ogm[-1])
                        # all_ogm_pred_frames[agent_id].append(np.concatenate(ego_pred_ogm, axis=1))
                        all_ogm_gt_frames[agent_id].append(np.concatenate(ogm_obs[0, agent_id] * 255, axis=1))

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                # prediction_time = time.time() - prediction_start

                # rewards -= costs

                self.insert(data)

                # print(f"Step {step}:")
                # print(f"  Collect Time: {collect_time:.2f}s")
                # print(f"  Env Step Time: {env_step_time:.2f}s")
                # print(f"  Prepare Obs Time: {prepare_obs_time:.2f}s")
                # print(f"  Prediction Time: {prediction_time:.2f}s")

            if self.visualization:
                env_gif_name = f"train_episode_{episode}.mp4"
                # imageio.mimsave(os.path.join(self.gif_path, env_gif_name), env_frames, fps=10)
                with imageio.get_writer(os.path.join(self.gif_path, env_gif_name), fps=10, codec='libx264', quality=8) as writer:
                    for f in env_frames:
                        writer.append_data(f)
                for agent_id in range(self.num_agents):
                    pred_gif_name = f"train_episode_{episode}_{agent_id}_pred.mp4"
                    # imageio.mimsave(os.path.join(self.gif_path, pred_gif_name), all_ogm_pred_frames, fps=10)
                    with imageio.get_writer(os.path.join(self.gif_path, pred_gif_name), fps=10, codec='libx264', quality=8) as writer:
                        for f in all_ogm_pred_frames[agent_id]:
                            writer.append_data(f)
                    gt_gif_name = f"train_episode_{episode}_{agent_id}_gt.mp4"
                    # imageio.mimsave(os.path.join(self.gif_path, gt_gif_name), all_ogm_gt_frames, fps=10)  
                    with imageio.get_writer(os.path.join(self.gif_path, gt_gif_name), fps=10, codec='libx264', quality=8) as writer:
                        for f in all_ogm_gt_frames[agent_id]:
                            writer.append_data(f)
        
            # train_start = time.time()
            # compute return and update network
            self.compute()
            train_infos = self.train()
            # train_time = time.time() - train_start

            # print(f"Episode {episode}:")
            # print(f"  Compute Time: {compute_time:.2f}s")
            # print(f"  Train Time: {train_time:.2f}s")
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int((total_num_steps) / (end - total_start))))
                log_message = ("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                           .format(self.all_args.scenario_name,
                                   self.algorithm_name,
                                   self.experiment_name,
                                   episode,
                                   episodes,
                                   total_num_steps,
                                   self.num_env_steps,
                                   int((total_num_steps) / (end - total_start))))
                self.log_file.write(log_message)
                self.log_file.flush()
                env_infos = {}
                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                avg_reward_message = "average episode rewards is {}\n".format(
                np.mean(self.buffer.rewards) * self.episode_length)
                self.log_file.write(avg_reward_message)
                self.log_file.flush()
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            # if not self.collect_data:
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
                
            if self.collect_data and len(all_traj_length[0]) >= 200:
                print("collect trajectory chunk {} data".format(chunk))
                for agent_id in range(self.num_agents):
                    np.save(os.path.join(self.save_data_dir, f"all_lidar_data_agent_{agent_id}_val_chunk_{chunk}.npy"), all_lidar_data[agent_id])
                    np.save(os.path.join(self.save_data_dir, f"all_pos_data_agent_{agent_id}_val_chunk_{chunk}.npy"), all_pos_data[agent_id])
                    np.save(os.path.join(self.save_data_dir, f"all_traj_length_agent_{agent_id}_val_chunk_{chunk}.npy"), all_traj_length[agent_id])
                    all_lidar_data[agent_id] = []
                    all_pos_data[agent_id] = []
                    all_traj_length[agent_id] = []
                chunk += 1  

        self.log_file.close()
        self.eval_log_file.close()

        # if self.collect_data:
        #     for agent_id in range(self.num_agents):
        #         np.save(os.path.join(self.save_data_dir, f"all_lidar_data_{agent_id}.npy"), all_lidar_data[agent_id])
        #         np.save(os.path.join(self.save_data_dir, f"all_pos_data_{agent_id}.npy"), all_pos_data[agent_id])
        #         np.save(os.path.join(self.save_data_dir, f"all_traj_length_{agent_id}.npy"), all_traj_length[agent_id])

    def warmup(self):
        # reset env
        obs = self.envs.reset() # n_rollout_threads, num_agents, vis_seq, obs_dim
        
        vis_obs, ogm_obs, depth, lidar, camera_intrinsics, camera_extrinsics, vec_obs, human_obs, ogm_cropped = self.prepare_obs(obs)

        if self.use_gt_ogm:
            obs = np.concatenate((ogm_obs.reshape(self.n_rollout_threads, self.num_agents, self.vis_seq, -1), vec_obs), axis=-1).reshape(self.n_rollout_threads, self.num_agents, -1)
        elif self.use_traj:
            obs = np.concatenate((human_obs.reshape(self.n_rollout_threads, self.num_agents, -1), vec_obs.reshape(self.n_rollout_threads, self.num_agents, -1)), axis=-1)
        else:
            ogm_pred, costs = self.cooperative_prediction(ogm_obs, lidar, vec_obs, step=-1)
            obs = np.concatenate((ogm_pred.reshape(self.n_rollout_threads, self.num_agents, -1), vec_obs.reshape(self.n_rollout_threads, self.num_agents, -1)), axis=-1)
        
        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    def prepare_obs(self, obs):
        
        n_rollout_threads, num_agents, vis_seq, _ = obs.shape
        h_ogm, w_ogm = self.gt_ogm_size
        c_ogm = 1
        n_points = self.n_points

        ogm_obs = obs[:,:,:,:c_ogm*h_ogm*w_ogm].reshape(n_rollout_threads, num_agents, vis_seq, h_ogm, w_ogm) # n_rollout_threads, num_agents, vis_seq, 50, 50
        ogm_obs = np.rot90(ogm_obs, k=1, axes=(-2, -1))  # Shape: (b, num_agents, vis_seq, 50, 50)
        ogm_obs = (ogm_obs != 0).astype(np.uint8)
        # ogm_obs = np.rot90(ogm_obs, k=1, axes=(-2, -1))  # Shape: (b, num_agents, vis_seq, 50, 50)
        lidar = obs[:,:,:,c_ogm*h_ogm*w_ogm:c_ogm*h_ogm*w_ogm+n_points].reshape(n_rollout_threads, num_agents, vis_seq, -1) # n_rollout_threads, num_agents, vis_seq, 1, 360
        vec_obs = obs[:,:,:,c_ogm*h_ogm*w_ogm+n_points:c_ogm*h_ogm*w_ogm+18+n_points].reshape(n_rollout_threads, num_agents, vis_seq, -1) # n_rollout_threads, num_agents, vis_seq, 27 
        human_obs = obs[:,:,:,c_ogm*h_ogm*w_ogm+18+n_points:].reshape(n_rollout_threads, num_agents, vis_seq, -1) # n_rollout_threads, num_agents, vis_seq, 9*npc

        # lidar = obs[:,:,:,c_ogm*h_ogm*w_ogm:c_ogm*h_ogm*w_ogm+n_points].reshape(n_rollout_threads, num_agents, vis_seq, -1) # n_rollout_threads, num_agents, vis_seq, 1, 360
        # lidar = obs[:,:,:,:self.n_points].reshape(n_rollout_threads, num_agents, vis_seq, -1) # n_rollout_threads, num_agents, vis_seq, 1, 360
        # vec_obs = obs[:,:,:,self.n_points:self.n_points+18].reshape(n_rollout_threads, num_agents, vis_seq, -1) # n_rollout_threads, num_agents, vis_seq, 27 
        # human_obs = obs[:,:,:,self.n_points+18:].reshape(n_rollout_threads, num_agents, vis_seq, -1) # n_rollout_threads, num_agents, vis_seq, 9*npc
        # vec_obs = obs[:,:,:,n_points:n_points+9].reshape(n_rollout_threads, num_agents, vis_seq, -1) # n_rollout_threads, num_agents, vis_seq, 27 
        # human_obs = obs[:,:,:,n_points+9:].reshape(n_rollout_threads, num_agents, vis_seq, -1) # n_rollout_threads, num_agents, vis_seq, 9*npc
        # vec_obs = obs[:,:,:,c_ogm*h_ogm*w_ogm+n_points:c_ogm*h_ogm*w_ogm+n_points+self.num_agents*6].reshape(n_rollout_threads, num_agents, vis_seq, -1) # n_rollout_threads, num_agents, vis_seq, 12
        # human_obs = obs[:,:,:,c_ogm*h_ogm*w_ogm+n_points+self.num_agents*6:].reshape(n_rollout_threads, num_agents, vis_seq, -1) # n_rollout_threads, num_agents, vis_seq, 6*npc

        # print("position")
        # print(vec_obs[0, 0, :, :3])
        # print("target")
        # print(vec_obs[0, 0, :, 6:9])
        
        return None, ogm_obs, None, lidar, None, None, vec_obs, human_obs, None
    
    def cooperative_prediction(self, ogm_gt, lidar, vec_obs, is_eval=False, step=0):
        ogm_obs = []

        batch_size = self.n_eval_rollout_threads if is_eval else self.n_rollout_threads

        for agent_id in range(self.num_agents):
            ego_x_odom = vec_obs[:, agent_id, :, 2] # n_rollout_threads, t
            ego_y_odom = vec_obs[:, agent_id, :, 0] # n_rollout_threads, t
            ego_lidar = lidar[:, agent_id, :] # n_rollout_threads, t, 1080

            ego_motion_translation_2d = np.stack([ego_x_odom[:, [self.vis_seq-1]] - ego_x_odom, ego_y_odom[:, [self.vis_seq-1]] - ego_y_odom], axis=-1)  
            ego_transformed_lidar = transform_point_clouds(ego_lidar, ego_motion_translation_2d)  

            fused_lidar = np.zeros((batch_size, self.vis_seq, self.n_points, self.num_agents))
            fused_lidar[:, :, :, agent_id] = ego_transformed_lidar
            
            # lidar visualization
            # angles = np.linspace(0, 2 * np.pi, self.n_points)  # 360-degree angles
            # for i in range(batch_size):
            #     # Plotting the current frame in polar coordinates
            #     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
            #     ax.scatter(
            #         angles,               # Theta (angular position)
            #         ego_lidar[i, -1, :],            # R (distance from the origin)
            #         c=ego_lidar[i, -1, :],          # Color based on distance
            #         s=5, cmap='viridis'   # Point size and color map
            #     )
            #     # Customize the plot
            #     ax.set_rmax(7.0)  # Adjust max radius (depends on your data scale)
            #     ax.set_theta_zero_location("N")  # Set 0 degrees to the top (North)
            #     ax.set_theta_direction(-1)       # Set direction of angles to be clockwise
            #     # Show the plot
            #     plt.show()

            #     plt.imshow(ogm_gt[i, agent_id, -1])
            #     plt.show()

            if self.coop_mode == 'early':
                for surround_agent_id in range(self.num_agents):
                    if agent_id != surround_agent_id:
                        rob_x_odom = vec_obs[:, surround_agent_id, :, 2] 
                        rob_y_odom = vec_obs[:, surround_agent_id, :, 0]
                        rob_lidar = lidar[:, surround_agent_id, :] # n_rollout_threads, t, 1080

                        rob_motion_translation_2d = np.stack([rob_x_odom[:, [self.vis_seq-1]] - rob_x_odom, rob_y_odom[:, [self.vis_seq-1]] - rob_y_odom], axis=-1)  
                        rob_transformed_lidar = transform_point_clouds(rob_lidar, rob_motion_translation_2d)

                        translation_2d = np.stack([ego_x_odom - rob_x_odom, ego_y_odom - rob_y_odom], axis=-1)
                        fused_lidar[:, :, :, surround_agent_id] = transform_point_clouds(rob_transformed_lidar, translation_2d, is_current=True) # n_rollout_threads, t, 1080
 
            ego_ogm = transform_lidar_to_ogm(fused_lidar, map_size=self.lidar_map_size, resolution=self.resolution) # n_rollout_threads, t, 100, 100

            # early fusion visualization
            # for i in range(batch_size):
            #     plt.imshow(ego_ogm[i, -1], origin='lower')
            #     plt.title('early_fusion_ogm_{}_{}'.format(agent_id, i))
            #     plt.show()
            
            if self.use_prediction:
                # model prediction
                ego_ogm_t = torch.tensor(ego_ogm, dtype=torch.float32, device=self.device)
                ego_occ_map = to_prob_occ_map(ego_ogm_t, self.device)

                ego_ogm_t = ego_ogm_t.repeat(8, 1, 1, 1, 1)
                ego_occ_map = ego_occ_map.repeat(8, 1, 1, 1)
                prediction_samples, kl_loss = self.prediction_model(ego_ogm_t, ego_occ_map)
                prediction_samples = prediction_samples.reshape(8, -1, self.vis_seq, self.lidar_map_size, self.lidar_map_size)
                prediction = prediction_samples.mean(dim=0)
                ego_ogm_pred = prediction.detach().cpu().numpy()
                
                if self.uncertainty_type == 'entropy':
                    ego_ogm_pred_entropy = shannon_entropy(prediction_samples).detach().cpu().numpy() # n_rollout_threads, t, map_size, map_size
                    # ego_ogm_pred = np.stack([ego_ogm_pred, ego_ogm_pred_entropy], axis=2) # n_rollout_threads, t, 2, map_size, map_size
                    entropy_mask = ego_ogm_pred_entropy > self.uq_threshold # n_rollout_threads, t, map_size, map_size
                    ego_ogm_pred_entropy_masked = ego_ogm_pred_entropy * entropy_mask # n_rollout_threads, t, map_size, map_size
                    ego_ogm_pred += ego_ogm_pred_entropy_masked
                    # ego_ogm_pred = expand_uncertain_areas(prediction, ego_ogm_pred_entropy)

                elif self.uncertainty_type == 'aci':
                    aci_predicted_conformity_scores = np.zeros((batch_size, self.vis_seq, self.aci_map_size, self.aci_map_size)) # n_rollout_threads, t, aci_map_size, aci_map_size

                    for b in range(batch_size):
                        for i in range(self.vis_seq):
                            if step-i < 0:
                                past_pred = self.buffer.obs[step-i+self.episode_length+1].reshape(self.n_rollout_threads, self.num_agents, self.vis_seq, -1)[:, agent_id, i, :self.lidar_map_size*self.lidar_map_size].reshape(self.n_rollout_threads, self.lidar_map_size, self.lidar_map_size)
                            else:
                                past_pred = self.buffer.obs[step-i].reshape(self.n_rollout_threads, self.num_agents, self.vis_seq, -1)[:, agent_id, i, :self.lidar_map_size*self.lidar_map_size].reshape(self.n_rollout_threads, self.lidar_map_size, self.lidar_map_size)

                            nonconformity_score = np.abs(ego_ogm[:, -1] - past_pred) # n_rollout_threads, map_size, map_size
                            nonconformity_score = nonconformity_score[:, self.center_start_idx:self.center_end_idx, self.center_start_idx:self.center_end_idx]

                            if is_eval and step > (self.last_episode_step_test[b, agent_id] + 5) % self.episode_length:
                                self.dtaci_estimator_test[b][agent_id][i].update_true_grid(nonconformity_score[b])
                                aci_prediction = np.clip(self.dtaci_estimator_test[b][agent_id][i].make_prediction(), 0, 1)
                                aci_predicted_conformity_scores[b, i] = aci_prediction        
                            elif not is_eval and step > (self.last_episode_step_train[b, agent_id] + 5) % self.episode_length:
                                self.dtaci_estimator_train[b][agent_id][i].update_true_grid(nonconformity_score[b])
                                aci_prediction = np.clip(self.dtaci_estimator_train[b][agent_id][i].make_prediction(), 0, 1)
                                aci_predicted_conformity_scores[b, i] = aci_prediction        
                    # ego_ogm_pred = np.stack([ego_ogm_pred, aci_predicted_conformity_scores], axis=2) # n_rollout_threads, t, 2, map_size, map_size    
                    aci_mask = aci_predicted_conformity_scores > self.uq_threshold
                    aci_predicted_conformity_scores_masked = aci_predicted_conformity_scores * aci_mask

                    ego_ogm_pred[:, :, self.center_start_idx:self.center_end_idx, self.center_start_idx:self.center_end_idx] += aci_predicted_conformity_scores_masked

                ego_ogm_pred = np.concatenate((ego_ogm[:, -1, None], ego_ogm_pred), axis=1) # batch_size, vis_seq+1, map_size, map_size
                ogm_obs.append(ego_ogm_pred)
            else:
                ogm_obs.append(ego_ogm)

        if self.coop_mode == 'late':
            for agent_id in range(self.num_agents):

                ego_x_odom = vec_obs[:, agent_id, -1, 2]
                ego_y_odom = vec_obs[:, agent_id, -1, 0]
                ego_ogm = ogm_obs[agent_id]

                for surround_agent_id in range(self.num_agents):
                    if agent_id != surround_agent_id:
                        rob_x_odom = vec_obs[:, surround_agent_id, -1, 2]
                        rob_y_odom = vec_obs[:, surround_agent_id, -1, 0]
                        rob_ogm = ogm_obs[surround_agent_id]

                        translation_2d = np.stack([rob_x_odom - ego_x_odom, rob_y_odom - ego_y_odom], axis=-1)
                        translation_2d /= self.resolution
                        ego_ogm = fuse_ogm(ego_ogm, rob_ogm, translation_2d, self.use_prediction)

                ogm_obs.append(ego_ogm)
            # ogm_obs = ogm_obs[3:]
            ogm_obs = ogm_obs[self.num_agents:]
        
        # late fusion visualization
        # for agent_id in range(self.num_agents):
        #     for i in range(batch_size):
        #         plt.imshow(ogm_obs[agent_id][i, -1], origin='lower')
        #         plt.title('late_fusion_ogm_{}_{}'.format(agent_id, i))
        #         plt.show()
                
                # plt.imshow(ogm_obs[agent_id][i, -1], origin='lower')
                # plt.title('late_fusion_ogm_{}_{}'.format(agent_id, i))
                # plt.show()

        ogm_obs = np.array(np.stack(ogm_obs, axis=1)) # n_rollout_threads, num_agents, horizon, 1 or 2, H, W
        if self.constrained:
            # Calculate cost (max total probability over time horizon for each agent)
            max_intrusion_costs = np.zeros((batch_size, self.num_agents))

            # For each agent, calculate intrusion costs for all batches
            for agent_id in range(self.num_agents):
                # Initialize minimum distances for each batch as infinity
                batch_min_distances = np.full(batch_size, np.inf)
                
                # For each timestep
                for t in range(self.vis_seq):
                    # Process all batches together for this timestep and agent
                    ogm_batch = ogm_obs[:, agent_id, t]  # shape: (batch_size, H, W)
                    
                    # Loop through each batch (unavoidable for individual mask processing)
                    for b in range(batch_size):
                        # Create mask for occupied cells within center region for this batch
                        occupied_cells_mask = (ogm_batch[b] * self.center_mask) > self.occupied_probability_threshold
                        
                        # If any cells are occupied for this batch
                        if np.any(occupied_cells_mask):
                            # Get distances of occupied cells from center
                            occupied_distances = self.dist_from_center[occupied_cells_mask]
                            
                            if len(occupied_distances) > 0:
                                # Find minimum distance for this batch and timestep
                                timestep_min_dist = np.min(occupied_distances)
                                # Update minimum distance for this batch
                                batch_min_distances[b] = min(batch_min_distances[b], timestep_min_dist)
                
                # Calculate intrusion costs for all batches using vector operations
                intrusion_depths = np.maximum(0, self.center_radius - batch_min_distances)
                intrusion_costs = intrusion_depths * self.time_step * self.discomfort_factor
                
                # Assign costs only where valid intrusions occurred
                valid_intrusions = batch_min_distances < np.inf
                max_intrusion_costs[:, agent_id] = np.where(valid_intrusions, intrusion_costs, 0)

            max_intrusion_costs = max_intrusion_costs.reshape(batch_size, self.num_agents, -1)
        
        # prediction visualization
        # for i in range(batch_size):
        #     for agent_id in range(self.num_agents):
        #         if max_intrusion_costs[i, agent_id] != 0.0:
        #             plt.imshow(ogm_obs[i, agent_id, 0], origin='lower')
        #             plt.show()
        #             plt.imshow(ogm_obs[i, agent_id, -1], origin='lower')
        #             plt.show()
        #             print(max_intrusion_costs[i, agent_id, 0])
            
            return ogm_obs, max_intrusion_costs
        else:
            return ogm_obs, None

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            #raise NotImplementedError
            actions_env = action.view(self.n_rollout_threads, self.num_agents, -1)
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        
        count = 0

        success_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        obstacle_collision_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        npc_collision_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        robot_collision_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        navigation_time_single = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        navigation_time_total = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        navigation_time_step_single = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        navigation_time_step_total = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        path_length_single = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        path_length_total = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        sharp_turn_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        previous_velocity = np.zeros((self.n_eval_rollout_threads, self.num_agents)) 
        velocity_changes_single = np.zeros((self.n_eval_rollout_threads, self.num_agents))  
        velocity_changes_total = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        timeout_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        end_counts = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        eval_episode_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        ITR_future_step = 10
        last_rob_pos = np.zeros((self.n_eval_rollout_threads, self.num_agents, ITR_future_step, 3))
        previous_pos = np.zeros((self.n_eval_rollout_threads, self.num_agents, 3))
        intrusion_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        intrusion_count_single = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        ITR_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        eval_obs = self.eval_envs.reset()
        vis_obs, ogm_obs, depth, lidar, camera_intrinsics, camera_extrinsics, vec_obs, human_obs, ogm_cropped = self.prepare_obs(eval_obs)

        if self.use_gt_ogm:
            eval_obs = np.concatenate((ogm_obs.reshape(self.n_eval_rollout_threads, self.num_agents, self.vis_seq, -1), vec_obs), axis=-1).reshape(self.n_eval_rollout_threads, self.num_agents, -1)
        elif self.use_traj:
            eval_obs = np.concatenate((human_obs.reshape(self.n_eval_rollout_threads, self.num_agents, -1), vec_obs.reshape(self.n_eval_rollout_threads, self.num_agents, -1)), axis=-1)
        else:
            ogm_pred, costs = self.cooperative_prediction(ogm_obs, lidar, vec_obs, is_eval=True, step=-1)
            eval_obs = np.concatenate((ogm_pred.reshape(self.n_eval_rollout_threads, self.num_agents, -1), vec_obs.reshape(self.n_eval_rollout_threads, self.num_agents, -1)), axis=-1)
        
        for eval_episode in range(self.all_args.eval_episodes):

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            # dones_ = np.zeros((self.n_eval_rollout_threads, self.num_agents))
            all_ogm_pred_frames = {i: [] for i in range(self.num_agents)}
            all_ogm_gt_frames = {i: [] for i in range(self.num_agents)} 
            env_frames = []

            for eval_step in range(self.episode_length):
                step_start = time.time()
                count += 1
                self.trainer.prep_rollout()
                eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
                
                if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[0].shape):
                        eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                        if i == 0:
                            eval_actions_env = eval_uc_actions_env
                        else:
                            eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
                elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                    eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
                else:
                    #raise NotImplementedError
                    eval_actions_env = eval_action.view(self.n_eval_rollout_threads, self.num_agents, -1)

                # Obser reward and next obs
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
                vis_obs, ogm_obs, depth, lidar, camera_intrinsics, camera_extrinsics, vec_obs, human_obs, ogm_cropped = self.prepare_obs(eval_obs)

                # if eval_dones.any():
                #     thread_ids, agent_ids = np.where(eval_dones)
                #     for t_id in thread_ids:
                #         for a_id in range(self.num_agents):
                #             self.last_episode_step_test[t_id, a_id] = eval_step
                #             self.dtaci_estimator_test[t_id][a_id] = [DtACIGrid(self.aci_map_size, self.aci_map_size, alpha=0.1, gammas=np.array([0.05, 0.1, 0.2]), sigma=1/1000, eta=2.72, initial_pred=0.1) for _ in range(self.vis_seq)]

                if self.use_gt_ogm:
                    eval_obs = np.concatenate((ogm_obs.reshape(self.n_eval_rollout_threads, self.num_agents, self.vis_seq, -1), vec_obs), axis=-1).reshape(self.n_eval_rollout_threads, self.num_agents, -1)
                elif self.use_traj:
                    eval_obs = np.concatenate((human_obs.reshape(self.n_eval_rollout_threads, self.num_agents, -1), vec_obs.reshape(self.n_eval_rollout_threads, self.num_agents, -1)), axis=-1)
                else:
                    ogm_pred, costs = self.cooperative_prediction(ogm_obs, lidar, vec_obs, is_eval=True, step=eval_step)
                    eval_obs = np.concatenate((ogm_pred.reshape(self.n_eval_rollout_threads, self.num_agents, -1), vec_obs.reshape(self.n_eval_rollout_threads, self.num_agents, -1)), axis=-1)
                
                if self.visualization:
                    env_frames.append(create_frame_in_memory(vec_obs[0,0,-1,:], human_obs[0,0,-1,:], robot_num=self.num_agents, human_num=self.human_num))
                    for agent_id in range(self.num_agents):
                        ego_pred_ogm = ogm_pred[0, agent_id] # [self.vis_seq+1, 64, 64]
                        ego_pred_ogm = np.flip(ego_pred_ogm, axis=1)  # Flip height to match (0,0) at bottom-left
                        ego_pred_ogm = (ego_pred_ogm * 255).astype(np.uint8) 
                        # all_ogm_pred_frames[agent_id].append(ego_pred_ogm[-1])
                        if self.use_prediction:
                            all_ogm_pred_frames[agent_id].append(ego_pred_ogm[0])
                        else:
                            all_ogm_pred_frames[agent_id].append(ego_pred_ogm[-1])
                        # all_ogm_pred_frames[agent_id].append(np.concatenate(ego_pred_ogm, axis=1))
                        all_ogm_gt_frames[agent_id].append(np.concatenate(ogm_obs[0, agent_id] * 255, axis=1))

                # eval_rewards -= costs
                
                step_end = time.time()
                # eval_episode_rewards.append(eval_rewards)
                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
                
                for rollout in range(self.n_eval_rollout_threads):
                    for agent_id in range(self.num_agents): # n_eval_rollout
                                                
                        if eval_actions_env[rollout,agent_id, 1] >= 0.8 or eval_actions_env[rollout,agent_id, 1] <= -0.8:
                                sharp_turn_count[rollout, agent_id] += 1

                        eval_episode_rewards[rollout, agent_id] += eval_rewards[rollout, agent_id]
                        # current_pos = eval_obs[rollout, agent_id, -27:-24]
                        # current_pos = eval_obs[rollout, agent_id, -9:-6]
                        # current_pos = eval_obs[rollout, agent_id, -72:-69]
                        current_pos = eval_obs[rollout, agent_id, -self.num_agents*6:-self.num_agents*6+3]
                        
                        if not eval_dones[rollout, agent_id] and previous_pos[rollout, agent_id].any():
                            navigation_time_single[rollout][agent_id] += 0.2
                            navigation_time_step_single[rollout][agent_id] += 1
                            path_length_single[rollout][agent_id] += np.linalg.norm(current_pos - previous_pos[rollout, agent_id])

                            velocity = np.linalg.norm(current_pos - previous_pos[rollout, agent_id]) / 0.2  # Velocity = distance / time step (0.2s)
                            
                            # Calculate velocity change (Δv) if there is a previous velocity
                            if previous_velocity[rollout, agent_id] > 0:
                                delta_v = abs(velocity - previous_velocity[rollout, agent_id])
                                velocity_changes_single[rollout, agent_id] += delta_v  # Add to the sum of velocity changes
                            
                            # Update the velocity for the current step
                            previous_velocity[rollout, agent_id] = velocity

                        previous_pos[rollout, agent_id] = current_pos
                        last_rob_pos[rollout, agent_id, int(ITR_count[rollout, agent_id])] = vec_obs[rollout, agent_id, -1, [0, 1, 2]]
                        
                        if ITR_count[rollout, agent_id] < ITR_future_step-1:
                            
                            human_obs = human_obs.reshape(self.n_eval_rollout_threads, self.num_agents, self.vis_seq, self.human_num, 9)
                            human_pos = human_obs[rollout, agent_id, -1, :, :3]
                            
                            for i in range(int(ITR_count[rollout, agent_id])):
                                
                                distances = np.linalg.norm(human_pos - last_rob_pos[rollout, agent_id, i], axis=-1)
                                
                                if np.any(distances < 0.5) and not np.any(human_pos == 0):
                                    intrusion_count_single[rollout, agent_id] += 1

                            ITR_count[rollout, agent_id] += 1
                        else:
                            ITR_count[rollout, agent_id] = 0
                                
                        if eval_dones[rollout, agent_id]:
                            end_counts[rollout, agent_id] += 1
                            
                            #print(intrusion_count_single[rollout, agent_id],navigation_time_step_single[rollout][agent_id],intrusion_count_single[rollout, agent_id]/navigation_time_step_single[rollout][agent_id])
                            if navigation_time_step_single[rollout][agent_id] > 0:
                                intrusion_count[rollout, agent_id] += intrusion_count_single[rollout, agent_id] / navigation_time_step_single[rollout][agent_id]
                            intrusion_count_single[rollout, agent_id] = 0
                            ITR_count[rollout, agent_id] = 0
                            last_rob_pos[rollout, agent_id] = np.zeros((ITR_future_step, 3))
                            velocity_changes_total[rollout, agent_id] += velocity_changes_single[rollout, agent_id]
                            
                            if eval_rewards[rollout, agent_id] == 10:
                                success_count[rollout, agent_id] += 1
                                navigation_time_total[rollout, agent_id] += navigation_time_single[rollout][agent_id]  
                                navigation_time_step_total[rollout, agent_id] += navigation_time_step_single[rollout][agent_id]   
                                path_length_total[rollout, agent_id] += path_length_single[rollout][agent_id]

                            elif eval_rewards[rollout, agent_id] == -10.0:
                                obstacle_collision_count[rollout, agent_id] += 1
                                #dones_[rollout,agent_id]=1
                            elif eval_rewards[rollout, agent_id] == -20:
                                npc_collision_count[rollout, agent_id] += 1
                            elif eval_rewards[rollout, agent_id] == -15:
                                robot_collision_count[rollout, agent_id] += 1
                            elif eval_rewards[rollout, agent_id] == 0.0:
                                timeout_count[rollout, agent_id] += 1
                                #dones_[rollout,agent_id]=1

                            navigation_time_step_single[rollout][agent_id] = 0
                            navigation_time_single[rollout][agent_id] = 0
                            path_length_single[rollout][agent_id] = 0
                            velocity_changes_single[rollout, agent_id] = 0
                            previous_pos[rollout, agent_id] = np.zeros(3)
                            previous_velocity[rollout, agent_id] = 0

            if self.visualization:
                env_gif_name = f"eval_episode_{eval_episode}.mp4"
                # imageio.mimsave(os.path.join(self.gif_path, env_gif_name), env_frames, fps=10)
                with imageio.get_writer(os.path.join(self.gif_path, env_gif_name), fps=10, codec='libx264', quality=8) as writer:
                    for f in env_frames:
                        writer.append_data(f)
                for agent_id in range(self.num_agents):
                    pred_gif_name = f"eval_episode_{eval_episode}_{agent_id}_pred.mp4"
                    # imageio.mimsave(os.path.join(self.gif_path, pred_gif_name), all_ogm_pred_frames, fps=10)
                    with imageio.get_writer(os.path.join(self.gif_path, pred_gif_name), fps=10, codec='libx264', quality=8) as writer:
                        for f in all_ogm_pred_frames[agent_id]:
                            writer.append_data(f)
                    gt_gif_name = f"eval_episode_{eval_episode}_{agent_id}_gt.mp4"
                    # imageio.mimsave(os.path.join(self.gif_path, gt_gif_name), all_ogm_gt_frames, fps=10)  
                    with imageio.get_writer(os.path.join(self.gif_path, gt_gif_name), fps=10, codec='libx264', quality=8) as writer:
                        for f in all_ogm_gt_frames[agent_id]:
                            writer.append_data(f)
                print(f"[Render] Episode {eval_episode} saved GIF to: {self.gif_path}")
                    
            # gif_name = f"eval_episode_{eval_episode}.gif"
            # imageio.mimsave(os.path.join(self.gif_path, gif_name), render_frames, fps=10)
            # print(f"[Render] Episode {eval_episode} saved GIF to: {self.gif_path}")
            # render_frames.clear()

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_episode_rewards = np.mean(eval_episode_rewards, axis=0)
        log_message = f"total_num_steps: {total_num_steps}\n"
        log_message += f"eval average episode rewards: {eval_episode_rewards}\n"
        log_message += f'success_count: {success_count}, obstacle_collision_count: {obstacle_collision_count}, npc_collision_count: {npc_collision_count}, robot_collision_count: {robot_collision_count}, timeout_count: {timeout_count}\n'
        log_message += f"end counts: {end_counts}\n"
        end_counts[end_counts == 0] += 1
        success_rate = np.mean(success_count/end_counts, axis=0)
        obstacle_collision_rate = np.mean(obstacle_collision_count/end_counts, axis=0)
        npc_collision_rate = np.mean(npc_collision_count/end_counts, axis=0)
        robot_collision_rate = np.mean(robot_collision_count/end_counts, axis=0)
        timeout_rate = np.mean(timeout_count/end_counts, axis=0)
    
        navigation_time = np.mean(navigation_time_total/success_count, axis=0)
        navigation_time_step = np.mean(navigation_time_step_total/success_count, axis=0)
        path_length = np.mean(path_length_total/success_count, axis=0)
        velocity_change = np.mean(velocity_changes_total/end_counts, axis=0)
        # log_message += f'{navigation_time},{navigation_time_step},{path_length}\n'
        sharp_turn_rate = np.mean(sharp_turn_count/count, axis=0)
        ITR = np.mean(intrusion_count/end_counts, axis=0)     

        for agent_id in range(self.num_agents):
            log_message += (
                f"Agent {agent_id} - SR: {np.round(success_rate[agent_id], 2)}, "
                f"obstacle_CR: {np.round(obstacle_collision_rate[agent_id], 2)}, "
                f"npc_CR: {np.round(npc_collision_rate[agent_id], 2)}, "
                f"robot_CR: {np.round(robot_collision_rate[agent_id], 2)}, "
                f"TR: {np.round(timeout_rate[agent_id], 2)}, "
                f"Reward: {np.round(eval_episode_rewards[agent_id], 2)}\n "
            )
        for agent_id in range(self.num_agents):
            log_message += (
                f"Agent {agent_id} - ANT: {np.round(navigation_time[agent_id],2)}, "
                f"ANT_step: {np.round(navigation_time_step[agent_id],2)}, "
                f"PL: {np.round(path_length[agent_id],2)}, "
                f"STR: {np.round(sharp_turn_rate[agent_id],2)}, "
                f"AA: {np.round(velocity_change[agent_id],2)}, "
                f"ITR: {np.round(ITR[agent_id],2)}\n "
            )
        self.eval_log_file.write(log_message)
        self.eval_log_file.flush()
        #self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
