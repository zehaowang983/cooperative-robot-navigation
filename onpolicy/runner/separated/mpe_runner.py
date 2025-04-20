    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import matplotlib.pyplot as plt
from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
       
    def run(self):
        self.warmup()   
        save_folder = 'visualizations'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        success_count=np.zeros((self.n_rollout_threads,1))
        collision_count=np.zeros((self.n_rollout_threads,1))
        timeout_count=np.zeros((self.n_rollout_threads,1))
        for episode in range(episodes):

            episode_rewards = np.zeros((self.n_rollout_threads, self.num_agents))
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)
            #unique_values = set()
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                #print(obs.shape, rewards.shape, dones.shape) # 32 3 2509 2527, 32 3 1, 32 3
                
                # ！ 0: free, 1: static, 2: dynamic, 3:unkown
                dones_ = np.zeros(self.n_rollout_threads)
                for rollout in range(self.n_rollout_threads):  # Iterate over rollouts
                    for agent_id in range(self.num_agents):  # Iterate over agent
                        #print(dones[rollout, agent_id],rewards[rollout, agent_id])
                        # if rewards[rollout, agent_id] == 1.0 or \
                        #     rewards[rollout, agent_id] == -0.01 or \
                        #     rewards[rollout, agent_id] == -0.1 or \
                        #     rewards[rollout, agent_id] == -0.5 or \
                        #     rewards[rollout, agent_id] == -0.25:
                        #         if not dones[rollout, agent_id]:
                        #             print('done incorrect')
                        #             exit()

                        # #v13
                        # if not dones[rollout, agent_id]:  # Only accumulate reward if not done
                        #     episode_rewards[rollout,agent_id] += rewards[rollout, agent_id]
                        # if rewards[rollout,agent_id]==20.0:
                        #     success_count[rollout]+=1
                        # if rewards[rollout,agent_id]==-10.:
                        #     collision_count[rollout]+=1
                        # if rewards[rollout,agent_id]==0:
                        #     timeout_count[rollout]+=1

                        #v8
                        if not dones[rollout, agent_id]:  # Only accumulate reward if not done
                            episode_rewards[rollout,agent_id] += rewards[rollout, agent_id]
                        if rewards[rollout,agent_id]==20.0 and not dones_[rollout]:
                            success_count[rollout]+=1
                            dones[rollout]=1
                        if rewards[rollout,agent_id]==-20.0 and not dones_[rollout]:
                            collision_count[rollout]+=1
                            dones[rollout]=1
                        if rewards[rollout,agent_id]==-0.01 and not dones_[rollout]:
                            timeout_count[rollout]+=1
                            dones[rollout]=1
                        # # Extract the first 10000 values from the last dimension of the obs tensor
                        # obs_data = obs[rollout, agent_id]  # Shape (10027,)
                        # obs_slice = obs_data[:10000]       # Get first 10000 values (10000,)
                        # unique_values.update(obs_slice)
                        # # print(unique_values)
                        
                        # # Reshape the data to 100x100 grid
                        # obs_reshaped = obs_slice.reshape(100, 100)

                        # # Normalize the data to [0, 1] for grayscale visualization (if necessary)
                        # obs_normalized = (obs_reshaped - np.min(obs_reshaped)) / (np.max(obs_reshaped) - np.min(obs_reshaped))

                        # # Save the image as grayscale in the specified folder
                        # image_filename = os.path.join(save_folder, f"visualization_step{step}_rollout{rollout}_agent{agent_id}.png")
                        # plt.imshow(obs_normalized, cmap='gray')
                        # plt.axis('off')  # Hide the axes
                        # plt.savefig(image_filename, bbox_inches='tight', pad_inches=0)
                        # plt.close()
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                
                end = time.time()
                episode_rewards=np.mean(episode_rewards,axis=0)
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, Avg episode reward 0 {}, Avg SR {} CR {} TR {},total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                episode_rewards[0],
                                np.mean(success_count)/(episode+1),
                                np.mean(collision_count)/(episode+1),
                                np.mean(timeout_count)/(episode+1),
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))
                

                # if self.env_name == "MPE":
                #     for agent_id in range(self.num_agents):
                #         idv_rews = []
                #         for info in infos:
                #             for count, info in enumerate(infos):
                #                 if 'individual_reward' in infos[count][agent_id].keys():
                #                     idv_rews.append(infos[count][agent_id].get('individual_reward', 0))
                #         train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                #         train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        # idv_rews = []
                        # for info in infos:
                        #     for count, info in enumerate(infos):
                        #         if 'individual_reward' in infos[count][agent_id].keys():
                        #             idv_rews.append(infos[count][agent_id].get('individual_reward', 0))
                        train_infos[agent_id].update({'individual_rewards': episode_rewards[agent_id]})
                        train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        
        obs = self.envs.reset()
        

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                #raise NotImplementedError
                action_env=action

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append( _t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        #print(rnn_states.shape, rnn_states_critic.shape) #(16, 3, 5, 64) (16, 3, 5, 64)
        
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                        np.array(list(obs[:, agent_id])),
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        success_count=[0 for _ in range(self.n_eval_rollout_threads)]
        collision_count=[0 for _ in range(self.n_eval_rollout_threads)]
        timeout_count=[0 for _ in range(self.n_eval_rollout_threads)]
        eval_all_rewards = [[] for agent_id in range(self.num_agents)]
        eval_episode_rewards = np.zeros((self.n_rollout_threads, self.num_agents))
        for eval_episode in range(self.all_args.eval_episodes):
            
            eval_obs = self.eval_envs.reset()

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            dones_ = np.zeros(self.n_eval_rollout_threads)
            for eval_step in range(self.episode_length):
                eval_temp_actions_env = []
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                    eval_rnn_states[:, agent_id],
                                                                                    eval_masks[:, agent_id],
                                                                                    deterministic=True)

                    eval_action = eval_action.detach().cpu().numpy()
                    # rearrange action
                    if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.eval_envs.action_space[agent_id].shape):
                            eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                            if i == 0:
                                eval_action_env = eval_uc_action_env
                            else:
                                eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                    elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                    else:
                        #raise NotImplementedError
                        eval_action_env=eval_action

                    eval_temp_actions_env.append(eval_action_env)
                    eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                    
                # [envs, agents, dim]
                eval_actions_env = []
                for i in range(self.n_eval_rollout_threads):
                    eval_one_hot_action_env = []
                    for eval_temp_action_env in eval_temp_actions_env:
                        eval_one_hot_action_env.append(eval_temp_action_env[i])
                    eval_actions_env.append(eval_one_hot_action_env)

                # Obser reward and next obs
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
                

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
                
                for rollout in range(self.n_eval_rollout_threads):
                    for agent_id in range(self.num_agents): # n_eval_rollout
                        if not eval_dones[rollout,agent_id]:
                            eval_episode_rewards[rollout,agent_id]+=eval_rewards[rollout,agent_id]

                        if eval_rewards[rollout,agent_id]==20.0 and not dones_[rollout]:
                            
                            success_count[rollout]+=1
                            dones_[rollout]=1
                        if eval_rewards[rollout,agent_id]==-20.0 and not dones_[rollout] :
                            
                            collision_count[rollout]+=1
                            dones_[rollout]=1
                        if eval_rewards[rollout,agent_id]==-0.01 and not dones_[rollout,agent_id]:
                            
                            timeout_count[rollout]+=1
                            dones_[rollout]=1
            
        eval_train_infos = []
        eval_episode_rewards=np.mean(eval_episode_rewards,axis=0)
        success_count=np.mean(success_count)
        collision_count=np.mean(collision_count)
        timeout_count=np.mean(timeout_count)

        for agent_id in range(self.num_agents):
            # eval_average_step_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            # eval_all_rewards[agent_id].append(eval_average_step_rewards)
            eval_train_infos.append({"eval_average_episode_rewards": eval_episode_rewards[agent_id]})
            eval_train_infos.append({"SR": success_count/self.all_args.eval_episodes})
            eval_train_infos.append({"CR": collision_count/self.all_args.eval_episodes})
            eval_train_infos.append({"TR": timeout_count/self.all_args.eval_episodes})
        print('Average SR {}, CR {}, TR {}'.format(success_count/self.all_args.eval_episodes,collision_count/self.all_args.eval_episodes,timeout_count/self.all_args.eval_episodes))
        

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
