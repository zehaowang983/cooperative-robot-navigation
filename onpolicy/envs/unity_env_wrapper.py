from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
from collections import deque
from onpolicy.envs.lazy_frame import LazyFrames

class UnityEnvWrapper:
    def __init__(self, unity_env_path, base_port, no_graphics, continuous_action=True, rank=0, vis_seq=3, num_agents=3):
        self.unity_env = UnityEnvironment(file_name=unity_env_path, base_port=base_port+rank, no_graphics=no_graphics)
        self.unity_env.reset()
        self.group_name = list(self.unity_env.behavior_specs.keys())[0]
        self.group_spec = self.unity_env.behavior_specs[self.group_name]
        self.continuous_action = continuous_action
        self.vis_seq = vis_seq
        self.num_agents = num_agents

        # Define action and observation spaces
        self.action_space = self.group_spec.action_spec
        
        self.observation_space = self.group_spec.observation_specs[0]

        self.lz4_compress = True
        self.frames = {i: deque(maxlen=self.vis_seq) for i in range(self.num_agents)}

    def reset(self):
        
        self.unity_env.reset()
        decision_steps, _ = self.unity_env.get_steps(self.group_name)
        observations = []
        
        # vis and vec obs
        # Iterate over each agent in decision_steps
        for agent_id in decision_steps.agent_id:
            # Get the current agent's observation
            # vis_obs = decision_steps[agent_id].obs[0]
            # vis_obs = vis_obs.flatten()
            # vec_obs = decision_steps[agent_id].obs[1]
            # obs = np.concatenate([vis_obs,vec_obs], axis=-1)
            obs = decision_steps[agent_id].obs[0]
            for _ in range(self.vis_seq):
                self.frames[agent_id].append(obs)
            
        #     observations.append(obs)
        for agent_id in decision_steps.agent_id:
            observations.append(LazyFrames(list(self.frames[agent_id]), self.lz4_compress))

        return observations

    def step(self, actions):
        # Convert actions to Unity-compatible format
        if self.continuous_action:
            action_tuple = ActionTuple(continuous=actions)
        else:
            action_tuple = ActionTuple(discrete=actions)

        self.unity_env.set_actions(self.group_name, action_tuple)
        self.unity_env.step()

        decision_steps, terminal_steps = self.unity_env.get_steps(self.group_name)

        # observations = []
        # rewards = []
        # dones = []
        # infos = []

        # Process decision steps for agents not in terminal steps
        observations = {}
        # human_observations = {}
        rewards = {}
        dones = {}
        infos = {}        # Process terminal steps first
        processed_agent_ids = set()
        for agent_id in terminal_steps.agent_id:
            # vis_obs = terminal_steps[agent_id].obs[0]
            # vis_obs = vis_obs.flatten()
            # vec_obs = terminal_steps[agent_id].obs[1]
            # self.frames[agent_id].append(np.concatenate([vis_obs, vec_obs], axis=-1))
            self.frames[agent_id].append(terminal_steps[agent_id].obs[0])
            observations[agent_id] = LazyFrames(list(self.frames[agent_id]), self.lz4_compress)
            reward = terminal_steps[agent_id].reward
            rewards[agent_id] = [reward]
            dones[agent_id] = True
            # if reward == 0.0:
            #     infos[agent_id] = "timeout"
            # elif reward == 20.0:
            #     infos[agent_id] = "reach goal"
            # elif reward == -20.0:
            #     infos[agent_id] = "collision with NPC"
            # elif reward == -10.0:
            #     infos[agent_id] = "collision with Obstacle"
            # elif reward == -15.0:
            #     infos[agent_id] = "collision with Robot"
            # # print("reward:", reward)
            # # print(infos[-1])
            processed_agent_ids.add(agent_id)        # Process decision steps for agents not in terminal steps

        for agent_id in decision_steps.agent_id:
            if agent_id not in processed_agent_ids:

                # vis_obs = decision_steps[agent_id].obs[0]
                # vis_obs = vis_obs.flatten()
                # vec_obs = decision_steps[agent_id].obs[1]
                # self.frames[agent_id].append(np.concatenate([vis_obs,vec_obs], axis=-1))
                self.frames[agent_id].append(decision_steps[agent_id].obs[0])
                observations[agent_id] = LazyFrames(list(self.frames[agent_id]), self.lz4_compress)
                rewards[agent_id] = [decision_steps[agent_id].reward]
                dones[agent_id] = False
                infos[agent_id] = {}        # convert to list
                
        observations = [observations[key] for key in sorted(observations.keys())]
        rewards = [rewards[key] for key in sorted(rewards.keys())]
        dones = [dones[key] for key in sorted(dones.keys())]
        infos = [infos[key] for key in sorted(infos.keys())]
        
        return [observations, rewards, dones, infos]

    def close(self):
        self.unity_env.close()