import numpy as np


# class EnvCore(object):
#     """
#     # 环境中的智能体
#     """

#     def __init__(self):
#         self.agent_num = 2  # 设置智能体(小飞机)的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two
#         self.obs_dim = 14  # 设置智能体的观测维度 # set the observation dimension of agents
#         self.action_dim = 5  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional

#     def reset(self):
#         """
#         # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
#         # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
#         """
#         sub_agent_obs = []
#         for i in range(self.agent_num):
#             sub_obs = np.random.random(size=(14,))
#             sub_agent_obs.append(sub_obs)
#         return sub_agent_obs

#     def step(self, actions):
#         """
#         # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
#         # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
#         # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
#         # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
#         """
#         sub_agent_obs = []
#         sub_agent_reward = []
#         sub_agent_done = []
#         sub_agent_info = []
#         for i in range(self.agent_num):
#             sub_agent_obs.append(np.random.random(size=(14,)))
#             sub_agent_reward.append([np.random.rand()])
#             sub_agent_done.append(False)
#             sub_agent_info.append({})

#         return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

from onpolicy.envs.unity_env_wrapper import UnityEnvWrapper

class EnvCore(object):
    def __init__(self, config):
        self.env = UnityEnvWrapper(
            unity_env_path=config.unity_env_path,
            base_port=config.base_port,
            no_graphics=config.no_graphics,
            continuous_action=config.continuous_action,
            rank=config.rank,
            vis_seq=config.vis_seq,
            num_agents=config.num_agents
        )
        # self.agent_num = 3
        self.agent_num = config.num_agents
        self.action_dim = 2
        if config.use_traj:
            self.obs_dim = (config.human_vec_size + config.ego_vec_size + config.other_vec_size) * config.vis_seq 
        # elif config.uncertainty_type == "entropy" or config.uncertainty_type == "aci":
        #     self.obs_dim = (config.lidar_map_size*config.lidar_map_size*2 + config.ego_vec_size + config.other_vec_size) * config.vis_seq
        elif config.use_prediction:
            self.obs_dim = (config.lidar_map_size*config.lidar_map_size + config.ego_vec_size + config.other_vec_size) * config.vis_seq + config.lidar_map_size*config.lidar_map_size
        else:
            self.obs_dim = (config.lidar_map_size*config.lidar_map_size + config.ego_vec_size + config.other_vec_size) * config.vis_seq

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        actions=np.array(actions.cpu())
        return self.env.step(actions)

    def close(self):
        self.env.close()
