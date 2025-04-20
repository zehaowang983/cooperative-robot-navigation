import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.utils.convlstm import ConvLSTM

class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_agents = args.num_agents
        self.share_policy = args.share_policy
        self.num_mini_batch=args.num_mini_batch
        self.episode_length=args.episode_length
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.ego_vec_size = args.ego_vec_size
        self.other_vec_size = args.other_vec_size
        # self.vis_size = (1, args.lidar_map_size, args.lidar_map_size) # tuple

        if args.uncertainty_type == "entropy" or args.uncertainty_type == "aci":
            self.vis_size = (2, args.lidar_map_size, args.lidar_map_size)
            args.conv_lstm_input_dim = 2
        else:
            self.vis_size = (1, args.lidar_map_size, args.lidar_map_size)

        self.vis_seq = args.vis_seq
        c,w,h = self.vis_size
        obs_shape = get_shape_from_obs_space(obs_space)
        
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        vis_base = CNNBase
        vec_base = MLPBase

        self.base = base(args, obs_shape)
        #0: free, 1: static, 2: dynamic, 3:robot, 4:unkown
        # self.vis_base=vis_base(args,self.vis_size)
        # ConvLSTM parameters
        self.conv_lstm = ConvLSTM(
            input_dim=args.conv_lstm_input_dim,  # Number of input channels
            hidden_dim=args.conv_lstm_hidden_dim,  # List of hidden dims for each layer
            kernel_size=(3, 3),
            num_layers=args.conv_lstm_num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )
        self.ego_vec_base = vec_base(args,(self.ego_vec_size,))
        self.other_vec_base = vec_base(args,(self.other_vec_size,))
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size*2+1*h*w, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args) #

        self.to(device)
        self.algo = args.algorithm_name

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        actions = []
        action_log_probs = []
        rnn_states_new = rnn_states.view(-1,self.num_agents,self._recurrent_N,rnn_states.shape[-1])
        c,w,h = self.vis_size
        for agent_id in range(self.num_agents):

            obs_i = obs.view(-1, self.num_agents, obs.shape[-1])[:, agent_id]
            B, obs_dim = obs_i.shape
            c,h,w = self.vis_size
            vis_obs = obs_i[:, :self.vis_seq*c*h*w].reshape(B, self.vis_seq, c, h, w)
            vec_obs = obs_i[:, self.vis_seq*c*h*w:].reshape(B, self.vis_seq, (self.ego_vec_size+self.other_vec_size))
            
            #no relative vec
            
            # ego_vec_obs = vec_obs[:, -1, :self.ego_vec_size]
            # other_vec_obs = vec_obs[:, -1, self.ego_vec_size:self.ego_vec_size+self.other_vec_size]
            
            # conv_lstm_output, _ = self.conv_lstm(vis_obs)
            # vis_x = conv_lstm_output[0].mean(dim=1).view(B, -1) # Average across time steps
            # ego_vec_x = self.ego_vec_base(ego_vec_obs)
            # other_vec_x = self.other_vec_base(other_vec_obs)


            #relative vec
            
            ego_vec_obs = vec_obs[:, -1, :self.ego_vec_size]
            other_vec_obs = vec_obs[:, -1, self.ego_vec_size:self.ego_vec_size+self.other_vec_size]
            relative_other_vec_obs = other_vec_obs - ego_vec_obs.repeat(1,2)
            conv_lstm_output, _ = self.conv_lstm(vis_obs)
            vis_x = conv_lstm_output[0].mean(dim=1).view(B, -1) # Average across time steps
            ego_vec_x = self.ego_vec_base(ego_vec_obs)
            other_vec_x = self.other_vec_base(relative_other_vec_obs)

            #static dynamic
            
            # s_vis_obs = ((vis_obs == 2) | (vis_obs == 4)).float()
            # d_vis_obs = ((vis_obs == 1) | (vis_obs == 3)).float()
            # conv_lstm_output, _ = self.conv_lstm(d_vis_obs)
            # s_vis_x = s_vis_obs[:, -1].view(B, -1)
            # vis_x = conv_lstm_output[0].mean(dim=1).view(B, -1) + s_vis_x # Average across time steps
            
            # ego_vec_obs = vec_obs[:, -1, :self.ego_vec_size]
            # other_vec_obs = vec_obs[:, -1, self.ego_vec_size:self.ego_vec_size+self.other_vec_size]
            # relative_other_vec_obs = other_vec_obs - ego_vec_obs.repeat(1,2)
            # ego_vec_x = self.ego_vec_base(ego_vec_obs)
            # other_vec_x = self.other_vec_base(relative_other_vec_obs)

            actor_features = torch.cat([vis_x,ego_vec_x,other_vec_x], dim=-1)
            if available_actions is not None:
                available_actions = check(available_actions).to(**self.tpdv)
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                actor_features, rnn_states_i = self.rnn(actor_features, rnn_states.view(-1,self.num_agents,self._recurrent_N,rnn_states.shape[-1])[:,agent_id], masks.view(-1,self.num_agents,masks.shape[-1])[:,agent_id])
                rnn_states_new[:,agent_id] = rnn_states_i
            action, action_log_prob= self.act(actor_features, available_actions, deterministic)
            actions.append(action)
            action_log_probs.append(action_log_prob)
            
        actions = torch.stack(actions,dim=1).view(-1,actions[0].shape[-1])
        action_log_probs = torch.stack(action_log_probs,dim=1).view(-1,action_log_probs[0].shape[-1])

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            rnn_states=rnn_states_new.view(-1,self._recurrent_N,rnn_states.shape[-1])
        
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        action_log_probs = []
        dist_entropys = []
        rnn_states_new = rnn_states.view(-1,self.num_agents,self._recurrent_N,rnn_states.shape[-1])
        
        c,w,h = self.vis_size
        if available_actions is not None:
            # None
            available_actions = check(available_actions).to(**self.tpdv)
            available_actions=available_actions.view(-1,self.num_agents,available_actions.shape[-1])
            
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv) # 960 1
            active_masks=active_masks.view(-1,self.num_agents,self.episode_length//self.num_mini_batch,active_masks.shape[-1]) # 16 3 20 1
        
        for agent_id in range(self.num_agents):
            
            obs_i = obs.view(-1, self.num_agents, obs.shape[-1])[:, agent_id]
            B, obs_dim = obs_i.shape
            c,h,w=self.vis_size
            vis_obs = obs_i[:, :self.vis_seq*c*h*w].reshape(B, self.vis_seq, c, h, w)
            vec_obs = obs_i[:, self.vis_seq*c*h*w:].reshape(B, self.vis_seq, (self.ego_vec_size+self.other_vec_size))
            
            # #no relative vec
            
            # ego_vec_obs = vec_obs[:, -1, :self.ego_vec_size]
            # other_vec_obs = vec_obs[:, -1, self.ego_vec_size:self.ego_vec_size+self.other_vec_size]
            # conv_lstm_output, _ = self.conv_lstm(vis_obs)
            # vis_x = conv_lstm_output[0].mean(dim=1).view(B, -1) # Average across time steps
            # ego_vec_x = self.ego_vec_base(ego_vec_obs)
            # other_vec_x = self.other_vec_base(other_vec_obs)

            # #relative vec
            
            ego_vec_obs = vec_obs[:, -1, :self.ego_vec_size]
            other_vec_obs = vec_obs[:, -1, self.ego_vec_size:self.ego_vec_size+self.other_vec_size]
            relative_other_vec_obs = other_vec_obs - ego_vec_obs.repeat(1, 2)
            conv_lstm_output, _ = self.conv_lstm(vis_obs)
            vis_x = conv_lstm_output[0].mean(dim=1).view(B, -1) # Average across time steps
            ego_vec_x = self.ego_vec_base(ego_vec_obs)
            other_vec_x = self.other_vec_base(relative_other_vec_obs)

            #static dynamic
            # s_vis_obs = ((vis_obs == 2) | (vis_obs == 4)).float()
            # d_vis_obs = ((vis_obs == 1) | (vis_obs == 3)).float()
            # conv_lstm_output, _ = self.conv_lstm(d_vis_obs)
            # s_vis_x = s_vis_obs[:, -1].view(B, -1)
            # vis_x = conv_lstm_output[0].mean(dim=1).view(B, -1) + s_vis_x # Average across time steps
            
            # ego_vec_obs = vec_obs[:, -1, :self.ego_vec_size]
            # other_vec_obs = vec_obs[:, -1, self.ego_vec_size:self.ego_vec_size+self.other_vec_size]
            # relative_other_vec_obs = other_vec_obs - ego_vec_obs.repeat(1,2)
            # ego_vec_x = self.ego_vec_base(ego_vec_obs)
            # other_vec_x = self.other_vec_base(relative_other_vec_obs)

            actor_features = torch.cat([vis_x, ego_vec_x, other_vec_x], dim=-1)
            
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                actor_features, rnn_states_i = self.rnn(actor_features, rnn_states.view(-1,self.num_agents,self._recurrent_N,rnn_states.shape[-1])[:,agent_id], masks.view(-1,self.num_agents,masks.shape[-1])[:,agent_id])
                rnn_states_new[:,agent_id]=rnn_states_i
            if self.algo == "hatrpo":
                action_log_probs, dist_entropy ,action_mu, action_std, all_probs= self.act.evaluate_actions_trpo(actor_features,
                                                                        action, available_actions,
                                                                        active_masks=
                                                                        active_masks[:,agent_id] if self._use_policy_active_masks
                                                                        else None)

                return action_log_probs, dist_entropy, action_mu, action_std, all_probs
            else:
                action_log_prob, dist_entropy = self.act.evaluate_actions(actor_features, action.view(-1,self.num_agents,action.shape[-1])[:,agent_id], available_actions, active_masks[:,agent_id].flatten().unsqueeze(-1) if active_masks is not None else None)
                #action_log_prob, dist_entropy = self.act.evaluate_actions(actor_features, action.view(-1,self.num_agents,action.shape[-1])[:,agent_id], available_actions, active_masks[:,agent_id].flatten())
                # 320 1 []
                action_log_probs.append(action_log_prob)
                dist_entropys.append(dist_entropy)
                
        action_log_probs = torch.stack(action_log_probs,dim=1).view(-1,action_log_probs[0].shape[-1])
        dist_entropys=torch.mean(torch.stack(dist_entropys))

        #return action_log_probs, dist_entropy
        return action_log_probs, torch.tensor(1.0)

class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)
        
        self.ego_vec_size = args.ego_vec_size
        self.other_vec_size = args.other_vec_size
        # self.vis_size=(1, args.lidar_map_size, args.lidar_map_size) # tuple

        if args.uncertainty_type == "entropy" or args.uncertainty_type == "aci":
            self.vis_size = (2, args.lidar_map_size, args.lidar_map_size)
            args.conv_lstm_input_dim = 2
        else:
            self.vis_size = (1, args.lidar_map_size, args.lidar_map_size)

        c,w,h = self.vis_size
        self.vis_seq = args.vis_seq
        
        vis_base = CNNBase
        vec_base = MLPBase
        self.vis_base = vis_base(args,self.vis_size)
        self.ego_vec_base = vec_base(args,(self.ego_vec_size,))
        self.other_vec_base = vec_base(args,(self.other_vec_size,))

        self.conv_lstm = ConvLSTM(
            input_dim=args.conv_lstm_input_dim,  # Number of input channels
            hidden_dim=args.conv_lstm_hidden_dim,  # List of hidden dims for each layer
            kernel_size=(3, 3),
            num_layers=args.conv_lstm_num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size*2+1*h*w, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        c, w, h = self.vis_size
        # print(cent_obs.shape) # torch.Size([B, 12605])
        B, cent_obs_dim = cent_obs.shape
        vis_obs = cent_obs[:, :self.vis_seq*c*h*w].reshape(B, self.vis_seq, c, h, w)
        vec_obs = cent_obs[:, self.vis_seq*c*h*w:].reshape(B, self.vis_seq, (self.ego_vec_size+self.other_vec_size))
        ego_vec_obs = vec_obs[:, -1, :self.ego_vec_size]
        other_vec_obs = vec_obs[:, -1, self.ego_vec_size:self.ego_vec_size+self.other_vec_size]
        
        # # no relative vec
        
        # conv_lstm_output, _ = self.conv_lstm(vis_obs)
        # vis_x = conv_lstm_output[0].mean(dim=1).view(B, -1)
        # ego_vec_x = self.ego_vec_base(ego_vec_obs)
        # other_vec_x = self.other_vec_base(other_vec_obs)

        # # relative vec

        relative_other_vec_obs = other_vec_obs - ego_vec_obs.repeat(1, 2)
        conv_lstm_output, _ = self.conv_lstm(vis_obs)
        vis_x = conv_lstm_output[0].mean(dim=1).view(B, -1)
        ego_vec_x = self.ego_vec_base(ego_vec_obs)
        other_vec_x = self.other_vec_base(relative_other_vec_obs)

        # static dynamic
        # s_vis_obs = ((vis_obs == 2) | (vis_obs == 4)).float()
        # d_vis_obs = ((vis_obs == 1) | (vis_obs == 3)).float()
        # conv_lstm_output, _ = self.conv_lstm(d_vis_obs)
        # s_vis_x = s_vis_obs[:, -1].view(B, -1)
        # vis_x = conv_lstm_output[0].mean(dim=1).view(B, -1) + s_vis_x # Average across time steps
        
        # ego_vec_obs = vec_obs[:, -1, :self.ego_vec_size]
        # other_vec_obs = vec_obs[:, -1, self.ego_vec_size:self.ego_vec_size+self.other_vec_size]
        # relative_other_vec_obs = other_vec_obs - ego_vec_obs.repeat(1,2)
        # ego_vec_x = self.ego_vec_base(ego_vec_obs)
        # other_vec_x = self.other_vec_base(relative_other_vec_obs)

        critic_features = torch.cat([vis_x, ego_vec_x, other_vec_x], dim=-1)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
