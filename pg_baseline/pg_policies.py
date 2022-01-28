from typing import Any, Callable, Dict, List, Optional, Type, Union

import gym
from pg_baseline.NoisyConv2d import NoisyConv2d
from pg_baseline.NoisyLinear import NoisyLinear
import torch as th
from torch import nn
from collections import OrderedDict
import numpy as np
from pg_baseline import pg_hourglass, pg_mask_net, pg_densenet
from torch.autograd import Variable
import torch.nn.functional as F
from stable_baselines3.common.policies import BasePolicy, register_policy

class HGNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN
    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        heightmap_resolution: int,
        num_rotations: int,
        ucb_confidence: float,
        use_masknet: bool,
        preload_mask_path: Optional[str],
        dueling: bool,
        noisy: bool,
    ):
        super(HGNetwork, self).__init__(
            observation_space,
            action_space,
            normalize_images=False,
        )

        self.heightmap_resolution = heightmap_resolution

        self.num_rotations = num_rotations

        params = pg_hourglass.get_pibn_parameters()

        params['resolution'] = heightmap_resolution
        params['output_channels'] = self.num_rotations
        params['dueling'] = dueling
        params['noisy'] = noisy

        self.net = pg_hourglass.Push_Into_Box_Net(params)

        self.noisy_layers = [module for module in self.net.modules() if isinstance(module, (NoisyConv2d, NoisyLinear))]

        self.ucb_confidence = ucb_confidence

        if self.ucb_confidence > 0:
            self.timestep = 1
            self.action_counter = np.ones((self.num_rotations), dtype=np.int)
        
        self.use_masknet = use_masknet

        if self.use_masknet:
            params_mask = pg_mask_net.get_psp_parameters()

            params_mask['output_channels'] = self.num_rotations

            self.mask_net = pg_mask_net.PushingSuccessPredictor(params_mask)

            self.mask_threshhold = 0.14

            self.preload_mask_path = preload_mask_path

            if self.preload_mask_path is not None:
                mask_state_dict = th.load(preload_mask_path)
                self.mask_net.load_state_dict(mask_state_dict, strict=True)



    def forward(self, obs: th.Tensor, mask=True) -> th.Tensor:
        """
        Predict the q-values.
        :param obs: Observation. Expects Tensor with shape N x C x H x W
        :return: The estimated Q-Value for each action.
        """
        batch_size = obs.shape[0]

        output_prob = self.net.forward(obs.to(self.device))

        if mask:
            if self.use_masknet:
                output_prob = self._mask(obs, output_prob)
            else:
                output_prob = self.explore_mask(obs, output_prob)

        return th.reshape(output_prob, (batch_size, self.num_rotations*self.heightmap_resolution*self.heightmap_resolution))

    def mask(self, obs: th.Tensor) -> th.Tensor:
        mask_output = self.mask_net.forward(obs)
        return th.reshape(mask_output, (obs.shape[0], self.num_rotations*self.heightmap_resolution*self.heightmap_resolution))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:        
        if deterministic:
            q_values = self.forward(observation)
            if self.ucb_confidence > 0:
                # UCB
                q_values = np.reshape(q_values.detach().cpu().numpy(), (self.num_rotations, -1))
                action_idxs = q_values.argmax(axis=1)
                actions = q_values.max(axis=1)
                confidence_value = self.ucb_confidence * np.sqrt(np.log(self.timestep)/self.action_counter)
                actions += confidence_value
                rotation = actions.argmax()
                action = rotation*self.heightmap_resolution*self.heightmap_resolution + action_idxs[rotation]
                action = th.tensor([action])
                self.action_counter[rotation] += 1
                self.timestep += 1
            else:
                action = q_values.argmax(dim=1)            
            
        else:
            actions = th.arange(start=0, end=self.num_rotations*self.heightmap_resolution*self.heightmap_resolution, dtype=th.long).to(self.device)
            actions = actions.reshape((1, self.num_rotations, self.heightmap_resolution, self.heightmap_resolution))

            masked_actions = self.explore_mask(observation, actions)

            valid_idx = masked_actions[masked_actions >= 0]
            if len(valid_idx) > 0:
                choice = np.random.randint(low=0, high=len(valid_idx))
                valid_idx = valid_idx.type(th.long)
                action = valid_idx[choice].reshape(-1)
            else:
                raise NoObjectsInSceneException("No valid actions after mask. Is scene empty?")

            print("\n\nExploring in next iteration: Random action index:", action)
        return action

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                heightmap_resolution=self.heightmap_resolution,
                num_rotations=self.num_rotations,
                ucb_confidence=self.ucb_confidence,
                preload_mask_path=self.preload_mask_path,
                use_masknet=self.use_masknet,         
            )
        )
        return data

    def _mask(self, obs: th.Tensor, output_prob: th.Tensor):
        mask =  self.mask_net.forward(obs)

        mask = mask >= self.mask_threshhold
        mask = mask.float() - 1.
        mask = mask * th.finfo(th.float).max

        return output_prob + mask

    def explore_mask(self, obs: th.Tensor, output_prob: th.Tensor):
        threshhold_depth = 0.01
        depth_min = 0.0
        depth_max = 0.1
        threshold_norm = (threshhold_depth - depth_min)/(depth_max - depth_min)

        masked_input = obs > threshold_norm
        masked_input = masked_input.type(th.float).to(self.device)
        diluted_mask = th.nn.functional.max_pool2d(input=masked_input,kernel_size=(34,34),stride=(1,1),padding=17)

        diluted_mask = th.narrow(diluted_mask, 2, 1, self.heightmap_resolution)
        diluted_mask = th.narrow(diluted_mask, 3, 1, self.heightmap_resolution)

        diluted_mask = diluted_mask - 1.
        diluted_mask = diluted_mask * th.finfo(th.float).max

        #diluted_mask = th.nan_to_num(diluted_mask)

        diluted_mask = th.repeat_interleave(diluted_mask, self.num_rotations, dim=1)

        return output_prob + diluted_mask

    def reset_noise(self):
        for noisy_module in self.noisy_layers:
            noisy_module.reset_noise()



class VPGNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN
    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        heightmap_resolution: int,
        num_rotations: int,
        ucb_confidence: float = 0,

    ):
        super(VPGNetwork, self).__init__(
            observation_space,
            action_space,
            normalize_images=False,
        )

        self.heightmap_resolution = heightmap_resolution

        # Initialize network trunks with our version of DenseNet with InstanceNormalization
        self.push_color_trunk = pg_densenet.PGdensenet121()
        self.push_depth_trunk = pg_densenet.PGdensenet121()

        self.num_rotations = num_rotations

        # Construct network branches for pushing and grasping
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.InstanceNorm2d(2048, affine=True)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.InstanceNorm2d(64, affine=True)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.InstanceNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        diag_length = float(observation_space.shape[1]) * np.sqrt(2)
        interm_feat_resolution = int(np.ceil(diag_length/32))
        diag_length = np.ceil(diag_length/32)*32
        self.padding_width = int((diag_length - observation_space.shape[1])/2)

        self.flow_grid_before = self._build_flow_grid_before(int(diag_length))
        self.flow_grid_after = self._build_flow_grid_after((self.num_rotations, self.push_color_trunk.num_output_features, interm_feat_resolution, interm_feat_resolution))

    def forward(self, obs: th.Tensor, mask=True) -> th.Tensor:
        """
        Predict the q-values.
        :param obs: Observation. Expects Tensor with shape N x C x H x W
        :return: The estimated Q-Value for each action.
        """
        batch_size = obs.shape[0]

        input_color_data, input_depth_data = th.narrow(obs, 1, 0, 3), th.narrow(obs, 1, 3, 1)

        # Pass input data through model
        output_prob = self._forward(input_color_data, input_depth_data)

        return th.reshape(output_prob, (batch_size, self.num_rotations*self.heightmap_resolution*self.heightmap_resolution))

    def forward_specific_rotations(self, obs: th.Tensor, rotation_indices: th.Tensor):
        assert obs.shape[0] == rotation_indices.shape[0], "Number of observations does not match number of rotation indices"

        batch_size = obs.shape[0]
       
        input_color_data, input_depth_data = th.narrow(obs, 1, 0, 3), th.narrow(obs, 1, 3, 1)

        output_prob = self._forward(input_color_data, input_depth_data, rotation_indices=rotation_indices)

        return th.reshape(output_prob, (batch_size, self.heightmap_resolution*self.heightmap_resolution))

    def _forward(self, input_color_data, input_depth_data, rotation_indices=None):
        """forward pass wrapper for external call

        Args:
            input_color_data: color data as input
            input_depth_data: depth data as input

        Returns:
            qmaps, features
        """
        #copy depth input to 3-channels
        depth_3_channel = th.repeat_interleave(input_depth_data, 3, dim=1)

        padded_color = th.nn.functional.pad(input_color_data, (self.padding_width,self.padding_width,self.padding_width,self.padding_width), mode='constant', value=0)
        padded_depth = th.nn.functional.pad(depth_3_channel, (self.padding_width,self.padding_width,self.padding_width,self.padding_width), mode='constant', value=0)

        batch_size = input_color_data.shape[0]
        if rotation_indices is None:
            batch_flow_grid_before = self.flow_grid_before.repeat((batch_size, 1, 1, 1))
            batch_flow_grid_after = self.flow_grid_after.repeat((batch_size, 1, 1, 1))
            input_color_datas = th.repeat_interleave(padded_color, self.num_rotations, dim=0)
            input_depth_datas = th.repeat_interleave(padded_depth, self.num_rotations, dim=0)
        else:
            batch_flow_grid_before, batch_flow_grid_after = self._get_flow_grids_for_indices(rotation_indices)
            input_color_datas = padded_color
            input_depth_datas = padded_depth

        batch_flow_grid_before = batch_flow_grid_before.to(self.device)
        batch_flow_grid_after = batch_flow_grid_after.to(self.device)

        rotated_color_images = F.grid_sample(Variable(input_color_datas, requires_grad=False).to(self.device), batch_flow_grid_before, mode='nearest')
        rotated_depth_images = F.grid_sample(Variable(input_depth_datas, requires_grad=False).to(self.device), batch_flow_grid_before, mode='nearest')
        
        # Compute intermediate features
        interm_push_color_feat = self.push_color_trunk.features(rotated_color_images)
        interm_push_depth_feat = self.push_depth_trunk.features(rotated_depth_images)
        interm_push_feat = th.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)

        # Forward pass through branches, undo rotation on output predictions, upsample results
        output_prob = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).forward(F.grid_sample(self.pushnet(interm_push_feat), batch_flow_grid_after, mode='nearest'))
        output_prob = th.narrow(output_prob, dim=2, start=int(self.padding_width/2), length=self.heightmap_resolution)
        output_prob = th.narrow(output_prob, dim=3, start=int(self.padding_width/2), length=self.heightmap_resolution)
        return output_prob

    def _build_flow_grid_after(self, interm_push_feat_size):
        affine_mat_after = None
        for rotate_idx in range(self.num_rotations):
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))
            # Compute sample grid for rotation AFTER branches
            tmp_mat = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            tmp_mat.shape = (2, 3, 1)
            tmp_mat = th.from_numpy(tmp_mat).permute(2, 0, 1).float()

            if affine_mat_after is None:
                affine_mat_after = tmp_mat
            else:
                affine_mat_after = th.cat((affine_mat_after, tmp_mat), dim=0)

        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).to(self.device), interm_push_feat_size)
        return flow_grid_after

    def _build_flow_grid_before(self, diag_length):
        affine_mat_before = None
        for rotate_idx in range(self.num_rotations):
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample grid for rotation BEFORE neural network
            tmp_mat = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            tmp_mat.shape = (2, 3, 1)
            tmp_mat = th.from_numpy(tmp_mat).permute(2, 0, 1).float()
            if affine_mat_before is None:
                affine_mat_before = tmp_mat
            else:
                affine_mat_before = th.cat((affine_mat_before, tmp_mat), dim=0)

        flow_grid_before = F.affine_grid(Variable(affine_mat_before).to(self.device), (self.num_rotations, 3, diag_length, diag_length))
        return flow_grid_before

    def _get_flow_grids_for_indices(self, rotation_indices: th.Tensor):
        flow_grids_before = th.index_select(self.flow_grid_before.to(self.device), dim=0, index=th.squeeze(rotation_indices))
        flow_grids_after = th.index_select(self.flow_grid_after.to(self.device), dim=0, index=th.squeeze(rotation_indices))
        return flow_grids_before, flow_grids_after


    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        if deterministic:
            q_values = self.forward(observation)
            # Greedy action
            action = q_values.argmax(dim=1).reshape(-1)
        else:
            action = th.as_tensor([self.action_space.sample()], dtype=th.int64)
            print("\n\nExploring in next iteration: Random action index:", action)
        return action

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                heightmap_resolution=self.heightmap_resolution,
                num_rotations=self.num_rotations,
            )
        )
        return data

class PGDQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        use_target: bool,
        heightmap_resolution: int,
        num_rotations: int,
        net_class: str,
        ucb_confidence: float,
        dueling: bool = False,
        noisy: bool = True,
        mask_lr: float = 1e-4,
        preload_mask_path: Optional[str] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.SGD,
        optimizer_kwargs: Optional[Dict[str, Any]] = {
            "momentum": 0.9,
            "weight_decay": 2e-5,
        },
    ):
        super(PGDQNPolicy, self).__init__(
            observation_space,
            action_space,
            normalize_images=False,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        #TODO: maybe assert size of obsvertion and action space match heightmap and num rotations

        self.heightmap_resolution = heightmap_resolution
        self.num_rotations = num_rotations
        self.ucb_confidence = ucb_confidence
        self.use_target = use_target
        self.net_class = net_class
        self.preload_mask_path = preload_mask_path
        self.dueling = dueling
        self.noisy = noisy

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "heightmap_resolution": self.heightmap_resolution,
            "num_rotations": self.num_rotations,
            "ucb_confidence": self.ucb_confidence,
            "dueling": self.dueling,
            "noisy": self.noisy,
        }
        if self.net_class == "HG_Mask":
            self.net_args["preload_mask_path"] = self.preload_mask_path
            self.net_args["use_masknet"] = True
            self.mask_lr = mask_lr
        elif self.net_class == "HG":
            self.net_args["preload_mask_path"] = None
            self.net_args["use_masknet"] = False

        self.q_net, self.q_net_target = None, None
        self._build(lr_schedule)


    def _build(self, lr_schedule: Callable) -> None:
        """
        Create the network and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        if self.use_target:
            self.q_net_target = self.make_q_net()
            self.q_net_target.load_state_dict(self.q_net.state_dict())

        # Setup optimizer with initial learning rate
        if self.net_class == "HG_Mask":
            self.optimizer = self.optimizer_class(self.q_net.net.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
            self.mask_optimizer = th.optim.Adam(self.q_net.mask_net.parameters(), lr=self.mask_lr)
        else:
            self.optimizer = self.optimizer_class(self.q_net.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


    def make_q_net(self) -> BasePolicy:
        if self.net_class == "HG" or self.net_class == "HG_Mask":
            return HGNetwork(**self.net_args).to(self.device)
        elif self.net_class == "VPG":
            return VPGNetwork(**self.net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def reset_noise(self):
        if self.net_class == "HG" or self.net_class == "HG_Mask":
            self.q_net.reset_noise()

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                heightmap_resolution=self.heightmap_resolution,
                num_rotations=self.num_rotations,
                ucb_confidence=self.ucb_confidence,  
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                use_target=self.use_target,
                net_class=self.net_class,
                preload_mask_path=self.preload_mask_path,
                dueling=self.dueling,
                noisy=self.noisy,
                mask_lr = self.mask_lr
            )
        )
        return data

class NoObjectsInSceneException(Exception):
    pass


register_policy("PGDQNPolicy", PGDQNPolicy)