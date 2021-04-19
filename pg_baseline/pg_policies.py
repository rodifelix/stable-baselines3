from typing import Any, Callable, Dict, List, Optional, Type, Union

import gym
import torch as th
from torch import nn
import torchvision
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from pg_baseline import pg_densenet

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, create_mlp

class PGQNetwork(BasePolicy):
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
    ):
        super(PGQNetwork, self).__init__(
            observation_space,
            action_space,
            normalize_images=False,
        )

        #TODO: assert observation space and action space specification is compatible with heightmap resolution

        self.heightmap_resolution = heightmap_resolution

        # Initialize network trunks with our version of DenseNet with InstanceNormalization
        self.push_color_trunk = pg_densenet.PGdensenet121()
        self.push_depth_trunk = pg_densenet.PGdensenet121()
        self.grasp_color_trunk = pg_densenet.PGdensenet121()
        self.grasp_depth_trunk = pg_densenet.PGdensenet121()

        self.num_rotations = 16

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

        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.InstanceNorm2d(2048, affine=True)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.InstanceNorm2d(64, affine=True)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            # ('grasp-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.InstanceNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        diag_length = float(heightmap_resolution*2) * np.sqrt(2)
        interm_feat_resolution = int(np.ceil(diag_length/32))
        diag_length = np.ceil(diag_length/32)*32
        self.padding_width = int((diag_length - heightmap_resolution*2)/2)

        self.flow_grid_before = self._build_flow_grid_before(int(diag_length))
        self.flow_grid_after = self._build_flow_grid_after((16, self.push_color_trunk.num_output_features, interm_feat_resolution, interm_feat_resolution))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.
        :param obs: Observation. Expects Tensor with shape N x C x H x W
        :return: The estimated Q-Value for each action.
        """
        batch_size = obs.shape[0]

        input_color_data, input_depth_data = th.narrow(obs, 1, 0, 3), th.narrow(obs, 1, 3, 3)

         # Pass input data through model
        output_prob = self._forward(input_color_data, input_depth_data)

        return th.reshape(output_prob, (batch_size, 2*self.num_rotations*self.heightmap_resolution*self.heightmap_resolution))
        
    def forward_specific_rotations(self, obs: th.Tensor, rotation_indices: th.Tensor):
        assert obs.shape[0] == rotation_indices.shape[0], "Number of observations does not match number of rotation indices"

        batch_size = obs.shape[0]
       
        input_color_data, input_depth_data = th.narrow(obs, 1, 0, 3), th.narrow(obs, 1, 3, 3)

        output_prob = self._forward(input_color_data, input_depth_data, rotation_indices=rotation_indices)

        return th.reshape(output_prob, (batch_size, 2*self.heightmap_resolution*self.heightmap_resolution))


    def _forward(self, input_color_data, input_depth_data, rotation_indices=None):
        """forward pass wrapper for external call

        Args:
            input_color_data: color data as input
            input_depth_data: depth data as input
            is_volatile (bool, optional): true if gradients should be generated for subsequent backward pass. Defaults to False.

        Returns:
            qmaps, features
        """
        batch_size = input_color_data.shape[0]
        if rotation_indices is None:
            batch_flow_grid_before = self.flow_grid_before.repeat((batch_size, 1, 1, 1))
            batch_flow_grid_after = self.flow_grid_after.repeat((batch_size, 1, 1, 1))
            input_color_datas = th.repeat_interleave(input_color_data, 16, dim=0)
            input_depth_datas = th.repeat_interleave(input_depth_data, 16, dim=0)
        else:
            batch_flow_grid_before, batch_flow_grid_after = self._get_flow_grids_for_indices(rotation_indices)
            input_color_datas = input_color_data
            input_depth_datas = input_depth_data

        batch_flow_grid_before = batch_flow_grid_before.to(self.device)
        batch_flow_grid_after = batch_flow_grid_after.to(self.device)

        # Rotate images clockwise
        rotated_color_images = F.grid_sample(Variable(input_color_datas, requires_grad=False).to(self.device), batch_flow_grid_before, mode='nearest')
        rotated_depth_images = F.grid_sample(Variable(input_depth_datas, requires_grad=False).to(self.device), batch_flow_grid_before, mode='nearest')

        
        # Compute intermediate features
        interm_push_color_feat = self.push_color_trunk.features(rotated_color_images)
        interm_push_depth_feat = self.push_depth_trunk.features(rotated_depth_images)
        interm_push_feat = th.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)

        interm_grasp_color_feat = self.grasp_color_trunk.features(rotated_color_images)
        interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotated_depth_images)
        interm_grasp_feat = th.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)

        # Forward pass through branches, undo rotation on output predictions, upsample results
        output_prob = th.cat((nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).forward(F.grid_sample(self.pushnet(interm_push_feat), batch_flow_grid_after, mode='nearest')),
                            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).forward(F.grid_sample(self.graspnet(interm_grasp_feat), batch_flow_grid_after, mode='nearest'))), dim=0)

        output_prob = th.narrow(output_prob, dim=2, start=int(self.padding_width/2), length=self.heightmap_resolution)
        output_prob = th.narrow(output_prob, dim=3, start=int(self.padding_width/2), length=self.heightmap_resolution)

        return output_prob

    def _build_flow_grid_after(self, interm_push_feat_size):
        affine_mat_after = None
        for rotate_idx in range(self.num_rotations):
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))
            # Compute sample grid for rotation AFTER branches
            tmp_mat = np.asarray([[np.cos(rotate_theta), np.sin(
                rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            tmp_mat.shape = (2, 3, 1)
            tmp_mat = th.from_numpy(
                tmp_mat).permute(2, 0, 1).float()

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
            tmp_mat = np.asarray(
                [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            tmp_mat.shape = (2, 3, 1)
            tmp_mat = th.from_numpy(
                tmp_mat).permute(2, 0, 1).float()
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
        q_values = self.forward(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                heightmap_resolution=self.heightmap_resolution,          
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
        heightmap_resolution: int,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.SGD,
        optimizer_kwargs: Optional[Dict[str, Any]] = {
            "lr": 1e-4,
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

        self.heightmap_resolution = heightmap_resolution

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "heightmap_resolution": self.heightmap_resolution,
        }

        self.q_net, self.q_net_target = None, None
        self._build(lr_schedule)


    def _build(self, lr_schedule: Callable) -> None:
        """
        Create the network and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

    def make_q_net(self) -> PGQNetwork:
        #TODO: DO WE NEED THIS?
        # Make sure we always have separate networks for features extractors etc
        # net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return PGQNetwork(**self.net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                heightmap_resolution=self.heightmap_resolution,
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
            )
        )
        return data

register_policy("PGDQNPolicy", PGDQNPolicy)