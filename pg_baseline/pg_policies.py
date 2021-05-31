from typing import Any, Callable, Dict, List, Optional, Type, Union

import gym
import torch as th
from torch import nn
import torchvision
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from pg_baseline import pg_hourglass

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

        self.heightmap_resolution = heightmap_resolution

        self.num_rotations = 8

        params = pg_hourglass.get_pibn_parameters()

        params['output_channels'] = self.num_rotations

        self.net = pg_hourglass.Push_Into_Box_Net(params)


    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.
        :param obs: Observation. Expects Tensor with shape N x C x H x W
        :return: The estimated Q-Value for each action.
        """
        batch_size = obs.shape[0]

        output_prob = self.net.forward(obs.to(self.device))

        output_prob = self._mask(obs, output_prob)

        return th.reshape(output_prob, (batch_size, self.num_rotations*self.heightmap_resolution*self.heightmap_resolution))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:        
        q_values = self.forward(observation)
        if deterministic:
            # Greedy action
            action = q_values.argmax(dim=1).reshape(-1)
        else:
            q_values = q_values.reshape(self.num_rotations, self.heightmap_resolution*self.heightmap_resolution)
            rand_rotation = np.random.randint(0,self.num_rotations)
            print("Exploring: Random rotation index:", rand_rotation)
            q_value_slice = th.narrow(q_values, dim=0, start=rand_rotation, length=1)
            action = q_value_slice.argmax(dim=1).reshape(-1) + rand_rotation*self.heightmap_resolution*self.heightmap_resolution
        return action

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                heightmap_resolution=self.heightmap_resolution,          
            )
        )
        return data

    def _mask(self, obs: th.Tensor, output_prob: th.Tensor):
        threshhold_depth = 0.01
        depth_min = 0.0
        depth_max = 0.1
        threshold_norm = (threshhold_depth - depth_min)/(depth_max - depth_min)

        masked_input = obs > threshold_norm
        masked_input = masked_input.type(th.float)
        diluted_mask = th.nn.functional.max_pool2d(input=masked_input,kernel_size=(28,28),stride=(1,1),padding=14)

        diluted_mask = th.narrow(diluted_mask, 2, 0, self.heightmap_resolution)
        diluted_mask = th.narrow(diluted_mask, 3, 0, self.heightmap_resolution)

        diluted_mask = diluted_mask - 1.
        diluted_mask = diluted_mask * th.finfo(th.float).max

        #diluted_mask = th.nan_to_num(diluted_mask)

        diluted_mask = th.repeat_interleave(diluted_mask, self.num_rotations, dim=1)

        return output_prob + diluted_mask


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