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
        num_rotations: int,
        ucb_confidence: float
    ):
        super(PGQNetwork, self).__init__(
            observation_space,
            action_space,
            normalize_images=False,
        )

        self.heightmap_resolution = heightmap_resolution

        self.num_rotations = num_rotations

        params = pg_hourglass.get_pibn_parameters()

        params['output_channels'] = self.num_rotations

        self.net = pg_hourglass.Push_Into_Box_Net(params)

        self.action_counter = np.ones((self.num_rotations), dtype=np.int)

        self.ucb_confidence = ucb_confidence

        self.timestep = 1


    def forward(self, obs: th.Tensor, mask=True) -> th.Tensor:
        """
        Predict the q-values.
        :param obs: Observation. Expects Tensor with shape N x C x H x W
        :return: The estimated Q-Value for each action.
        """
        batch_size = obs.shape[0]

        output_prob = self.net.forward(obs.to(self.device))

        if mask:
            output_prob = self._mask(obs, output_prob)

        return th.reshape(output_prob, (batch_size, self.num_rotations*self.heightmap_resolution*self.heightmap_resolution))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:        
        if deterministic:
            q_values = self.forward(observation).detach().cpu().numpy()
            q_values = np.reshape(q_values, (self.num_rotations, -1))
            # UCB
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
            actions = th.arange(start=0, end=self.num_rotations*self.heightmap_resolution*self.heightmap_resolution, dtype=th.long).to(self.device)
            actions = actions.reshape((1, self.num_rotations, self.heightmap_resolution, self.heightmap_resolution))

            masked_actions = self._mask(observation, actions)

            valid_idx = masked_actions[masked_actions >= 0]
            if len(valid_idx) > 0:
                choice = np.random.randint(low=0, high=len(valid_idx))
                valid_idx = valid_idx.type(th.long)
                action = valid_idx[choice].reshape(-1)
            else:
                print("\n\n No valid actions after mask. Is scene empty? Returning action 0")
                action = th.zeros([1], device=self.device, dtype=th.int64)

            print("\n\nExploring in next iteration: Random action index:", action)
        return action

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                heightmap_resolution=self.heightmap_resolution,
                num_rotations=self.num_rotations,
                ucb_confidence=self.ucb_confidence,         
            )
        )
        return data

    def _mask(self, obs: th.Tensor, output_prob: th.Tensor):
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
        num_rotations: int,
        ucb_confidence: float,
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

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "heightmap_resolution": self.heightmap_resolution,
            "num_rotations": self.num_rotations,
            "ucb_confidence": self.ucb_confidence,
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
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

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
                num_rotations=self.num_rotations,
                ucb_confidence=self.ucb_confidence,  
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
            )
        )
        return data

register_policy("PGDQNPolicy", PGDQNPolicy)