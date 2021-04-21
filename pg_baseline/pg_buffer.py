import warnings
from typing import Dict, Generator, Optional, Union

import numpy as np
import torch as th
from gym import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class PGBuffer(ReplayBuffer):
    """
    Prioritized experience replay buffer

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.surprise = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.save_indices = []

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes + self.surprise.nbytes
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            batch_size = min(upper_bound, batch_size)
            most_recent_element_idx = self.buffer_size - 1 if self.pos == 0 else self.pos-1
            assert batch_size > 0, "Error: Nothing in buffer to sample or batch_size set to 0" 
            if batch_size > 1:                
                sorted_surprise_ind = np.argsort(self.surprise[:upper_bound,0]).astype(int)
                #skip element most recent element at self.pos-1, as this will always be selected
                sorted_surprise_ind = sorted_surprise_ind[sorted_surprise_ind != most_recent_element_idx]
                for i in range(batch_size-1):
                    rand_sample_ind = np.round(np.random.power(2, 1)*(sorted_surprise_ind.size-1)).astype(int)
                    if i == 0:
                        self.save_indices = sorted_surprise_ind[rand_sample_ind]
                    else:
                        self.save_indices = np.append(self.save_indices, sorted_surprise_ind[rand_sample_ind])
                    sorted_surprise_ind = np.delete(sorted_surprise_ind, rand_sample_ind)

                # Always sample the most recent entry to generate surprise value
                self.save_indices = np.append(self.save_indices, [most_recent_element_idx])
            else:
                self.save_indices = [most_recent_element_idx]

            return self._get_samples(self.save_indices, env=env)
        else:
            raise NotImplementedError("Optimize memory usage with prio replay not supported yet")
        #TODO: implement for optimize memory usage
        """if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)"""

    def update_sample_surprise_values(self, new_values: np.ndarray):
        assert len(new_values) == len(self.save_indices), "Amount of saved indices and provided amount of new values not the same"
        self.surprise[self.save_indices] = new_values.copy()
        self.save_indices = []



