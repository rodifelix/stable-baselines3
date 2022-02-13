from collections import deque
import warnings
from typing import Dict, Generator, Optional, Union, NamedTuple

import numpy as np
import torch as th
from gym import spaces
import random

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

class PGBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    change: th.Tensor
    terminal: th.Tensor
    rewards: th.Tensor
    iterations: th.Tensor
    n_length: th.Tensor

class PGBufferSamplesWithFutures(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    change: th.Tensor
    terminal: th.Tensor
    rewards: th.Tensor
    iterations: th.Tensor
    n_length: th.Tensor
    future_rewards: th.Tensor

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
        gamma: float,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        save_future_rewards: bool = False,
        n_step: int = 1,
        prio_exp: bool = True,
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
        self.terminal = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.surprise = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.change = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.n_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self.save_indices = []
        self.iteration = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self.iteration_counter = 0

        self.n_step = n_step
        self.gamma = gamma

        self.prio_exp = prio_exp

        self.n_step_storage = deque()

        self.unsampled_pos_start = 0

        self.save_future_rewards = save_future_rewards
        if self.save_future_rewards:
            self.future_reward = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes + self.surprise.nbytes + self.iteration.nbytes + self.terminal.nbytes + self.change.nbytes + self.n_length.nbytes
            if self.save_future_rewards:
                total_memory_usage += self.future_reward.nbytes

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

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, change: bool, done: np.ndarray, terminal_state: bool, future_reward: np.ndarray = None) -> None:
        if not change:
            # Copy to avoid modification by reference
            self._add(
                obs=np.array(obs).copy(),
                next_obs=np.array(next_obs).copy(),
                action=np.array(action).copy(),
                reward=np.array(reward).copy(),
                change=np.array(change),
                done=np.array(done).copy(),
                terminal_state=np.array(terminal_state),
                n_counter=1,
                iteration=self.iteration_counter,
                future_reward=None if future_reward is None else np.array(future_reward).copy()
                )
        else:
            # Copy to avoid modification by reference
            to_store = {
                "obs" : np.array(obs).copy(),
                "next_obs" : np.array(next_obs).copy(),
                "action" : np.array(action).copy(),
                "reward" : np.array(reward).copy(),
                "change" : np.array(change),
                "done" : np.array(done).copy(),
                "terminal_state" : np.array(terminal_state),
                "future_reward" : None if future_reward is None else np.array(future_reward).copy(),
                "iteration" : self.iteration_counter
            }
            
            self.n_step_storage.append(to_store)

        if len(self.n_step_storage) > 0 and ((change and terminal_state) or done or len(self.n_step_storage) >= self.n_step):
            initial_obs = self.n_step_storage[0]["obs"]
            last_next_obs = self.n_step_storage[-1]["next_obs"]
            initial_action = self.n_step_storage[0]["action"]

            reward_sum = 0
            for i in range(len(self.n_step_storage)):
                reward_sum += (self.gamma ** i) * self.n_step_storage[i]["reward"]

            initial_change = self.n_step_storage[0]["change"]
            last_terminal = self.n_step_storage[-1]["terminal_state"]
            last_done = self.n_step_storage[-1]["done"]

            last_future_reward = self.n_step_storage[-1]["future_reward"]

            initial_iteration = self.n_step_storage[0]["iteration"]

            self._add(
                obs=initial_obs,
                next_obs=last_next_obs,
                action=initial_action,
                reward=reward_sum,
                change=initial_change,
                done=last_done,
                terminal_state=last_terminal,
                n_counter=len(self.n_step_storage),
                iteration=initial_iteration,
                future_reward=last_future_reward
                )

            self.n_step_storage.clear()

        self.iteration_counter += 1


    def _add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, change: np.ndarray, done: np.ndarray, terminal_state: np.ndarray, n_counter: int, iteration: int, future_reward: np.ndarray = None) -> None:
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.terminal[self.pos] = terminal_state
        self.change[self.pos] = change
        self.iteration[self.pos] = [iteration]
        self.n_length[self.pos] = [n_counter]

        if self.save_future_rewards and future_reward is not None:
            self.future_reward[self.pos] = future_reward

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0



    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> PGBufferSamples:
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
        if self.optimize_memory_usage:
            raise NotImplementedError("Optimize memory usage not supported")
        
        upper_bound = self.buffer_size if self.full else self.pos
        batch_size = min(upper_bound, batch_size)
        assert batch_size > 0, "Error: Nothing in buffer to sample or batch_size set to 0" 
                
        if self.prio_exp:
            self.save_indices = self.get_unsampled_indices()
            if len(self.save_indices) < batch_size:                
                sorted_surprise_ind = np.argsort(self.surprise[:upper_bound,0]).astype(int)
                sorted_surprise_ind = np.array([index for index in sorted_surprise_ind if index not in self.save_indices], dtype=np.int)
                while len(self.save_indices) < batch_size and len(sorted_surprise_ind) > 0:
                    rand_sample_ind = np.round(np.random.power(2, 1)*(sorted_surprise_ind.size-1)).astype(int)
                    self.save_indices = np.append(self.save_indices, sorted_surprise_ind[rand_sample_ind])
                    sorted_surprise_ind = np.delete(sorted_surprise_ind, rand_sample_ind)
            else:
                self.save_indices = random.sample(range(upper_bound), k=batch_size)

        return self._get_samples(self.save_indices, env=env)
        
    def sample_new_transitions(self, env: Optional[VecNormalize] = None) -> PGBufferSamples:
        self.save_indices = self.get_unsampled_indices()

        if len(self.save_indices) > 1:
            return self._get_samples(self.save_indices, env=env)
        else:
            return None

    def get_unsampled_indices(self):       
        if self.unsampled_pos_start <= self.pos:
            indices = [*range(self.unsampled_pos_start, self.pos)]
        else:
            indices = [*range(self.unsampled_pos_start, self.buffer_size), *range(0, self.pos)]

        self.unsampled_pos_start = self.pos
        return np.array(indices, dtype=np.int)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> PGBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        if self.save_future_rewards:
            data = (
                self._normalize_obs(self.observations[batch_inds, 0, :], env),
                self.actions[batch_inds, 0, :],
                next_obs,
                self.dones[batch_inds],
                self.change[batch_inds],
                self.terminal[batch_inds],
                self._normalize_reward(self.rewards[batch_inds], env),
                self.iteration[batch_inds],
                self.n_length[batch_inds],
                self.future_reward[batch_inds]
            )
            return PGBufferSamplesWithFutures(*tuple(map(self.to_torch, data)))
        else:
            data = (
                self._normalize_obs(self.observations[batch_inds, 0, :], env),
                self.actions[batch_inds, 0, :],
                next_obs,
                self.dones[batch_inds],
                self.change[batch_inds],
                self.terminal[batch_inds],
                self._normalize_reward(self.rewards[batch_inds], env),
                self.iteration[batch_inds],
                self.n_length[batch_inds]
            )
            return PGBufferSamples(*tuple(map(self.to_torch, data)))

    def update_sample_surprise_values(self, new_values: np.ndarray):
        if self.prio_exp:
            assert len(new_values) == len(self.save_indices), "Amount of saved indices and provided amount of new values not the same"
            self.surprise[self.save_indices] = new_values.copy()
        
        self.save_indices = []

def merge_buffers(*buffers, 
    buffer_size: int,
    observation_space: spaces.Space,
    action_space: spaces.Space,
    device: Union[th.device, str] = "cpu",
    n_envs: int = 1,
    optimize_memory_usage: bool = False,):
    
    merged_buffer = PGBuffer(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage)
    
    samples_count = sum(buffer.buffer_size if buffer.full else buffer.pos for buffer in buffers)

    pos = 0
    if samples_count <= buffer_size:
        for buffer in buffers:
            if buffer.full:
                count = buffer.buffer_size
            else:
                count = buffer.pos
        
            merged_buffer.observations[pos:count] = buffer.observations[:count]
            merged_buffer.next_observations[pos:count] = buffer.next_observations[:count]
            merged_buffer.actions[pos:count] = buffer.actions[:count]
            merged_buffer.rewards[pos:count] = buffer.rewards[:count]
            merged_buffer.dones[pos:count] = buffer.dones[:count]
            merged_buffer.terminal[pos:count] = buffer.terminal[:count]
            merged_buffer.surprise[pos:count] = buffer.surprise[:count]            
            pos +=count
        
        merged_buffer.iteration[:pos] = np.reshape(np.arange(pos, dtype=np.int32), (pos, 1))
        merged_buffer.iteration_offset = 0
        merged_buffer.pos = pos
    else:
        raise NotImplementedError("Can only merge if resulting buffer size is less or equal specified buffer size argument")
    



