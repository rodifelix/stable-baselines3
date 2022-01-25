from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F
import os

from stable_baselines3.common import logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn
from stable_baselines3.common.utils import get_linear_fn, polyak_update
from pg_baseline.pg_policies import PGDQNPolicy, NoObjectsInSceneException
from pg_baseline.pg_buffer import PGBuffer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback

import time

from cw2 import cw_error

class PGDQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[PGDQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable] = 1e-4,
        buffer_size: int = 5000,
        learning_starts: int = 1000,
        batch_size: Optional[int] = 50,
        tau: float = 1.0,
        start_gamma : float = 0,
        final_gamma: float = 0.8,
        gamma_fraction: float = 0.75,
        train_freq: int = 4,
        gradient_steps: int = 1,
        n_episodes_rollout: int = -1,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 250,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        testing_mode : bool = False,
        update_mask: bool = True,
        use_target: bool = True,
        use_double_q: bool = True,
        net_class: str = "HG_Mask",
        loss_function: F = F.mse_loss,
        n_step = 1,
    ):
        if optimize_memory_usage:
            raise NotImplementedError("Optimize memory usage not supported")

        self.start_gamma = start_gamma
        self.final_gamma = final_gamma
        self.gamma_fraction = gamma_fraction

        if self.gamma_fraction == 0:
            self.start_gamma = self.final_gamma

        super(PGDQN, self).__init__(
            policy,
            env,
            PGDQNPolicy,
            learning_rate,
            buffer_size,
            0, #learning_starts
            batch_size,
            tau,
            self.start_gamma,
            train_freq,
            gradient_steps,
            n_episodes_rollout,
            action_noise=None,  # No action noise
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = exploration_initial_eps
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None
        self.n_step = n_step
        
        self.use_target = use_target
        self.use_double_q = use_target and use_double_q #no target implies no double-q

        self.net_class = net_class

        self.loss_function = loss_function

        self.trainings_starts = learning_starts

        self.testing_mode = testing_mode

        self.update_mask = update_mask and self.net_class == "HG_Mask"

        if _init_setup_model:
            self._setup_model()

        self.sample_time = 0
        self.backward_time = 0

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        if not self.testing_mode:
            if self.replay_buffer is None:
                self.replay_buffer = PGBuffer(
                    buffer_size = self.buffer_size,
                    observation_space = self.observation_space,
                    action_space = self.action_space,
                    device = self.device,
                    gamma=self.gamma,
                    optimize_memory_usage=self.optimize_memory_usage,
                    save_future_rewards=not self.use_target,
                    n_step=self.n_step
                )
            else:
                self.replay_buffer.device = self.device

        self.policy_kwargs["use_target"] = self.use_target
        self.policy_kwargs["net_class"] = self.net_class
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        self.policy.reset_noise()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )
        self.gamma_schedule = get_linear_fn(
            start=self.start_gamma, end=self.final_gamma, end_fraction=self.gamma_fraction
        )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target
        self.heightmap_resolution = self.q_net.heightmap_resolution

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollout()`` after each step in the environment.
        """
        if self.num_timesteps % self.target_update_interval == 0 and self.num_timesteps > self.trainings_starts:
            if self.use_target:
                polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            self.gamma = self.gamma_schedule(self._current_progress_remaining)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        logger.record("rollout/exploration rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if self.num_timesteps < self.trainings_starts:
            replay_data = self.replay_buffer.sample(1, env=self._vec_normalize_env)
            with th.no_grad():
                if self.use_target:           
                    target_q = replay_data.rewards
                else:
                    target_q = replay_data.rewards + (1 - replay_data.terminal) * self.gamma * replay_data.future_rewards

                if self.net_class == "VPG":
                    current_q = self.q_net.forward_specific_rotations(replay_data.observations,  th.floor_divide(replay_data.actions.long(), self.heightmap_resolution*self.heightmap_resolution))
                    current_q = th.gather(current_q, dim=1, index=th.remainder(replay_data.actions.long(), self.heightmap_resolution*self.heightmap_resolution))
                else:
                    current_q = self.q_net.forward(replay_data.observations, mask=False)
                    current_q = th.gather(current_q, dim=1, index=replay_data.actions.long())

                new_surprise_values = np.abs(current_q.detach().cpu().numpy() - target_q.detach().cpu().numpy())
                self.replay_buffer.update_sample_surprise_values(new_surprise_values)
            return

        self._update_learning_rate([self.policy.optimizer])

        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            start_sample_time = time.time()
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            end_sample_time = time.time()
            self.sample_time += end_sample_time-start_sample_time

            future_rewards = None
            if not self.use_target:
                future_rewards = replay_data.future_rewards

            start_backward_time = time.time()
            target_q, current_q = self.backward_step(observations=replay_data.observations,
                                                    actions=replay_data.actions,
                                                    next_observations=replay_data.next_observations,
                                                    rewards=replay_data.rewards,
                                                    change=replay_data.change,
                                                    terminal=replay_data.terminal,
                                                    n_length=replay_data.n_length,
                                                    future_rewards=future_rewards,
                                                    iterations=replay_data.iterations,
                                                    losses=losses)
            end_backward_time = time.time()  
            self.backward_time += end_backward_time-start_backward_time 

            np.add.at(self.train_counter, replay_data.iterations.cpu(), 1)
            new_surprise_values = np.abs(current_q.detach().cpu().numpy() - target_q.detach().cpu().numpy())
            self.replay_buffer.update_sample_surprise_values(new_surprise_values)

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))

    def backward_step(self, observations, actions, next_observations, rewards, change, terminal, n_length, future_rewards, iterations=None, losses=None):
        with th.no_grad():
            if self.gamma > 0:
                if self.use_target:
                    if self.use_double_q:
                        _, next_max_idx = self.q_net.forward(next_observations).max(dim=1)
                        next_max_idx = next_max_idx.reshape(-1, 1)
                            # Evaluate action selected by q-network with target network
                        if self.net_class == "VPG":
                            target_q = self.q_net_target.forward_specific_rotations(next_observations, th.floor_divide(next_max_idx.long(), self.heightmap_resolution*self.heightmap_resolution))
                            target_q = target_q.gather(dim=1, index=th.remainder(next_max_idx.long(), self.heightmap_resolution*self.heightmap_resolution))
                        else:
                            target_output = self.q_net_target.forward(next_observations, mask=False)
                            target_q = target_output.gather(dim=1, index=next_max_idx.long())
                    else:
                            # Evaluate action selected by target network with target network
                        target_output = self.q_net_target.forward(next_observations, mask=True)
                        target_q, _ = target_output.max(dim=1)
                else:
                        # Get future reward from replay buffer
                    target_q = future_rewards

                    # Avoid potential broadcast issue
                target_q = target_q.reshape(-1, 1)
                    # 1-step TD target
                target_q = rewards + (1 - terminal) * (self.gamma ^ n_length) * target_q
            else:
                target_q = rewards

        if self.net_class == "VPG":
                # forward type, batch_size images, each with one specific rotation 
            current_q = self.q_net.forward_specific_rotations(observations,  th.floor_divide(actions.long(), self.heightmap_resolution*self.heightmap_resolution))
            current_q = th.gather(current_q, dim=1, index=th.remainder(actions.long(), self.heightmap_resolution*self.heightmap_resolution))
        else:
            current_q = self.q_net.forward(observations, mask=False)
            current_q = th.gather(current_q, dim=1, index=actions.long())

        loss = self.loss_function(current_q, target_q)
        if losses is not None:
            losses.append(loss.item())

            # Optimize the policy
        self.policy.optimizer.zero_grad()
        loss.backward()
            # Clip gradient norm
        if self.net_class == "HG_Mask":
            th.nn.utils.clip_grad_norm_(self.policy.q_net.net.parameters(), self.max_grad_norm)
        else:
            th.nn.utils.clip_grad_norm_(self.policy.q_net.parameters(), self.max_grad_norm)

        self.policy.optimizer.step()

        self.policy.reset_noise()

        if self.update_mask:
            mask = self.q_net.mask(observations)
                
            predictions = th.gather(mask, dim=1, index=actions.long())

            labels = change
            th.set_printoptions(profile="full")
            print("Mask Predictions: ", predictions, "\n Mask Labels: ", labels)
            th.set_printoptions(profile="default")

            if th.isnan(predictions).any():
                raise cw_error.ExperimentSurrender()

            mask_loss = F.binary_cross_entropy(predictions, labels)

            if iterations is not None:
                max_loss_index = th.gather(iterations, dim=0, index=th.argmax(th.abs(th.sub(predictions, labels)), dim=0, keepdim=True)).item()
                print("Max Mask Loss Iteration Index: ", max_loss_index)
                if mask_loss > 5 and max_loss_index == self.num_timesteps-1:
                    print("Save iteration image", self.num_timesteps-1)
                    self.env.envs[0].save_last_transition(os.path.join(self.tensorboard_log, '..'))

            print("Mask loss:", mask_loss)

            self.policy.mask_optimizer.zero_grad()
            mask_loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.q_net.mask_net.parameters(), self.max_grad_norm)
            self.policy.mask_optimizer.step()

        return target_q,current_q

    def testing_backward(self, observation, action, next_observation, reward, change, terminal):
        future_reward = None
        with th.no_grad():
            if self.gamma > 0:
                if not self.use_target:
                    future_reward = self.q_net.forward(next_observation.to(self.device)).max(dim=1)[0].reshape(-1, 1)                    
        
        self.backward_step(observations=observation.to(self.device),
                            actions=action.to(self.device),
                            next_observations=next_observation.to(self.device),
                            rewards=th.tensor(reward).reshape(-1, 1).to(self.device),
                            change = th.tensor(change, dtype=th.float).reshape(-1, 1).to(self.device),
                            terminal=th.tensor(terminal, dtype=th.float).reshape(-1, 1).to(self.device),
                            future_rewards=future_reward)
                

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        explore = np.random.rand() < self.exploration_rate or (self.num_timesteps < self.trainings_starts)
        try:
            action, state = self.policy.predict(observation, state, mask, not explore)
        except NoObjectsInSceneException:
            self.env.envs[0].save_current_state_as_image(os.path.join(self.tensorboard_log, '..'))
            raise cw_error.ExperimentSurrender
        return action, state

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PGDQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        if reset_num_timesteps or not hasattr(self, 'train_counter'):
            self.train_counter = np.zeros(total_timesteps, dtype=np.uint16)
        else:
            self.train_counter = np.append(self.train_counter, np.zeros(total_timesteps, dtype=np.uint16), axis=0)

        super(PGDQN, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

        np.savetxt(os.path.join(self.tensorboard_log, '..', "training_log.txt"), self.train_counter, fmt='%i')
        print("Total runtime sample", self.sample_time)
        print("Total runtime backward", self.backward_time)
        return self

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        n_episodes: int = 1,
        n_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        replay_buffer: Optional[ReplayBuffer] = None,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ReplayBuffer.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_episodes: Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and total_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)

                self.num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer
                if replay_buffer is not None:
                    # Store only the unnormalized version
                    if self._vec_normalize_env is not None:
                        new_obs_ = self._vec_normalize_env.get_original_obs()
                        reward_ = self._vec_normalize_env.get_original_reward()
                    else:
                        # Avoid changing the original ones
                        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

                    next_obs = infos[0]["terminal_observation"].detach().cpu().numpy() if done else new_obs_

                    future_reward = None
                    if not self.use_target:
                        with th.no_grad():
                            future_reward = self.q_net.forward(th.tensor(next_obs, device=self.device)).max(dim=1)[0].detach().cpu().numpy()

                    replay_buffer.add(self._last_original_obs, next_obs, buffer_action, reward_, infos[0]["change"], done, infos[0]["terminal_state"], future_reward)

                self._last_obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obs_

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if 0 < n_steps <= total_steps:
                    break

            if done:
                total_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)

    def _excluded_save_params(self) -> List[str]:
        return super(PGDQN, self)._excluded_save_params() + ["q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        
        if self.net_class == "HG_Mask":
            state_dicts += ["policy.mask_optimizer"]

        return state_dicts, []

