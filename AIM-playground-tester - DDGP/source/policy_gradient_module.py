import time
from typing import Any, Callable, Dict, NamedTuple, Type

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from policy_network import ActorCriticPolicy
from pytorch_lightning import LightningModule
from rollout import RolloutBufferDataset, RolloutAgent, RolloutSample
from torch import Tensor, nn
from torch.optim import SGD, Adam, Optimizer, RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

OPTIMIZERS: Dict[str, Type[torch.optim.Optimizer]] = {'Adam': Adam,
                                                      'RMSprop': RMSprop,
                                                      'SGD': SGD}

# Namedtuple class for loss policy loss, value loss, entropy loss
class Losses(NamedTuple):
    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy_loss: Tensor


class PolicyGradientModule(LightningModule):
    def __init__(self,
                 env_id: str = 'Ant-v4',
                 algorithm: str = 'A2C',
                 perform_testing: bool = False,
                 steps_per_epoch: int = 20,
                 num_envs: int = 8,
                 num_rollout_steps: int = 5,
                 optimizer: str = 'RMSProp',
                 learning_rate: float = 1e-3,
                 lr_decay: float = 1.,
                 weight_decay: float = 0.,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 gamma: float = 0.99,
                 gae_lambda: float = 1.,
                 init_std: float = 1.,
                 hidden_size: int = 128,
                 shared_extractor: bool = False,
                 ppo_batch_size: int = 64,
                 ppo_epochs: int = 10,
                 ppo_clip_ratio: float = 0.2,
                 ppo_clip_anneal: bool = False,
                 max_epochs: int = 1000,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Does frame inspection to find parameters
        self.save_hyperparameters(ignore=['max_epochs'])

        # Makes envs in init already
        self.env = gym.vector.make(
            env_id, num_envs=num_envs, asynchronous=True)

        self.state_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.policy = ActorCriticPolicy(state_space=self.state_space,
                                        action_space=self.action_space,
                                        hidden_size=hidden_size,
                                        init_std=init_std,
                                        shared_extractor=shared_extractor)

        self.rollout_agent = RolloutAgent(self.env,
                                          self.policy,
                                          num_rollout_steps=num_rollout_steps,
                                          gamma=gamma,
                                          gae_lambda=gae_lambda,
                                          on_rollout_end_cb=self.rollout_cb)

        self.mse_loss = nn.MSELoss()

        algorithm = algorithm.lower()
        if algorithm == 'a2c':
            self.batch_size = num_envs * num_rollout_steps
            self.n_epochs = 1
        elif algorithm == 'ppo':
            self.batch_size = ppo_batch_size
            self.n_epochs = ppo_epochs

        step_fn_dict = {'a2c': self.a2c_step, 'ppo': self.ppo_step}
        self.step = step_fn_dict[algorithm]

        self.total_frames = 0
        self.max_frames = max_epochs * steps_per_epoch * num_envs * num_rollout_steps
        self.rewards = None
        self.test_count = 0

    def ppo_step(self, batch: RolloutSample) -> Losses:
        # PPO
        # Evaluate policy for state action pair pi(a_t|s_t)
        # Retrieves log pi(a_t|s_t), V(s_t), entropy: H(pi)
        log_prob, value, entropy = self.policy.evaluate(
            batch.state, batch.action)
        value = value.flatten()

        # Advantage estimate
        # A(s_t, a_t) = R_t - V(s_t)
        advantage = batch.return_ - batch.value

        # pi_theta / pi_theta_old = exp(log pi_theta - log pi_theta_old)
        prob_ratio = (log_prob - batch.log_prob).exp()

        clip_ratio = self.hparams.ppo_clip_ratio
        if self.hparams.ppo_clip_anneal:
            # Anneal clip ratio from 1 to clip_ratio over training
            clip_ratio = self.hparams.ppo_clip_ratio * \
                (1 - self.total_frames / self.max_frames)
        # Clipped surrogate loss
        conservative_policy_loss = prob_ratio * advantage
        clipped_policy_loss = (torch.clamp(prob_ratio,
                                            1. - clip_ratio,
                                            1. + clip_ratio)
                                * advantage)
        # Minus for gradient ascent
        policy_loss = -torch.min(conservative_policy_loss, clipped_policy_loss).mean()

        # Value loss
        # L_v = E_t [(R_t - V(s_t))^2]
        # Note this loss is simply advantage squared, value requires grad here.
        value_loss = self.mse_loss(batch.return_, value)

        # Entropy loss
        # L_H = E_t [H(pi)]
        entropy_loss = -entropy.mean()

        # Total loss
        # L = L_pi + L_v + L_H
        loss = policy_loss + self.hparams.value_coef * value_loss \
                           + self.hparams.entropy_coef * entropy_loss
        
        return Losses(loss, policy_loss, value_loss, entropy_loss)


    def a2c_step(self, batch: RolloutSample) -> Losses:
        # A2C
        # Evaluate policy for state action pair pi(a_t|s_t)
        # Retrieves log pi(a_t|s_t), V(s_t), entropy: H(pi)
        log_prob, value, entropy = self.policy.evaluate(
            batch.state, batch.action)
        value = value.flatten()

        # Advantage estimate
        # A(s_t, a_t) = R_t - V(s_t)
        # Note, we use batch.value here as this one requires no grad.
        # Alternatively we could detach: advantage.detach()
        advantage = batch.return_ - batch.value

        # Optionally could normalize advantage

        # Policy loss
        # L_pi = E_t [- log pi(a_t|s_t) * A(s_t, a_t)]
        policy_loss = -(log_prob * advantage).mean()

        # Value loss
        # L_v = E_t [(R_t - V(s_t))^2]
        # Note this loss is simply advantage squared, value requires grad here.
        value_loss = self.mse_loss(batch.return_, value)

        # Entropy loss
        # L_H = E_t [H(pi)]
        entropy_loss = -entropy.mean()

        # Total loss
        # L = L_pi + L_v + L_H
        loss = policy_loss + self.hparams.value_coef * value_loss \
                           + self.hparams.entropy_coef * entropy_loss
        
        return Losses(loss, policy_loss, value_loss, entropy_loss)


    def training_step(self, batch: RolloutSample, batch_idx: int) -> Tensor:
        # Run for one rollout sample
        losses = self.step(batch)
        
        # Log metrics
        self.log('policy_loss', losses.policy_loss)
        self.log('value_loss', losses.value_loss)
        self.log('entropy_loss', losses.entropy_loss)
        self.log('loss', losses.loss, prog_bar=True)
        self.log('frame_count', self.total_frames, prog_bar=True)
        self.log('std', self.policy.log_std[0].exp().item(), prog_bar=True)
        self.log('return', batch.return_.mean().item(), prog_bar=True)

        if self.rewards is not None:
            self.log('rewards', self.rewards, prog_bar=True)
            self.rewards = None # To avoid logging same rewards again

        return losses.loss

    def on_train_start(self) -> None:
        self.rollout_agent.prepare()
        # Inform policy of new device
        self.policy.update_device()

    def on_train_epoch_end(self) -> None:
        if self.hparams.perform_testing:
            self.test_count += 1
            if self.test_count % 20 == 0:
                self.policy.train(False)
                self.test_epoch()
                self.policy.train(True)

    def test_epoch(self):
        env = gym.make(self.hparams.env_id, render_mode='human')

        # Run a few episodes
        for episode in range(1):
            done = truncated = False
            total_reward = 0.

            state, _ = env.reset()
            state = torch.as_tensor(
                state, dtype=torch.float32, device=self.device)

            while not done:
                # time.sleep(0.01)
                # Sample action
                action = self.policy.act(state)
                action = action.cpu().numpy()

                # Environment has boxed action space, so clip actions.
                if isinstance(env.action_space, spaces.Box):
                    clipped_action = np.clip(action, env.action_space.low,
                                             env.action_space.high)

                # Perform the action in the environment
                state, reward, done, truncated, info = env.step(clipped_action)
                state = torch.as_tensor(
                    state, dtype=torch.float32, device=self.device)

                # Update the total reward
                total_reward += float(reward)

                if done or truncated:
                    break

            # Print the total reward for the episode
            self.log('test_reward', total_reward, prog_bar=True)
            # print("Episode:", episode + 1, "Total Reward:", total_reward)

        env.close()

    def rollout_cb(self) -> Callable:
        # Customises rollout behaviour by collecting and flushing rollout rewards
        ra = self.rollout_agent
        if len(ra.episode_rewards) > 0:
            self.rewards = np.array(ra.episode_rewards).mean()
            ra.episode_rewards = []
            ra.episode_lengths = []

        self.total_frames += ra.rollout_buffer.out_size
        self.rollout_performed = False

    def train_dataloader(self) -> DataLoader:
        self.dataset = RolloutBufferDataset(
            self.rollout_agent,
            max_steps=self.hparams.steps_per_epoch, # type: ignore
            n_epochs=self.n_epochs)
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def configure_optimizers(self) -> Optimizer:
        optim = OPTIMIZERS[self.hparams.optimizer](self.policy.parameters(),  # type: ignore
                                                  lr=self.hparams.learning_rate,
                                                  weight_decay=self.hparams.weight_decay,
                                                  eps=1e-05)  # type: ignore
        scheduler = ExponentialLR(optim, gamma=self.hparams.lr_decay)  # type: ignore
        return [optim], [scheduler]
