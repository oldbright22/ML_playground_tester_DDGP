from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import wandb
from gymnasium import spaces
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from policy_gradient_module import PolicyGradientModule

VIDEO_PATH = Path("videos")


class VideoLoggerCallback(Callback):
    def __init__(self, save_every_n_epochs=10, max_video_length=200) -> None:
        self.interval = save_every_n_epochs
        self.max_video_length = max_video_length
        self.epoch = 0

        self.env = None

    def _init_env(self, pl_module: PolicyGradientModule) -> None:
        env_id = pl_module.hparams.env_id
        self.video_name_prefix = f"{env_id}"

        env = gym.make(env_id, render_mode="rgb_array")
        # Recorder for every episode
        env = gym.wrappers.RecordVideo(env,
                                       video_folder=str(VIDEO_PATH),
                                       name_prefix=self.video_name_prefix,
                                       video_length=self.max_video_length)

        self.env = env

    def on_train_epoch_start(self, trainer: Trainer, pl_module: PolicyGradientModule) -> None:
        if self.epoch % self.interval != 0:
            self.epoch += 1
            return

        self._init_env(pl_module)

        done = False
        total_reward = 0.

        state, _ = self.env.reset()
        state = torch.as_tensor(
            state, dtype=torch.float32, device=pl_module.device)

        while not done:
            # Sample action
            action = pl_module.policy.act(state)
            action = action.cpu().numpy()

            # Environment has boxed action space, so clip actions.
            if isinstance(self.env.action_space, spaces.Box):
                clipped_action = np.clip(action, self.env.action_space.low,
                                         self.env.action_space.high)

            # Perform the action in the environment
            state, reward, terminated, truncated, _ = self.env.step(
                clipped_action)
            state = torch.as_tensor(
                state, dtype=torch.float32, device=pl_module.device)

            # Update the total reward
            total_reward += float(reward)

            done = terminated or truncated or not self.env.recording 

        self.env.close()

        file_path = f"{str(VIDEO_PATH)}/{self.video_name_prefix}-episode-0.mp4"

        wandb.log({"test/play_video": wandb.Video(file_path, fps=30, format="mp4")})

        self.epoch += 1
