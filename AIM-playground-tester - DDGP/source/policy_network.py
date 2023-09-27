import math
from typing import Tuple

import torch
from gymnasium import spaces
from torch import Tensor, nn
from torch.distributions import Distribution, Normal

from action_utils import scale_and_clip_actions, scale_actions

class FeatureExtractor(nn.Module):
    def __init__(self,
                 state_size: int,
                 hidden_size: int = 128,
                 shared_extractor: bool = False) -> None:
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size

        self.shared_extractor = shared_extractor

        if self.shared_extractor:
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.state_size, hidden_size),
                nn.Tanh(),
            )
        else:
            self.feature_extractor_v = nn.Sequential(
                nn.Linear(self.state_size, hidden_size),
                nn.Tanh(),
            )
            self.feature_extractor_pi = nn.Sequential(
                nn.Linear(self.state_size, hidden_size),
                nn.Tanh(),
            )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.shared_extractor:
            features = self.feature_extractor(x)
            return features, features
        else:
            features_v = self.feature_extractor_v(x)
            features_pi = self.feature_extractor_pi(x)
            return features_v, features_pi

    


class ActorCriticPolicy(nn.Module):
    def __init__(self,
                 state_space: spaces.Space,
                 action_space: spaces.Space,
                 hidden_size: int = 128,
                 init_std: float = 1.,
                 shared_extractor: bool = False) -> None:
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.state_size = state_space.shape[-1]
        self.n_actions = action_space.shape[-1]
        self.hidden_size = hidden_size

        self.shared_extractor = shared_extractor

        # Default shared feature extractor.
        self.feature_extractor = FeatureExtractor(
            self.state_size, hidden_size, shared_extractor=self.shared_extractor)
        

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.n_actions),
            nn.Tanh()  # Squash actions to [-1, 1]
        )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Initialize the log standard deviation to -1 (std = 0.36)
        # Use log std to make sure std is always positive (such that it is differentiable)
        self.log_std = nn.Parameter(math.log(init_std)*torch.ones(
            self.n_actions), requires_grad=True)

        self.init_weights()
        self.update_device()

    def init_weights(self) -> None:
        # Initialize the weights of the network.
        # Action head weights are initialized to 0.01 to ensure the initial policy
        # will be close to the zero policy (zero actions).

        module_gains = {
            # self.feature_extractor_v: 1.0,
            # self.feature_extractor_pi: 1.0,
            self.actor_head: 0.01,
            self.critic_head: 1.0
        }

        def _init_weights(m: nn.Module, gain):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain)
                nn.init.zeros_(m.bias)

        for module, gain in module_gains.items():
            module.apply(lambda x: _init_weights(x, gain))

    def forward(self, x: Tensor) -> Tuple[Distribution, Tensor]:
        features_v, features_pi = self.feature_extractor(x)

        value = self.critic_head(features_v)
        mu = self.actor_head(features_pi)
        mu = scale_actions(mu, self.action_space)

        std = self.log_std.exp().expand_as(mu)

        action_distribution = Normal(mu, std)

        return action_distribution, value

    def sample(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        action_distribution, value = self(x)
        action_distribution: Distribution

        # Sample an action vector given the policy distribution conditioned on state.
        action = action_distribution.sample()
        # sc_action = scale_and_clip_actions(action, self.action_space)

        # Note this is the log pdf of the sampled action. (So can be positive)
        log_prob = action_distribution.log_prob(action)
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob, value

    def evaluate(self, x: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        action_distribution, value = self(x)

        # log pi(a_t|s_t) = sum_i log pi(a^(i)_t|s_t)
        log_prob = action_distribution.log_prob(action)
        log_prob = log_prob.sum(dim=-1)

        entropy = action_distribution.entropy()
        return log_prob, value, entropy

    def act(self, x: Tensor) -> Tensor:
        action_distribution, _ = self(x)
        action = action_distribution.sample()
        # return scale_and_clip_actions(action, self.action_space)
        return action

    def predict_value(self, x: Tensor) -> Tensor:
        features, _ = self.feature_extractor(x)
        value = self.critic_head(features)
        return value

    def update_device(self):
        self.device = next(self.parameters()).device
