import torch
from gymnasium import spaces



def scale_and_clip_actions(action: torch.Tensor, action_space: spaces.Box) -> torch.Tensor:
    """
    Scale actions from [-1, 1] to the action space's bounds, then clip them.
    """
    low = torch.tensor(action_space.low[-1], dtype=torch.float32).to(action.device)
    high = torch.tensor(action_space.high[-1], dtype=torch.float32).to(action.device)

    scaled_action = 0.5 * (action + 1) * (high - low) + low
    clipped_action = torch.clamp(scaled_action, low, high)

    return clipped_action

def scale_actions(action: torch.Tensor, action_space: spaces.Box) -> torch.Tensor:
    """
    Scale actions from [-1, 1] to the action space's bounds.
    """
    low = torch.tensor(action_space.low[-1], dtype=torch.float32).to(action.device)
    high = torch.tensor(action_space.high[-1], dtype=torch.float32).to(action.device)

    scaled_action = 0.5 * (action + 1) * (high - low) + low
    return scaled_action