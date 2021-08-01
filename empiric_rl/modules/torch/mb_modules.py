from abc import ABC
from typing import List, Union, Optional, Tuple
from abc import ABC, abstractmethod
from gym.spaces import Discrete, Box
import torch

from modular_baselines.algorithms.a2c.torch_policy import TorchA2CPolicy
from empiric_rl.modules.torch.base_module import BaseDenseActorCritic


class MBDenseActorCritic(BaseDenseActorCritic, TorchA2CPolicy):

    def __init__(self,
                 observation_space: Box,
                 action_space: Union[Box,  Discrete],
                 lr: float,
                 pi_layer_widths: List[int],
                 value_layer_widths: List[int],
                 pi_activation_fn: Optional[torch.nn.Module] = torch.nn.ReLU,
                 value_activation_fn: Optional[torch.nn.Module] = torch.nn.ReLU,
                 device: Optional[str] = "cpu") -> None:

        BaseDenseActorCritic.__init__(self,
                                      observation_space,
                                      action_space,
                                      lr,
                                      pi_layer_widths,
                                      value_layer_widths,
                                      pi_activation_fn,
                                      value_activation_fn)
        self._device = device

    def forward(self, observation: torch.Tensor) -> None:
        raise NotImplementedError

    @property
    def device(self):
        return self._device

    def evaluate_rollout(self,
                         observation: torch.Tensor,
                         policy_state: Union[None, torch.Tensor],
                         action: torch.Tensor,
                         last_next_obseration: torch.Tensor,
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logits = self.pi_network(observation)
        pi_dist = self.get_dist(logits=pi_logits)
        concat_values = self.value_network(
            torch.cat([observation, last_next_obseration.unsqueeze(1)], dim=1)
        )
        values, last_value = concat_values[:, :-1], concat_values[:, -1]
        if len(action.shape) != len(observation.shape):
            action = action.unsqueeze(-1)
        log_probs = pi_dist.log_prob(action).unsqueeze(-1)
        entropies = pi_dist.entropy().unsqueeze(-1)
        return values, log_probs, entropies, last_value

    def init_state(self, batch_size=None):
        # Initialize Policy State. None for non-reccurent models
        return None

    def sample_action(self,
                      observation: torch.Tensor,
                      policy_state: Union[None, torch.Tensor]
                      ) -> torch.Tensor:
        observation = torch.from_numpy(observation).to(self.device)
        pi_logits = self.pi_network(observation)
        pi_dist = self.get_dist(logits=pi_logits)
        action = pi_dist.sample()
        if len(action.shape) != len(observation.shape):
            action = action.unsqueeze(-1)
        return action.cpu().numpy(), None, {}
