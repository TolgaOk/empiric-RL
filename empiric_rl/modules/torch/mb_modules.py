from typing import List, Union, Optional, Tuple, Dict, Any
import os
import numpy as np
from gym.spaces import Discrete, Box
import torch
import warnings

from modular_baselines.algorithms.a2c.torch_policy import TorchA2CPolicy
from empiric_rl.modules.torch.base_module import BaseDenseActorCritic, ConvNet


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
        assert len(observation_space.shape) == 1, "Dense AC only accepts flattened obs space"
        BaseDenseActorCritic.__init__(self,
                                      observation_space.shape[0],
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
        if isinstance(self.action_space, Discrete):
            action = action.squeeze(-1)
        log_probs = pi_dist.log_prob(action).unsqueeze(-1)
        entropies = pi_dist.entropy().unsqueeze(-1)
        return values, log_probs, entropies, last_value

    def init_state(self, batch_size=None):
        # Initialize Policy State. None for non-reccurent models
        return None

    def sample_action(self,
                      observation: Union[np.ndarray, torch.Tensor],
                      policy_state: Union[None, torch.Tensor]
                      ) -> torch.Tensor:
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).to(self.device)
        pi_logits = self.pi_network(observation)
        pi_dist = self.get_dist(logits=pi_logits)
        action = pi_dist.sample()
        if isinstance(self.action_space, Discrete):
            action = action.unsqueeze(-1)
        return action.cpu().numpy(), None, {}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            dict(
                modules=self.state_dict(),
                optim=self.optimizer.state_dict()
            ),
            path)


class MBConvActorCritic(MBDenseActorCritic):

    def __init__(self,
                 observation_space: Box,
                 action_space: Union[Box, Discrete],
                 lr: float,
                 pi_layer_widths: List[int],
                 value_layer_widths: List[int],
                 pi_activation_fn: Optional[torch.nn.Module],
                 value_activation_fn: Optional[torch.nn.Module],
                 conv_net_kwargs: Dict[str, Any],
                 device: Optional[str] = "cpu") -> None:
        if observation_space.shape[-3] > 4:
            warnings.warn("Expected channel axis is [-3]")
        feature_obs_space = Box(low=-np.inf, high=np.inf,
                                shape=(conv_net_kwargs["maxpool"]**2 *
                                       conv_net_kwargs["channel_depths"][-1],))
        super().__init__(feature_obs_space,
                         action_space,
                         lr,
                         pi_layer_widths,
                         value_layer_widths,
                         pi_activation_fn=pi_activation_fn,
                         value_activation_fn=value_activation_fn,
                         device=device)
        self.conv_net = ConvNet(observation_space.shape[-3], **conv_net_kwargs)

    def sample_action(self,
                      obs_image: np.ndarray,
                      policy_state: Union[None, torch.Tensor]
                      ) -> torch.Tensor:
        obs_image = torch.from_numpy(obs_image).to(self.device)
        flat_features = self.conv_net(self.conv_net.pre_process(obs_image))
        return super().sample_action(flat_features, policy_state)

    def evaluate_rollout(self,
                         obs_image: torch.Tensor,
                         policy_state: Union[None, torch.Tensor],
                         action: torch.Tensor,
                         last_next_obs_img: torch.Tensor,
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        concat_obs = torch.cat([self.conv_net.pre_process(obs_image),
                                self.conv_net.pre_process(last_next_obs_img).unsqueeze(1)], dim=1)
        batch_shape = concat_obs.shape[:2]
        concat_obs = concat_obs.reshape(np.product(batch_shape), *concat_obs.shape[2:])
        concat_features = self.conv_net(concat_obs)
        concat_features = concat_features.reshape(
            *batch_shape, *concat_features.shape[1:])
        features, last_next_feature = concat_features[:, :-1], concat_features[:, -1]

        return super().evaluate_rollout(features,
                                        policy_state,
                                        action,
                                        last_next_feature)
