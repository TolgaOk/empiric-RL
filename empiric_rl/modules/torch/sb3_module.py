from typing import List, Union, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
import warnings
import numpy as np
from gym.spaces import Discrete, Box
import torch

from stable_baselines3.common.type_aliases import Schedule

from empiric_rl.modules.torch.base_module import BaseDenseActorCritic, ConvNet


class BaseSB3Modules(ABC):

    @abstractmethod
    def forward(self, observation: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def evaluate_actions(self,
                         observation: torch.Tensor,
                         action: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class SB3DenseActorCritic(BaseDenseActorCritic, BaseSB3Modules):

    def __init__(self,
                 observation_space: Box,
                 action_space: Union[Box,  Discrete],
                 lr_schduler: Schedule,
                 use_sde: bool,
                 pi_layer_widths: List[int],
                 value_layer_widths: List[int],
                 pi_activation_fn: Optional[torch.nn.Module] = torch.nn.ReLU,
                 value_activation_fn: Optional[torch.nn.Module] = torch.nn.ReLU,
                 ) -> None:
        assert len(observation_space.shape) == 1, "Dense AC only accepts flattened obs space"
        BaseDenseActorCritic.__init__(self,
                                      in_size=observation_space.shape[0],
                                      action_space=action_space,
                                      lr=lr_schduler(1),
                                      pi_layer_widths=pi_layer_widths,
                                      value_layer_widths=value_layer_widths,
                                      pi_activation_fn=pi_activation_fn,
                                      value_activation_fn=value_activation_fn)

    def forward(self, observation: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logits = self.pi_network(observation)
        pi_dist = self.get_dist(logits=pi_logits)
        value = self.value_network(observation)
        action = pi_dist.sample()
        log_prob = pi_dist.log_prob(action)
        return action, value, log_prob

    def evaluate_actions(self,
                         observation: torch.Tensor,
                         action: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logits = self.pi_network(observation)
        pi_dist = self.get_dist(logits=pi_logits)
        value = self.value_network(observation)
        log_prob = pi_dist.log_prob(action)
        entropy = pi_dist.entropy()
        return value, log_prob, entropy


class SB3ConvActorCritic(SB3DenseActorCritic):

    def __init__(self,
                 observation_space: Box,
                 action_space: Union[Box, Discrete],
                 lr_schduler: Schedule,
                 use_sde: bool,
                 pi_layer_widths: List[int],
                 value_layer_widths: List[int],
                 pi_activation_fn: Optional[torch.nn.Module],
                 value_activation_fn: Optional[torch.nn.Module],
                 conv_net_kwargs: Dict[str, Any],
                 ) -> None:
        if observation_space.shape[-3] > 4:
            warnings.warn("Expected channel axis is [-3]")
        feature_obs_space = Box(low=-np.inf, high=np.inf,
                                shape=(conv_net_kwargs["maxpool"]**2 *
                                       conv_net_kwargs["channel_depths"][-1],))
        super().__init__(feature_obs_space,
                         action_space,
                         lr_schduler,
                         use_sde,
                         pi_layer_widths,
                         value_layer_widths,
                         pi_activation_fn=pi_activation_fn,
                         value_activation_fn=value_activation_fn)
        self.conv_net = ConvNet(observation_space.shape[-3], **conv_net_kwargs)

    def forward(self, observation: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.conv_net(self.conv_net.pre_process(observation))
        return super().forward(features)

    def evaluate_actions(self,
                         observation: torch.Tensor,
                         action: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.conv_net(self.conv_net.pre_process(observation))
        return super().evaluate_actions(features, action)
