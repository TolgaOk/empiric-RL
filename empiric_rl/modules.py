from typing import Union, List, Callable, Optional, Tuple
import torch
from gym.spaces import Box, Discrete
from stable_baselines3.common.type_aliases import Schedule


class DenseNet(torch.nn.Module):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_widths: List[int],
                 activation_fn: Optional[torch.nn.Module] = torch.nn.ReLU) -> None:
        super().__init__()
        module_list = []
        for width in hidden_widths:
            module_list.append(torch.nn.Linear(input_size, width))
            module_list.append(activation_fn())
            input_size = width
        module_list.append(torch.nn.Linear(input_size, output_size))
        self.network = torch.nn.Sequential(*module_list)

    def forward(self, input_tensor: torch.Tensor):
        return self.network(input_tensor)


class DenseActorCritic(torch.nn.Module):

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
        super().__init__()

        assert len(observation_space.shape) == 1, (
            "{} expects flatten observations, but given: {}".format(
                __class__.__name__, observation_space.shape))

        self.action_space = action_space
        self.input_size = observation_space.shape[0]
        self.pi_out_size = action_space.n if isinstance(
            action_space, Discrete) else action_space.shape[0] * 2

        self.pi_network = DenseNet(self.input_size, self.pi_out_size,
                                   pi_layer_widths, pi_activation_fn)
        self.value_network = DenseNet(self.input_size, 1, value_layer_widths, value_activation_fn)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schduler(1))

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

    def get_dist(self, logits):
        if isinstance(self.action_space, Discrete):
            return torch.distributions.Categorical(logits=logits)
        if isinstance(self.action_space, Box):
            mean_logits, std_logits = logits.split(logits.shape[-1]//2, dim=-1)
            std = torch.nn.functional.softplus(std_logits)
            return torch.distributions.Independent(
                torch.distributions.Normal(loc=mean_logits, scale=std_logits),
                reinterpreted_batch_ndims=-1)
        else:
            raise NotImplementedError("Unsupported action space: {}".format(self.action_space))
