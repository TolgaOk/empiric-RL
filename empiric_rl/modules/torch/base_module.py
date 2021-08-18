from abc import abstractmethod, ABC
from typing import Union, List, Optional
import torch
from gym.spaces import Box, Discrete


class DenseNet(torch.nn.Module):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_widths: List[int],
                 activation_fn: Optional[torch.nn.Module] = torch.nn.ReLU) -> None:
        """ Fully Connected Network

        Args:
            input_size (int): Size of the input features
            output_size (int): Size of the output features
            hidden_widths (List[int]): List of hidden neurons
            activation_fn (Optional[torch.nn.Module], optional): Activation module. Defaults to 
                torch.nn.ReLU.
        """
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


class BaseDenseActorCritic(torch.nn.Module, ABC):

    def __init__(self,
                 observation_space: Box,
                 action_space: Union[Box,  Discrete],
                 lr: float,
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
        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Last action Layer Initialization
        self.pi_network.network[-1].apply(lambda layer: layer.weight.data.div_(100))

    @property
    def optimizer(self):
        return self._optimizer

    @abstractmethod
    def forward(self, observation: torch.Tensor):
        pass

    def get_dist(self, logits):
        if isinstance(self.action_space, Discrete):
            return torch.distributions.Categorical(logits=logits)
        if isinstance(self.action_space, Box):
            mean_logits, std_logits = logits.split(logits.shape[-1]//2, dim=-1)
            std = torch.nn.functional.softplus(std_logits) + 0.1
            return torch.distributions.Independent(
                torch.distributions.Normal(loc=mean_logits, scale=std),
                reinterpreted_batch_ndims=1)
        else:
            raise NotImplementedError("Unsupported action space: {}".format(self.action_space))
