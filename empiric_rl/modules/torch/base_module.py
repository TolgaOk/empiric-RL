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


class ConvNet(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 activation_fn: torch.nn.Module = torch.nn.ELU,
                 channel_depths: List[int] = [64, 64, 128, 256, 512],
                 kernel_size: Union[int, List[int]] = 5,
                 padding: Union[int, List[int]] = 2,
                 stride: Union[int, List[int]] = 2,
                 maxpool: Optional[int] = None,
                 use_flatten: bool = True,
                 use_end_nonlinearity = True,):
        super().__init__()

        self.channel_depths = channel_depths
        self.kernel_size = self._check_conv_parameters(kernel_size)
        self.stride = self._check_conv_parameters(stride)
        self.padding = self._check_conv_parameters(padding)

        layers = []
        prev_depth = in_channels

        for depth, n_kernel, n_stride, n_pad in zip(self.channel_depths,
                                                    self.kernel_size,
                                                    self.stride,
                                                    self.padding):
            layers += [
                torch.nn.Conv2d(prev_depth, depth,
                                kernel_size=n_kernel, padding=n_pad, stride=n_stride),
                activation_fn(),
            ]
            prev_depth = depth
        if use_end_nonlinearity is False:
            layers.pop()
        if maxpool is not None:
            layers.append(torch.nn.AdaptiveMaxPool2d(maxpool))
        if use_flatten:
            layers.append(torch.nn.Flatten())
        self.net = torch.nn.Sequential(*layers)

        self.apply(self.orthogonal_init)

    def orthogonal_init(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.orthogonal_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, img):
        return self.net(img)

    def pre_process(self, obs_image: torch.Tensor):
        return obs_image.float() / 255

    def _check_conv_parameters(self, parameter):
        if isinstance(parameter, list):
            assert len(parameter) == len(self.channel_depths)
        elif isinstance(parameter, int):
            parameter = [parameter] * len(self.channel_depths)
        else:
            raise ValueError("Convolution parameters must be an integer or list type")
        return parameter


class BaseDenseActorCritic(torch.nn.Module, ABC):

    def __init__(self,
                 in_size: int,
                 action_space: Union[Box,  Discrete],
                 lr: float,
                 pi_layer_widths: List[int],
                 value_layer_widths: List[int],
                 pi_activation_fn: Optional[torch.nn.Module] = torch.nn.ReLU,
                 value_activation_fn: Optional[torch.nn.Module] = torch.nn.ReLU,
                 ) -> None:
        super().__init__()

        self.action_space = action_space
        self.input_size = in_size
        self.lr = lr
        self.pi_out_size = action_space.n if isinstance(
            action_space, Discrete) else action_space.shape[0] * 2

        self.pi_network = DenseNet(self.input_size, self.pi_out_size,
                                   pi_layer_widths, pi_activation_fn)
        self.value_network = DenseNet(self.input_size, 1, value_layer_widths, value_activation_fn)
        self._optimizer = None

        # Last action Layer Initialization
        self.pi_network.network[-1].apply(lambda layer: layer.weight.data.div_(100))

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
