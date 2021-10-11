from typing import Any, Dict, Optional, Union, List, Tuple, Type
from abc import abstractmethod
from dataclasses import dataclass
import argparse

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy

from empiric_rl.trainers.base_trainer import BaseConfig, BaseExperiment
from empiric_rl.modules.torch.sb3_module import BaseSB3Modules


@dataclass
class SB3Config(BaseConfig):
    policy: Union[str, Type[ActorCriticPolicy], Type[BaseSB3Modules]]
    sb3_wrappers: List[Dict[str, Union[VecEnvWrapper, Dict[str, Any]]]]


class SB3Experiment(BaseExperiment):

    def __init__(self,
                 configs: SB3Config,
                 cl_args: Dict[str, Any],
                 algo_name: Optional[str] = ""):
        exp_name_prefix = "_".join(["SB3", algo_name])
        super().__init__(configs, cl_args, exp_name_prefix)

    @abstractmethod
    def _setup(self,
               hyperparameters: Dict[Union[Dict[str, Any]], Any],
               vecenv: VecEnv,
               logger: Logger,
               seed: int,
               ) -> Tuple[BaseAlgorithm, float]:
        pass

    @staticmethod
    def add_parse_arguments(parser: argparse.ArgumentParser) -> None:
        BaseExperiment.add_parse_arguments(parser)
        parser.add_argument("--seed", type=int, default=None,
                            help="Global seed")
        parser.add_argument("--device", type=str, default="cpu",
                            help="Torch device")

