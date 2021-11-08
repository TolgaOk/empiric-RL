from typing import List, Dict, Any, Optional, Union, Tuple
from abc import abstractmethod
from dataclasses import dataclass
import argparse

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from modular_baselines.policies.policy import BasePolicy
from stable_baselines3.common.logger import Logger
from modular_baselines.algorithms.algorithm import BaseAlgorithm
from empiric_rl.trainers.base_trainer import BaseExperiment, BaseConfig


@dataclass
class MBConfig(BaseConfig):
    policy: BasePolicy
    sb3_wrappers: List[Dict[str, Union[VecEnvWrapper, Dict[str, Any]]]]


class MBExperiment(BaseExperiment):

    def __init__(self,
                 configs: MBConfig,
                 cl_args: Dict[str, Any],
                 algo_name: Optional[str] = ""):
        exp_name_prefix = "_".join(["MB", algo_name])
        super().__init__(configs, cl_args, exp_name_prefix=exp_name_prefix)

    @abstractmethod
    def _setup(self, hyperparameters: Dict[Dict[str, Any], Any],
               vecenv: VecEnv,
               logger: Logger,
               seed: int,
               eval_vecenv: VecEnv) -> Tuple[BaseAlgorithm, float]:
        pass

    @staticmethod
    def add_parse_arguments(parser: argparse.ArgumentParser) -> None:
        BaseExperiment.add_parse_arguments(parser)
        parser.add_argument("--seed", type=int, default=None,
                            help="Global seed")
        parser.add_argument("--device", type=str, default="cpu",
                            help="Torch device")
