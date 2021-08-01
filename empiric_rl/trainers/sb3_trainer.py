from typing import Any, Dict, Optional, Union, List, Tuple, Type
from abc import abstractmethod
from dataclasses import dataclass
import argparse
import optuna

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy


from empiric_rl.utils import (realize_hyperparameter,
                              apply_wrappers,
                              make_run_dir)
from empiric_rl.trainers.base_trainer import BaseConfig, BaseExperiment
from empiric_rl.modules.torch.sb3_module import BaseSB3Modules


@dataclass
class SB3Config(BaseConfig):
    policy: Union[str, Type[ActorCriticPolicy], Type[BaseSB3Modules]]
    sb3_wrappers: List[VecEnvWrapper]


class SB3Experiment(BaseExperiment):
    # TODO: Seed

    def __init__(self,
                 configs: SB3Config,
                 cl_args: Dict[str, Any],
                 algo_name: Optional[str] = ""):
        exp_name_prefix = "_".join(["SB3", algo_name])
        super().__init__(configs, cl_args, exp_name_prefix)

    def setup(self, trial: Optional[optuna.Trial] = None):
        hyperparameters = realize_hyperparameter(self.config.hyperparameters, trial=trial)
        vecenv = make_vec_env(
            lambda: apply_wrappers(self.make_env(), self.config.gym_wrappers),
            n_envs=hyperparameters["n_envs"],
            seed=self.cl_args["seed"],
            vec_env_cls=SubprocVecEnv)
        vecenv = apply_wrappers(vecenv, self.config.sb3_wrappers)

        log_dir = make_run_dir(self.main_dir, self.exp_name)
        logger = configure(log_dir, ["stdout", "json", "tensorboard", "csv"])

        agent, score = self._setup(hyperparameters, logger, vecenv, log_dir)

        if self.cl_args["save_model"]:
            agent.save(log_dir)

        return score

    @abstractmethod
    def _setup(self,
               hyperparameters: Dict[Union[Dict[str, Any]], Any],
               logger: Logger,
               vecenv: VecEnv,
               log_dir: str
               ) -> Tuple[BaseAlgorithm, float]:
        pass

    @staticmethod
    def add_parse_arguments(parser: argparse.ArgumentParser) -> None:
        BaseExperiment.add_parse_arguments(parser)
        parser.add_argument("--seed", type=int, default=None,
                            help="Global seed")
        parser.add_argument("--device", type=str, default="cpu",
                            help="Torch device")

