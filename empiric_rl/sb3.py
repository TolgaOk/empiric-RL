from typing import Any, Dict, NamedTuple, Optional, Union, List, Tuple
from abc import ABC, abstractmethod
from multiprocessing import Process
import gym
import argparse
import os
import optuna
import torch

from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.base_class import BaseAlgorithm


from empiric_rl.common import (realize_hyperparameter,
                               apply_wrappers,
                               make_run_dir,
                               HyperParameter,)


class TunerInfo(NamedTuple):
    sampler_cls: optuna.samplers.BaseSampler
    n_startup_trials: int
    n_trials: int
    n_procs: int
    direction: str


class Config(NamedTuple):
    policy: torch.nn.Module
    hyperparameters: Dict[str, Union[Dict[str, HyperParameter], HyperParameter]]
    gym_wrappers: List[gym.Wrapper]
    sb3_wrappers: List[VecEnvWrapper]
    tuner: Optional[TunerInfo]


class SB3Experiment(ABC):
    # TODO: Seed

    def __init__(self,
                 configs: Config,
                 cl_args: Dict[str, Any],
                 algo_name: Optional[str] = ""):
        self.cl_args = cl_args
        env_class_name = self.make_env().env.__class__.__name__
        self.config = configs[env_class_name]
        self.exp_name = "_".join(["SB3", algo_name, env_class_name])
        self.main_dir = cl_args["log_dir"]
        if self.cl_args["tune"]:
            self.main_dir = make_run_dir(self.main_dir, "Tune_"+self.exp_name)


    def setup(self, trial: Optional[optuna.Trial] = None):
        hyperparameters = realize_hyperparameter(self.config.hyperparameters, trial=trial)
        vecenv = make_vec_env(
            lambda: apply_wrappers(self.make_env(), self.config.gym_wrappers),
            n_envs=self.cl_args["n_envs"],
            seed=self.cl_args["seed"],
            vec_env_cls=SubprocVecEnv)
        vecenv = apply_wrappers(vecenv, self.config.sb3_wrappers)

        log_dir = make_run_dir(self.main_dir, self.exp_name)
        logger = configure(log_dir, ["stdout", "json", "tensorboard", "csv"])

        agent, score = self._setup(hyperparameters, logger, vecenv, log_dir)

        if self.cl_args["save_model"]:
            agent.save(log_dir)

        return score

    def run(self) -> Union[None, float]:
        if self.cl_args["tune"]:
            return self.tune()
        return self.setup()

    def make_env(self):
        return gym.make(self.cl_args["env_name"])

    @abstractmethod
    def _setup(self,
               hyperparameters: Dict[Union[Dict[str, Any]], Any],
               logger: Logger,
               vecenv: VecEnv,
               log_dir: str
               ) -> Tuple[BaseAlgorithm, float]:
        pass

    def tune(self) -> None:
        storage_url = "".join(("sqlite:///", os.path.join(self.main_dir, "store.db")))
        study_name = self.exp_name
        sampler = self.config.tuner.sampler_cls(
            n_startup_trials=self.config.tuner.n_startup_trials)
        study = optuna.create_study(
            storage=storage_url,
            sampler=sampler,
            study_name=study_name,
            direction=self.config.tuner.direction,
            load_if_exists=True)
        study.optimize(
            self.setup,
            n_trials=self.config.tuner.n_trials,
            n_jobs=self.config.tuner.n_procs)

    @staticmethod
    def common_parse_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--env-name", type=str, required=True,
                            help="Gym environment name")
        parser.add_argument("--seed", type=int, default=None,
                            help="Global seed")
        parser.add_argument("--n-envs", type=int, default=8,
                            help="Number of parallel environments")
        parser.add_argument("--device", type=str, default="cpu",
                            help="Torch device")

        parser.add_argument("--tune", action="store_true",
                            help="Use tune_fn to get hyperparameters instead of default")
        parser.add_argument("--save-model", action="store_true",
                            help="Save the model to the log directory")
        parser.add_argument("--log-interval", type=int, default=500,
                            help=("Logging interval in terms of training"
                                  " iterations"))
        parser.add_argument("--log-dir", type=str, default=None,
                            help=("Logging dir"))
