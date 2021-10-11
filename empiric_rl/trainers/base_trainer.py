from abc import ABC, abstractmethod
from typing import List, Any, Dict, Union, Optional, Union
from dataclasses import dataclass
import os
import argparse
import numpy as np
import optuna
import gym
import json

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from empiric_rl.utils import (HyperParameter,
                              realize_hyperparameter,
                              apply_wrappers,
                              make_run_dir)


@dataclass
class TunerInfo:
    sampler_cls: optuna.samplers.BaseSampler
    n_startup_trials: int
    n_trials: int
    n_procs: int
    direction: str


@dataclass
class BaseConfig:
    policy: Any
    hyperparameters: Dict[str, Union[Dict[str, HyperParameter], HyperParameter]]
    gym_wrappers: List[Dict[str, Union[gym.Wrapper, Dict[str, Any]]]]
    tuner: Optional[TunerInfo]


class BaseaConfigEncoder():
    @staticmethod
    def encode(config: BaseConfig,
               realized_hyperparameters: Dict[str, Any],
               seed: int):
        return dict(
            policy=config.policy.__class__.__name__,
            hyperparameters=realized_hyperparameters,
            seed=seed,
            gym_wrappers=[dict(wrapper=info["class"].__name__,
                               kwargs=info["kwargs"])
                          for info in config.gym_wrappers],
            tuner=dict(
                sampler_cls=config.tuner.sampler_cls.__name__,
                n_startup_trials=config.tuner.n_startup_trials,
                n_trials=config.tuner.n_trials,
                n_procs=config.tuner.n_procs,
                direction=config.tuner.direction)
        )


class BaseExperiment(ABC):

    def __init__(self,
                 configs: BaseConfig,
                 cl_args: Dict[str, Any],
                 exp_name_prefix: Optional[str] = ""):
        self.cl_args = cl_args
        env_class_name = self.make_env().env.__class__.__name__
        self.config = configs[env_class_name]
        self.exp_name = "_".join([exp_name_prefix, env_class_name])
        self.main_dir = cl_args["log_dir"]
        if self.cl_args["tune"]:
            self.main_dir = make_run_dir(self.main_dir, "Tune_"+self.exp_name)

    @property
    def config_encoder_class(self):
        return BaseaConfigEncoder

    @abstractmethod
    def setup(self, trial: Optional[optuna.Trial] = None):
        pass

    def run(self) -> Union[None, float]:
        if self.cl_args["tune"]:
            return self.tune()
        return self.setup()

    @staticmethod
    def make_seed(seed, trial):
        seed = seed if seed is not None else np.random.randint(0, 2**20)
        if trial is not None:
            seed = seed + 1000 * trial.number
        return seed

    def make_env(self):
        return gym.make(self.cl_args["env_name"])

    def setup(self, trial: Optional[optuna.Trial] = None) -> float:
        hyperparameters = realize_hyperparameter(self.config.hyperparameters, trial=trial)
        seed = self.make_seed(self.cl_args["seed"], trial)
        vecenv = make_vec_env(
            lambda: apply_wrappers(self.make_env(), self.config.gym_wrappers),
            n_envs=hyperparameters["n_envs"],
            seed=seed,
            vec_env_cls=SubprocVecEnv)
        vecenv = apply_wrappers(vecenv, self.config.sb3_wrappers)

        log_dir = make_run_dir(self.main_dir, self.exp_name)
        logger = configure(log_dir, ["stdout", "json", "tensorboard", "csv"])

        agent, score = self._setup(hyperparameters, vecenv, logger, seed)

        if self.cl_args["save_model"]:
            agent.save(log_dir)
        with open(os.path.join(log_dir, "meta-data.json"), "w") as file:
            json.dump(dict(commandline_args=self.cl_args,
                           config=self.config_encoder_class.encode(self.config, hyperparameters, seed)), file)

        return score

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
    def add_parse_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--env-name", type=str, required=True,
                            help="Gym environment name")
        parser.add_argument("--tune", action="store_true",
                            help="Use tune_fn to get hyperparameters instead of default")
        parser.add_argument("--save-model", action="store_true",
                            help="Save the model to the log directory")
        parser.add_argument("--log-interval", type=int, default=500,
                            help=("Logging interval in terms of training"
                                  " iterations"))
        parser.add_argument("--log-dir", type=str, default=None,
                            help=("Logging dir"))
