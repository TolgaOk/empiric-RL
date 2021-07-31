from abc import ABC, abstractmethod
from typing import List, Any, Dict, Union, Optional, Union
from dataclasses import dataclass
import os
import argparse
import optuna
import gym

from empiric_rl.utils import HyperParameter, make_run_dir


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
    gym_wrappers: List[gym.Wrapper]
    tuner: Optional[TunerInfo]


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

    @abstractmethod
    def setup(self, trial: Optional[optuna.Trial] = None):
        pass

    def run(self) -> Union[None, float]:
        if self.cl_args["tune"]:
            return self.tune()
        return self.setup()

    def make_env(self):
        return gym.make(self.cl_args["env_name"])

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
