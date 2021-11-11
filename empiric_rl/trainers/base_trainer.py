from abc import ABC, abstractmethod
import warnings
from typing import List, Any, Dict, Union, Optional, Tuple, Callable
from dataclasses import dataclass
import os
import argparse
import numpy as np
import optuna
import pickle
import gym
import json
import socket
import tempfile
from optuna.trial import TrialState

from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from empiric_rl.utils import (HyperParameter,
                              realize_hyperparameters,
                              apply_wrappers,
                              make_run_dir,
                              auto_redis_name)
from empiric_rl.redis_writer import RedisWriter


@dataclass
class TunerInfo:
    sampler_cls: optuna.samplers.BaseSampler
    n_startup_trials: int
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
                direction=config.tuner.direction)
        )


class BaseExperiment(ABC):

    def __init__(self,
                 configs: BaseConfig,
                 cl_args: Dict[str, Any],
                 exp_name_prefix: Optional[str] = "",):
        self.cl_args = cl_args
        env_class_name = self.make_env().env.__class__.__name__
        self.config = configs[env_class_name]
        self.exp_name = "_".join([exp_name_prefix, env_class_name])
        self.main_dir = cl_args["log_dir"]
        if self.main_dir is None:
            self.main_dir = tempfile.TemporaryDirectory().name
        if self.cl_args["tune"] is False:
            if self.cl_args["n_seeds"] > 1 or self.cl_args["start_tune_with_default_params"]:
                raise ValueError("CL argumenents 'n-seeds' and 'start-tune-with-default-params'"
                                 " can only be used in tune mode")
        if self.cl_args["tune"]:
            self.main_dir = make_run_dir(self.main_dir, "Tune_"+self.exp_name)
        
        self._modify_default_params()

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

    def _modify_default_params(self):
        default_param_path = self.cl_args["default_parameter_path"]
        if default_param_path is None:
            return
        with open(default_param_path, "r") as file:
            defaults = json.load(file)

        def modify(defaults_dict, hyperparameters_dict):
            for key, value in defaults_dict.items():
                if isinstance(value, dict):
                    modify(value[key], hyperparameters_dict[key])
                else:
                    hyperparameters_dict[key].default = value
        
        modify(defaults, self.config.hyperparameters)
        

    def setup(self, trial: Optional[optuna.Trial] = None) -> float:
        hyperparameters, jsonized_hyperparameters = realize_hyperparameters(
            self.config.hyperparameters,
            trial=trial,
            n_repeat=self.cl_args["n_seeds"],
            start_with_default=self.cl_args["start_tune_with_default_params"])
        seed = self.make_seed(self.cl_args["seed"], trial)
        json_ready_meta_data = dict(commandline_args=self.cl_args,
                                        config=self.config_encoder_class.encode(
                                            self.config, jsonized_hyperparameters, seed),
                                        local_ip_adress=self.get_local_ip())

        vecenv = make_vec_env(
            self.make_env,
            n_envs=hyperparameters["n_envs"],
            seed=seed,
            wrapper_class=lambda env: apply_wrappers(env, self.config.gym_wrappers),
            vec_env_cls=SubprocVecEnv)
        vecenv = apply_wrappers(vecenv, self.config.sb3_wrappers)

        eval_env = make_vec_env(
            self.make_env,
            n_envs=1,
            seed=seed,
            wrapper_class=lambda env: apply_wrappers(env, self.config.gym_wrappers),
            vec_env_cls=DummyVecEnv
        )
        eval_env = apply_wrappers(eval_env, self.config.sb3_wrappers)

        log_dir = make_run_dir(self.main_dir, self.exp_name)
        logger = configure(log_dir, ["stdout", "json", "tensorboard", "csv"])
        if trial is not None:
            logger.output_formats.append(RedisWriter(trial))
            trial.storage.set_trial_user_attr(trial._trial_id, "meta-data", json_ready_meta_data)

        agent, score = self._setup(hyperparameters, vecenv, logger, seed, eval_env)

        if self.cl_args["save_model"]:
            agent.save(log_dir)
        with open(os.path.join(log_dir, "meta-data.json"), "w") as file:
            json.dump(json_ready_meta_data, file)
        return score

    def tune(self) -> None:
        storage_url = self.cl_args["storage_url"]
        study_name = self.exp_name
        if storage_url is None:
            storage_url = "".join(("sqlite:///", os.path.join(self.main_dir, "store.db")))
        if storage_url.startswith("redis://") and not self.cl_args["continue_study"]:
            study_name = auto_redis_name(self.get_all_redis_study_names(storage_url), study_name)
        if self.cl_args["study_name"] is not None:
            study_name = self.cl_args["study_name"]

        sampler = self.config.tuner.sampler_cls(
            n_startup_trials=self.config.tuner.n_startup_trials)
        study = optuna.create_study(
            storage=storage_url,
            sampler=sampler,
            study_name=study_name,
            direction=self.config.tuner.direction,
            load_if_exists=self.cl_args["continue_study"])
        study.set_user_attr("max_trials", self.cl_args["max_trials"])

        self.optimize_study(study, n_trials=self.cl_args["max_trials"])

    def get_all_redis_study_names(self, redis_storage_url: str) -> List[str]:
        redis_storage = optuna.storages.RedisStorage(redis_storage_url)
        study_ids = [pickle.loads(sid) for sid in redis_storage._redis.lrange("study_list", 0, -1)]
        study_names = [redis_storage.get_study_name_from_id(_id) for _id in study_ids]
        return study_names

    def get_local_ip(self):
        socket_obj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        socket_obj.connect(('8.8.8.8', 1))
        return socket_obj.getsockname()[0]

    def optimize_study(self, study: optuna.Study, n_trials: int):
        if len(study.trials) >= n_trials:
            warnings.warn("Study: {} has already {} >= {} many trials".format(
                study.study_name, len(study.trials), n_trials))
        while len(study.trials) < n_trials:
            trial = study.ask()
            try:
                if trial.number > n_trials:
                    study.tell(trial, None, state=TrialState.FAIL)
                else:
                    score = self.setup(trial)
                    study.tell(trial, score, state=TrialState.COMPLETE)
            except:
                study.tell(trial, None, state=TrialState.FAIL)
                raise

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
        parser.add_argument("--n-seeds", type=int, default=1,
                            help=("Run the same parameters for n-seeds many trials. "
                                  "Used for multi seed experiments"))
        parser.add_argument("--start-tune-with-default-params", action="store_true",
                            help="Use default hyperparameters for the first trial")
        parser.add_argument("--storage-url", type=str, default=None,
                            help="Optuna storage url. Default: sqlite")
        parser.add_argument("--continue-study", action="store_true",
                            help="If given, tuner tries to load a study from the storage")
        parser.add_argument("--study-name", type=str, default=None,
                            help=("Name of the study name in the storage. "
                                  "Default: Environment name"))
        parser.add_argument("--max-trials", type=int, default=1,
                            help="Set the maximum number of trials if a new study is created")
        parser.add_argument("--default-parameter-path", type=str, default=None,
                            help="Json file path of the default hyperparamters")