from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Union, Tuple
import gym
import os
import json
import numpy as np
import optuna
from warnings import warn
from functools import partial

from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper


@dataclass
class HyperParameter():
    default: Any
    tune_fn: Optional[Callable] = None
    interpret: Optional[Callable] = None


def apply_wrappers(environment: gym.Env,
                   wrappers: List[Dict[str, Union[VecEnvWrapper, Dict[str, Any]]]]
                   ) -> gym.Env:
    for wrapper_info in wrappers:
        environment = wrapper_info["class"](environment, **wrapper_info["kwargs"])
    return environment


def _realize_by_default_values(name: str, value: HyperParameter, trial: optuna.Trial
                               ) -> Tuple[Any, Any]:
    _realize_by_sample(name, value, trial)
    if trial is not None and value.tune_fn is not None:
        trial.storage.set_trial_param(
            trial_id=trial._trial_id,
            param_name=name,
            param_value_internal=trial.distributions[name].to_internal_repr(value.default),
            distribution=trial.distributions[name])
    if value.interpret is not None:
        return value.interpret(value.default), value.default
    return value.default, value.default


def _realize_by_repeat(master_trial: optuna.Trial, name: str,
                       value: HyperParameter, trial: optuna.Trial
                       ) -> Tuple[Any, Any]:
    if trial is None:
        raise RuntimeError("Repeating can only be used in tune mode")
    if value.tune_fn is None:
        param_value = value.default
    else:
        param_value = master_trial.params[name]
        trial.storage.set_trial_param(
            trial_id=trial._trial_id,
            param_name=name,
            param_value_internal=master_trial.distributions[name].to_internal_repr(param_value),
            distribution=master_trial.distributions[name])
    if value.interpret is not None:
        return value.interpret(param_value), param_value
    return param_value, param_value


def _realize_by_sample(name: str, value: HyperParameter, trial: optuna.Trial
                       ) -> Tuple[Any, Any]:
    if trial is None or value.tune_fn is None:
        param_value = value.default
    else:
        param_value = value.tune_fn(trial)
    if value.interpret is not None:
        return value.interpret(param_value), param_value
    return param_value, param_value


def _realize_hyperparameter(collection: Dict[str, Union[Dict[str, HyperParameter], HyperParameter]],
                            trial: Optional[optuna.Trial] = None,
                            realize_fn: Callable[[HyperParameter],
                                                 optuna.Trial] = _realize_by_sample
                            ) -> Tuple[Dict[Dict[str, Any], Any],
                                       Dict[Dict[str, Any], Any]]:
    realization_collection = dict()
    json_realization_collection = dict()
    for name, value in collection.items():
        if isinstance(value, HyperParameter):
            realization, json_realization = realize_fn(name, value, trial)
        elif isinstance(value, dict):
            realization, json_realization = _realize_hyperparameter(value, trial, realize_fn)
        else:
            raise ValueError("Unexpected value type: {} in the collection".format(type(value)))
        realization_collection[name] = realization
        json_realization_collection[name] = json_realization
    return realization_collection, json_realization_collection


def realize_hyperparameters(collection: Dict[str, Union[Dict[str, HyperParameter], HyperParameter]],
                            trial: Optional[optuna.Trial] = None,
                            n_repeat: int = 1,
                            start_with_default: bool = False
                            ) -> Tuple[Dict[Dict[str, Any], Any],
                                       Dict[Dict[str, Any], Any]]:
    if n_repeat > 1 and trial is None:
        raise ValueError("Trial is missing! Parameter n_repeat > 1 must be used in tune mode")
    realize_fn = _realize_by_sample
    if trial is not None and trial.number == 0 and start_with_default:
        realize_fn = _realize_by_default_values
    if trial is not None and trial.number % n_repeat != 0:
        master_trial_id = trial._trial_id - (trial.number % n_repeat)
        frozen_master_trial = trial.storage.get_trial(master_trial_id)
        realize_fn = partial(_realize_by_repeat, frozen_master_trial)
    return _realize_hyperparameter(collection, trial, realize_fn)


def auto_file_name(dir_path: str, name: str):

    def is_used_fn(name: str) -> bool:
        path = os.path.join(dir_path, name)
        return os.path.exists(path)

    unique_name = _auto_name(is_used_fn, name)
    return os.path.join(dir_path, unique_name)


def auto_redis_name(redis_study_names: List[str], name:  str):

    def is_used_fn(name: str) -> bool:
        return name in redis_study_names

    return _auto_name(is_used_fn, name)


def _auto_name(is_used_fn: Callable[[str], bool], name: str, suffix=""):
    if is_used_fn(name + suffix) is False:
        return name + suffix
    else:
        new_suffix = "_1" if suffix == "" else "_{}".format(int(suffix[1:])+1)
        return _auto_name(is_used_fn, name, suffix=new_suffix)


def make_run_dir(dir_path: str, log_folder_name: str):
    dir_path = auto_file_name(dir_path, log_folder_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def load_from_progress(log_dir: str, key_name: str):
    path = os.path.join(log_dir, "progress.json")
    if not os.path.exists(path):
        raise FileNotFoundError("progress.json not found in {}".format(path))

    with open(path, "r") as fobj:
        json_strings = fobj.readlines()

    progress = list(map(json.loads, json_strings))

    values = [float(row[key_name]) for row in progress if key_name in row.keys()]
    if len(values) == 0:
        warn("Key: {} is not found in progress.json".format(key_name))
        return [-np.inf]
    return values


def log_weighted_average_score(log_dir: str, key_name: str, minimum_decay: float = 0.01):
    values = np.array(load_from_progress(log_dir, key_name), dtype=np.float64)
    decay_ratio = np.exp(np.log(minimum_decay) / len(values))
    decays = decay_ratio ** np.arange(len(values))[::-1]
    weights = decays / np.sum(decays)

    not_nan_indices = ~np.isnan(values)
    return np.sum(values[not_nan_indices] * weights[not_nan_indices])


def percentage_score(log_dir: str, key_name: str, percentage: float):
    values = np.array(load_from_progress(log_dir, key_name), dtype=np.float64)
    not_nan_indices = ~np.isnan(values)
    values = values[not_nan_indices]
    index = np.clip(1, len(values), int(len(values) * percentage))
    return np.mean(values)
