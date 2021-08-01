from typing import Callable, Dict, NamedTuple, Any, Optional, List, Union
import gym
import os
import json
import numpy as np
import optuna


class HyperParameter(NamedTuple):
    default: Any
    tune_fn: Optional[Callable] = None
    interpret: Optional[Callable] = None


def apply_wrappers(environment: gym.Env, wrappers: List[gym.Wrapper]) -> gym.Env:
    for wrapper in wrappers:
        environment = wrapper(environment)
    return environment


def realize_hyperparameter(collection: Dict[str, Union[Dict[str, HyperParameter], HyperParameter]],
                           trial: Optional[optuna.Trial] = None
                           ) -> Dict[Union[Dict[str, Any]], Any]:
    realized_collection = dict()
    for name, value in collection.items():
        if isinstance(value, HyperParameter):
            realization = _realize_value(value, trial)
        elif isinstance(value, dict):
            realization = realize_hyperparameter(value, trial)
        else:
            raise ValueError("Unexpected value type: {} in the collection".format(type(value)))
        realized_collection[name] = realization
    return realized_collection


def _realize_value(value: HyperParameter, trial: optuna.Trial):
    if trial is None:
        return value.default
    if value.tune_fn is None:
        return value.default
    if value.interpret is not None:
        return value.interpret(value.tune_fn(trial))
    return value.tune_fn(trial)

def auto_file_name(dir_path: str, name: str, suffix=""):
    path = os.path.join(dir_path, name + suffix)
    if os.path.exists(path) is False:
        return path
    else:
        suffix = "_1" if suffix == "" else "_{}".format(int(suffix[1:])+1)
        return auto_file_name(dir_path, name, suffix)


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
        raise ValueError("Key: {} is not found in progress.json".format(key_name))
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
