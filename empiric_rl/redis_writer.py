from typing import Any, Union, Tuple, Dict
import optuna

from stable_baselines3.common.logger import (KVWriter, FormatUnsupportedError,
                                             Video, Figure, Image, filter_excluded_keys)


class RedisWriter(KVWriter):

    def __init__(self, trial: optuna.Trial) -> None:
        super().__init__()
        self.trial = trial

    def write(self,
              key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
              step: int = 0
              ) -> None:
        def cast_to_json_serializable(value: Any):
            if isinstance(value, Video):
                raise FormatUnsupportedError(["json"], "video")
            if isinstance(value, Figure):
                raise FormatUnsupportedError(["json"], "figure")
            if isinstance(value, Image):
                raise FormatUnsupportedError(["json"], "image")
            if hasattr(value, "dtype"):
                if value.shape == () or len(value) == 1:
                    # if value is a dimensionless numpy array or of length 1, serialize as a float
                    return float(value)
                else:
                    # otherwise, a value is a numpy array, serialize as a list or nested lists
                    return value.tolist()
            return value

        key_values = {
            key: cast_to_json_serializable(value)
            for key, value in filter_excluded_keys(key_values, key_excluded, "json").items()
        }

        storage = self.trial.storage
        user_attrs = storage.get_trial_user_attrs(self.trial._trial_id)
        prev_progress = [] if "progress" not in user_attrs else user_attrs["progress"]
        storage.set_trial_user_attr(self.trial._trial_id, "progress", [*prev_progress, key_values])

    def close(self) -> None:
        pass
