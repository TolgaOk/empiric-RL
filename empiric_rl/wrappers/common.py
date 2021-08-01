import numpy as np
import gym


class FloatObservation(gym.ObservationWrapper):

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.astype(np.float32)