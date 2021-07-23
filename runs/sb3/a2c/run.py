from typing import Any, Dict, Tuple, Union
import argparse

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import A2C

from configs import all_configs
from empiric_rl.common import log_weighted_average_score
from empiric_rl.sb3 import SB3Experiment


class A2CExperiment(SB3Experiment):

    def _setup(self,
               hyperparameters: Dict[Union[Dict[str, Any]], Any],
               logger: Logger,
               vecenv: VecEnv,
               log_dir: str
               ) -> Tuple[BaseAlgorithm, float]:
        agent = A2C(
            policy=self.config.policy,
            env=vecenv,
            learning_rate=hyperparameters["lr"],
            n_steps=hyperparameters["n_steps"],
            gamma=hyperparameters["gamma"],
            gae_lambda=hyperparameters["gae_lambda"],
            ent_coef=hyperparameters["ent_coef"],
            vf_coef=hyperparameters["vf_coef"],
            max_grad_norm=hyperparameters["max_grad_norm"],
            use_rms_prop=False,
            use_sde=False,
            normalize_advantage=False,
            tensorboard_log=log_dir,
            create_eval_env=False,
            policy_kwargs=hyperparameters["policy_kwargs"],
            verbose=1,
            seed=args["seed"],
            device=args["device"],
        )
        agent.set_logger(logger)
        agent.learn(
            total_timesteps=hyperparameters["total_timesteps"],
            callback=None,
            log_interval=args["log_interval"],
            eval_env=None,
            eval_freq=-1,
            n_eval_episodes=0,
            tb_log_name=self.exp_name,
            eval_log_path=None,
        )
        score = log_weighted_average_score(log_dir, "rollout/ep_rew_mean")
        return agent, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Baselines 3 A2C")
    A2CExperiment.common_parse_arguments(parser)

    args = parser.parse_args()
    args = vars(args)

    print(A2CExperiment(all_configs, args, algo_name="A2C").run())
