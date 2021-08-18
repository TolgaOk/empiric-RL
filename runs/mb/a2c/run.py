from typing import List
import argparse
import numpy as np
import torch

from modular_baselines.algorithms.a2c.a2c import A2C
from modular_baselines.loggers.basic import InitLogCallback, LogRolloutCallback, LogLossCallback

from empiric_rl.trainers.mb_trainer import MBExperiment
from empiric_rl.utils import log_weighted_average_score
from configs import all_configs


class A2CExperiment(MBExperiment):

    def _setup(self, hyperparameters, vecenv, logger, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        policy = self.config.policy(observation_space=vecenv.observation_space,
                                    action_space=vecenv.action_space,
                                    lr=hyperparameters["lr"],
                                    **hyperparameters["policy_kwargs"])
        policy.to(self.cl_args["device"])
        agent = A2C.setup(
            env=vecenv,
            policy=policy,
            rollout_len=hyperparameters["n_steps"],
            ent_coef=hyperparameters["ent_coef"],
            value_coef=hyperparameters["vf_coef"],
            gamma=hyperparameters["gamma"],
            gae_lambda=hyperparameters["gae_lambda"],
            max_grad_norm=hyperparameters["max_grad_norm"],
            buffer_callbacks=None,
            collector_callbacks=LogRolloutCallback(logger),
            algorithm_callbacks=[InitLogCallback(logger,
                                                 self.cl_args["log_interval"]),
                                 LogLossCallback(logger)])

        agent.learn(total_timesteps=hyperparameters["total_timesteps"])
        score = log_weighted_average_score(logger.get_dir(), "rollout/ep_rew_mean")
        return agent, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Baselines 3 A2C")
    A2CExperiment.add_parse_arguments(parser)

    args = parser.parse_args()
    args = vars(args)

    print(A2CExperiment(all_configs, args, algo_name="A2C").run())
