import torch
import optuna

from empiric_rl.modules import DenseActorCritic
from empiric_rl.common import HyperParameter
from empiric_rl.sb3 import Config, TunerInfo


LunarLanderConfig = Config(
    policy=DenseActorCritic,
    hyperparameters=dict(
        lr=HyperParameter(
            default=3e-4,
            tune_fn=None),
        n_steps=HyperParameter(
            default=5,
            tune_fn=lambda trial: trial.suggest_int("n_steps", 1, 20, 1)),
        gae_lambda=HyperParameter(
            default=0.95,
            tune_fn=None),
        gamma=HyperParameter(
            default=0.99,
            tune_fn=None),
        ent_coef=HyperParameter(
            default=0.1,
            tune_fn=None),
        vf_coef=HyperParameter(
            default=0.5,
            tune_fn=None),
        max_grad_norm=HyperParameter(
            default=5.0,
            tune_fn=None),
        total_timesteps=HyperParameter(
            default=100000,
            tune_fn=None),
        policy_kwargs=dict(
            pi_layer_widths=HyperParameter(
                default=[128, 128],
                tune_fn=None),
            value_layer_widths=HyperParameter(
                default=[128, 128],
                tune_fn=None),
            pi_activation_fn=HyperParameter(
                default=torch.nn.ELU,
                tune_fn=None),
            value_activation_fn=HyperParameter(
                default=torch.nn.ELU,
                tune_fn=None),
        )
    ),
    gym_wrappers=[],
    sb3_wrappers=[],
    tuner=TunerInfo(
        sampler_cls=optuna.samplers.TPESampler,
        n_startup_trials=5,
        n_trials=5,
        n_procs=3,
        direction="maximize",
    )
)


all_configs = dict(
    LunarLander=LunarLanderConfig,
)
