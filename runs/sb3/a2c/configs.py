import torch
import optuna
import json

from empiric_rl.modules.torch.sb3_module import SB3DenseActorCritic
from empiric_rl.utils import HyperParameter
from empiric_rl.trainers.sb3_trainer import SB3Config
from empiric_rl.trainers.base_trainer import TunerInfo
from empiric_rl.wrappers.common import FloatObservation


LunarLanderConfig = SB3Config(
    policy=SB3DenseActorCritic,
    hyperparameters=dict(
        lr=HyperParameter(
            default=3e-4,
            tune_fn=None),
        n_steps=HyperParameter(
            default=5,
            tune_fn=lambda trial: trial.suggest_int("n_steps", 1, 20, 1)),
        n_envs=HyperParameter(
            default=8,
            tune_fn=None),
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


BipedalWalkerConfig = SB3Config(
    policy=SB3DenseActorCritic,
    hyperparameters=dict(
        lr=HyperParameter(
            default=1e-4,
            tune_fn=lambda trial: trial.suggest_loguniform("lr", 1e-5, 1e-2)),
        n_steps=HyperParameter(
            default=8,
            tune_fn=None),
        n_envs=HyperParameter(
            default=16,
            tune_fn=None),
        gae_lambda=HyperParameter(
            default=0.95,
            tune_fn=lambda trial: trial.suggest_uniform("gae_lambda", 0.5, 1.0)),
        gamma=HyperParameter(
            default=0.99,
            tune_fn=lambda trial: trial.suggest_uniform("gamma", 0.95, 0.999)),
        ent_coef=HyperParameter(
            default=0.0,
            tune_fn=lambda trial: trial.suggest_loguniform("ent_coef", 1e-5, 1e-2)),
        vf_coef=HyperParameter(
            default=0.4,
            tune_fn=lambda trial: trial.suggest_uniform("vf_coef", 0.1, 1.0)),
        max_grad_norm=HyperParameter(
            default=0.5,
            tune_fn=None),
        total_timesteps=HyperParameter(
            default=5000000,
            tune_fn=None),
        policy_kwargs=dict(
            pi_layer_widths=HyperParameter(
                default=[128, 128],
                tune_fn=lambda trial: trial.suggest_categorical(
                    "pi_layer_widths",
                    ["[128, 128]", "[64, 64, 64]", "[200, 300]"]),
                interpret=lambda choice: json.loads(choice)),
            value_layer_widths=HyperParameter(
                default=[128, 128],
                tune_fn=lambda trial: trial.suggest_categorical(
                    "value_layer_widths",
                    ["[128, 128]", "[64, 64, 64]", "[200, 300]", "[256, 256, 256]"]),
                interpret=lambda choice: json.loads(choice)),
            pi_activation_fn=HyperParameter(
                default=torch.nn.ELU,
                tune_fn=lambda trial: trial.suggest_categorical(
                    "pi_activation_fn",
                    ["ELU", "Tanh", "ReLU"]),
                interpret=lambda choice: getattr(torch.nn, choice)),
            value_activation_fn=HyperParameter(
                default=torch.nn.ELU,
                tune_fn=lambda trial: trial.suggest_categorical(
                    "value_activation_fn",
                    ["ELU", "Tanh", "ReLU"]),
                interpret=lambda choice: getattr(torch.nn, choice)),
        )
    ),
    gym_wrappers=[FloatObservation],
    sb3_wrappers=[],
    tuner=TunerInfo(
        sampler_cls=optuna.samplers.TPESampler,
        n_startup_trials=10,
        n_trials=100,
        n_procs=10,
        direction="maximize",
    )
)


all_configs = dict(
    LunarLander=LunarLanderConfig,
    BipedalWalker=BipedalWalkerConfig,
)
