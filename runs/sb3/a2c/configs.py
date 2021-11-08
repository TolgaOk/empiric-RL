import torch
import optuna
import json

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.a2c import CnnPolicy
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

from empiric_rl.modules.torch.sb3_module import SB3DenseActorCritic, SB3ConvActorCritic
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
        direction="maximize",
    )
)

# tuned with log score of 147.5278713145453
BipedalWalkerConfig = SB3Config(
    policy=SB3DenseActorCritic,
    hyperparameters=dict(
        lr=HyperParameter(
            default=0.000236,
            tune_fn=lambda trial: trial.suggest_loguniform("lr", 1e-5, 1e-2)),
        n_steps=HyperParameter(
            default=8,
            tune_fn=lambda trial: trial.suggest_int("n_steps", 1, 16)),
        n_envs=HyperParameter(
            default=16,
            tune_fn=None),
        gae_lambda=HyperParameter(
            default=0.790030,
            tune_fn=lambda trial: trial.suggest_uniform("gae_lambda", 0.5, 1.0)),
        gamma=HyperParameter(
            default=0.964923,
            tune_fn=lambda trial: trial.suggest_uniform("gamma", 0.95, 0.999)),
        ent_coef=HyperParameter(
            default=3.0012928903865465e-05,
            tune_fn=lambda trial: trial.suggest_loguniform("ent_coef", 1e-5, 1e-2)),
        vf_coef=HyperParameter(
            default=0.390421,
            tune_fn=lambda trial: trial.suggest_uniform("vf_coef", 0.1, 1.0)),
        max_grad_norm=HyperParameter(
            default=0.5,
            tune_fn=None),
        total_timesteps=HyperParameter(
            default=5000000,
            tune_fn=None),
        policy_kwargs=dict(
            pi_layer_widths=HyperParameter(
                default=[200, 300],
                tune_fn=lambda trial: trial.suggest_categorical(
                    "pi_layer_widths",
                    ["[128, 128]", "[64, 64, 64]", "[200, 300]", "[256, 256, 256]"]),
                interpret=lambda choice: json.loads(choice)),
            value_layer_widths=HyperParameter(
                default=[256, 256, 256],
                tune_fn=lambda trial: trial.suggest_categorical(
                    "value_layer_widths",
                    ["[128, 128]", "[64, 64, 64]", "[200, 300]", "[256, 256, 256]"]),
                interpret=lambda choice: json.loads(choice)),
            pi_activation_fn=HyperParameter(
                default=torch.nn.Tanh,
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
        direction="maximize",
    )
)

AtariConfig = SB3Config(
    policy="CnnPolicy",
    hyperparameters=dict(
        lr=HyperParameter(
            default=7e-4,
            tune_fn=lambda trial: trial.suggest_loguniform("lr", 1e-5, 1e-2)),
        n_steps=HyperParameter(
            default=5,
            tune_fn=lambda trial: trial.suggest_int("n_steps", 1, 16)),
        n_envs=HyperParameter(
            default=16,
            tune_fn=None),
        gae_lambda=HyperParameter(
            default=1.0,
            tune_fn=lambda trial: trial.suggest_uniform("gae_lambda", 0.5, 1.0)),
        gamma=HyperParameter(
            default=0.99,
            tune_fn=lambda trial: trial.suggest_uniform("gamma", 0.95, 0.999)),
        ent_coef=HyperParameter(
            default=0.01,
            tune_fn=lambda trial: trial.suggest_loguniform("ent_coef", 1e-5, 1e-2)),
        vf_coef=HyperParameter(
            default=0.25,
            tune_fn=lambda trial: trial.suggest_uniform("vf_coef", 0.1, 1.0)),
        max_grad_norm=HyperParameter(
            default=0.5,
            tune_fn=None),
        total_timesteps=HyperParameter(
            default=10000000,
            tune_fn=None),
    ),
    gym_wrappers=[{"class": AtariWrapper, "kwargs": {}}],
    sb3_wrappers=[{"class": VecFrameStack, "kwargs": {"n_stack": 4}},
                  {"class": VecTransposeImage, "kwargs": {}}],
    tuner=TunerInfo(
        sampler_cls=optuna.samplers.TPESampler,
        n_startup_trials=10,
        direction="maximize",
    )
)

all_configs = dict(
    LunarLander=LunarLanderConfig,
    BipedalWalker=BipedalWalkerConfig,
    AtariEnv=AtariConfig,
)
