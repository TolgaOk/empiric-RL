import torch
import optuna
import json

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

from empiric_rl.modules.torch.mb_modules import MBDenseActorCritic, MBConvActorCritic
from empiric_rl.utils import HyperParameter
from empiric_rl.trainers.mb_trainer import MBConfig
from empiric_rl.trainers.base_trainer import TunerInfo
from empiric_rl.wrappers.common import FloatObservation


LunarLanderConfig = MBConfig(
    policy=MBDenseActorCritic,
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
                default="[128, 128]",
                tune_fn=lambda trial: trial.suggest_categorical(
                    "pi_layer_widths",
                    ["[128, 128]", "[64, 64, 64]", "[200, 300]", "[256, 256, 256]"]),
                interpret=lambda choice: json.loads(choice)),
            value_layer_widths=HyperParameter(
                default=[128, 128],
                tune_fn=None),
            pi_activation_fn=HyperParameter(
                default="ELU",
                tune_fn=None,
                interpret=lambda name: getattr(torch.nn, name)),
            value_activation_fn=HyperParameter(
                default="ELU",
                tune_fn=None,
                interpret=lambda name: getattr(torch.nn, name)),
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

BipedalWalkerConfig = MBConfig(
    policy=MBDenseActorCritic,
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
                default="[200, 300]",
                tune_fn=lambda trial: trial.suggest_categorical(
                    "pi_layer_widths",
                    ["[128, 128]", "[64, 64, 64]", "[200, 300]", "[256, 256, 256]"]),
                interpret=lambda choice: json.loads(choice)),
            value_layer_widths=HyperParameter(
                default="[256, 256, 256]",
                tune_fn=lambda trial: trial.suggest_categorical(
                    "value_layer_widths",
                    ["[128, 128]", "[64, 64, 64]", "[200, 300]", "[256, 256, 256]"]),
                interpret=lambda choice: json.loads(choice)),
            pi_activation_fn=HyperParameter(
                default="Tanh",
                tune_fn=lambda trial: trial.suggest_categorical(
                    "pi_activation_fn",
                    ["ELU", "Tanh", "ReLU"]),
                interpret=lambda choice: getattr(torch.nn, choice)),
            value_activation_fn=HyperParameter(
                default="ELU",
                tune_fn=lambda trial: trial.suggest_categorical(
                    "value_activation_fn",
                    ["ELU", "Tanh", "ReLU"]),
                interpret=lambda choice: getattr(torch.nn, choice)),
        )
    ),
    gym_wrappers=[{"class": FloatObservation, "kwargs": {}}],
    sb3_wrappers=[],
    tuner=TunerInfo(
        sampler_cls=optuna.samplers.TPESampler,
        n_startup_trials=10,
        direction="maximize",
    )
)

AtariConfig = MBConfig(
    policy=MBConvActorCritic,
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
            default=0.95,
            tune_fn=lambda trial: trial.suggest_uniform("gae_lambda", 0.5, 1.0)),
        gamma=HyperParameter(
            default=0.99,
            tune_fn=lambda trial: trial.suggest_uniform("gamma", 0.95, 0.999)),
        ent_coef=HyperParameter(
            default=0.01,
            tune_fn=lambda trial: trial.suggest_loguniform("ent_coef", 1e-5, 1e-2)),
        vf_coef=HyperParameter(
            default=0.5,
            tune_fn=lambda trial: trial.suggest_uniform("vf_coef", 0.1, 1.0)),
        max_grad_norm=HyperParameter(
            default=0.5,
            tune_fn=None),
        total_timesteps=HyperParameter(
            default=10000000,
            tune_fn=None),
        policy_kwargs=dict(
            pi_layer_widths=HyperParameter(
                default="[512]",
                tune_fn=lambda trial: trial.suggest_categorical(
                    "pi_layer_widths",
                    ["[512]", "[128, 128]", "[64, 64, 64]", "[200, 300]", "[256, 256, 256]"]),
                interpret=lambda choice: json.loads(choice)),
            value_layer_widths=HyperParameter(
                default="[512]",
                tune_fn=lambda trial: trial.suggest_categorical(
                    "value_layer_widths",
                    ["[512]", "[128, 128]", "[64, 64, 64]", "[200, 300]", "[256, 256, 256]"]),
                interpret=lambda choice: json.loads(choice)),
            pi_activation_fn=HyperParameter(
                default="ReLU",
                tune_fn=lambda trial: trial.suggest_categorical(
                    "pi_activation_fn",
                    ["ELU", "Tanh", "ReLU"]),
                interpret=lambda choice: getattr(torch.nn, choice)),
            value_activation_fn=HyperParameter(
                default="ELU",
                tune_fn=lambda trial: trial.suggest_categorical(
                    "value_activation_fn",
                    ["ELU", "Tanh", "ReLU"]),
                interpret=lambda choice: getattr(torch.nn, choice)),
            conv_net_kwargs=dict(
                activation_fn=HyperParameter(
                    default="ReLU",
                    tune_fn=lambda trial: trial.suggest_categorical(
                        "activation_fn",
                        ["ELU", "Tanh", "ReLU"]),
                    interpret=lambda choice: getattr(torch.nn, choice)),
                channel_depths=HyperParameter(
                    default="[32, 64, 64]",
                    tune_fn=lambda trial: trial.suggest_categorical(
                        "channel_depths",
                        ["[32, 64, 64]", "[64, 64, 64]"]),
                    interpret=lambda choice: json.loads(choice)),
                kernel_size=HyperParameter(
                    default=[8, 4, 3],
                    tune_fn=None),
                padding=HyperParameter(
                    default=[0, 0, 0],
                    tune_fn=None),
                stride=HyperParameter(
                    default=[4, 2, 1],
                    tune_fn=None),
                maxpool=HyperParameter(
                    default=7,
                    tune_fn=None),
            )
        )
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

MujocoWalkerConfig = MBConfig(
    policy=MBDenseActorCritic,
    hyperparameters=dict(
        lr=HyperParameter(
            default=0.00025,
            tune_fn=lambda trial: trial.suggest_loguniform("lr", 1e-5, 3e-3)),
        n_steps=HyperParameter(
            default=8,
            tune_fn=lambda trial: trial.suggest_int("n_steps", 1, 16)),
        n_envs=HyperParameter(
            default=16,
            tune_fn=None),
        gae_lambda=HyperParameter(
            default=0.95,
            tune_fn=lambda trial: trial.suggest_uniform("gae_lambda", 0.5, 1.0)),
        gamma=HyperParameter(
            default=0.999,
            tune_fn=lambda trial: trial.suggest_uniform("gamma", 0.95, 0.9999)),
        ent_coef=HyperParameter(
            default=1e-4,
            tune_fn=lambda trial: trial.suggest_loguniform("ent_coef", 1e-5, 1e-2)),
        vf_coef=HyperParameter(
            default=0.5,
            tune_fn=lambda trial: trial.suggest_uniform("vf_coef", 0.1, 1.0)),
        max_grad_norm=HyperParameter(
            default=0.5,
            tune_fn=None),
        total_timesteps=HyperParameter(
            default=5000000,
            tune_fn=None),
        policy_kwargs=dict(
            pi_layer_widths=HyperParameter(
                default="[200, 300]",
                tune_fn=lambda trial: trial.suggest_categorical(
                    "pi_layer_widths",
                    ["[128, 128]", "[64, 64, 64]", "[200, 300]", "[256, 256, 256]"]),
                interpret=lambda choice: json.loads(choice)),
            value_layer_widths=HyperParameter(
                default="[256, 256, 256]",
                tune_fn=lambda trial: trial.suggest_categorical(
                    "value_layer_widths",
                    ["[128, 128]", "[64, 64, 64]", "[200, 300]", "[256, 256, 256]"]),
                interpret=lambda choice: json.loads(choice)),
            pi_activation_fn=HyperParameter(
                default="Tanh",
                tune_fn=lambda trial: trial.suggest_categorical(
                    "pi_activation_fn",
                    ["ELU", "Tanh", "ReLU"]),
                interpret=lambda choice: getattr(torch.nn, choice)),
            value_activation_fn=HyperParameter(
                default="ELU",
                tune_fn=lambda trial: trial.suggest_categorical(
                    "value_activation_fn",
                    ["ELU", "Tanh", "ReLU"]),
                interpret=lambda choice: getattr(torch.nn, choice)),
        )
    ),
    gym_wrappers=[{"class": FloatObservation, "kwargs": {}}],
    sb3_wrappers=[],
    tuner=TunerInfo(
        sampler_cls=optuna.samplers.TPESampler,
        n_startup_trials=10,
        direction="maximize",
    )
)

all_configs = dict(
    LunarLander=LunarLanderConfig,
    BipedalWalker=BipedalWalkerConfig,
    MujocoWalker=MujocoWalkerConfig,
    AtariEnv=AtariConfig,
)
