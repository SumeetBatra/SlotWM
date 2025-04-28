import gymnasium as gym
import numpy as np
import mani_skill.envs

from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
from envs.wrappers.record_episode import RecordEpisodeWrapper
from envs.wrappers.pixels import PixelWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils import gym_utils
from functools import partial


def make_envs(cfg, num_envs, record_video_path, is_eval, logger):
    """Make ManiSkill3 environment."""
    record_episode_kwargs = dict(save_video=True, save_trajectory=False, record_single=True, info_on_video=False)

    env_make_fn = partial(
        gym.make,
        disable_env_checker=True,
        id=cfg.task,
        obs_mode=cfg.obs,
        render_mode=cfg.render_mode,
        sensor_configs=dict(width=cfg.render_size, height=cfg.render_size),
    )
    if cfg.control_mode != 'default':
        env_make_fn = partial(env_make_fn, control_mode=cfg.control_mode)
    if is_eval:
        env_make_fn = partial(env_make_fn, reconfiguration_freq=cfg.eval_reconfiguration_frequency)

    assert cfg.env_type == 'gpu', "Only gpu environments are supported at this time"
    env = env_make_fn(num_envs=num_envs)
    max_episode_steps = gym_utils.find_max_episode_steps_value(env)
    if cfg.obs == 'rgb':
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=cfg.include_state)
        env = PixelWrapper(cfg, env, num_envs)

    if record_video_path is not None:
        env = RecordEpisodeWrapper(
            env,
            record_video_path,
            trajectory_name=f"trajectory",
            max_steps_per_video=max_episode_steps,
            save_video=record_episode_kwargs["save_video"],
            save_trajectory=record_episode_kwargs["save_trajectory"],
            logger=logger,
        )
    env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    env.max_episode_steps = max_episode_steps
    return env
