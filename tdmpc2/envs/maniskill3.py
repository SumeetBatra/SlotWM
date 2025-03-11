import gymnasium as gym
import numpy as np
import mani_skill.envs

from envs.wrappers.timeout import Timeout

MANISKILL_TASKS = {
    'pick-cube': dict(
        env='PickCube-v1',
        control_mode='pd_ee_delta_pos',
        max_episode_steps=50,
    ),
    'stack-cube': dict(
        env='StackCube-v1',
        control_mode='pd_ee_delta_pos',
        max_episode_steps=50,
    ),
    'pick-ycb': dict(
        env='PickSingleYCB-v1',
        control_mode='pd_ee_delta_pose',
        max_episode_steps=50,
    ),
    'turn-faucet': dict(
        env='TurnFaucet-v1',
        control_mode='pd_ee_delta_pose',
        max_episode_steps=200,
    ),
}


class ManiSkillWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.observation_space = self.env.single_observation_space
        self.action_space = gym.spaces.Box(
            low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
            high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
            dtype=self.env.action_space.dtype,
        )

    def reset(self):
        # TODO: maybe don't discard the info?
        obs, info = self.env.reset()
        return obs[0]

    def step(self, action):
        reward = 0
        for _ in range(2):
            obs, r, _, _, info = self.env.step(action)
            reward += r
        return obs[0], reward[0], False, info

    def render(self):
        return self.env.render()[0]

    @property
    def unwrapped(self):
        return self.env.unwrapped


def make_env(cfg):
    """
    Make ManiSkill3 environment.
    """
    if cfg.task not in MANISKILL_TASKS:
        raise ValueError('Unknown task:', cfg.task)
    # TODO: support image observations
    assert cfg.obs == 'state', 'This task only supports state observations.'
    task_cfg = MANISKILL_TASKS[cfg.task]
    env = gym.make(
        task_cfg['env'],
        obs_mode='state',
        control_mode=task_cfg['control_mode'],
        render_mode='all',
        human_render_camera_configs=dict(width=384, height=384),
    )
    env = ManiSkillWrapper(env, cfg)
    env = Timeout(env, max_episode_steps=task_cfg['max_episode_steps'])
    return env