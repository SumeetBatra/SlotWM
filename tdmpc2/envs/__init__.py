from copy import deepcopy
import warnings

import gymnasium as gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.tensor import TensorWrapper

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from envs.dmcontrol import make_env as make_dm_control_env
except:
	make_dm_control_env = missing_dependencies
try:
	from envs.maniskill3 import make_env as make_maniskill_env
except:
	make_maniskill_env = missing_dependencies
try:
	from envs.metaworld import make_env as make_metaworld_env
except:
	make_metaworld_env = missing_dependencies
try:
	from envs.myosuite import make_env as make_myosuite_env
except:
	make_myosuite_env = missing_dependencies
try:
	from envs.maniskill3 import make_envs as make_maniskill3_vec_env
except:
	make_maniskill3_vec_env = missing_dependencies


warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise ValueError('Unknown task:', task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env
	

def make_envs(cfg, num_envs, video_path: str = None, is_eval=False, logger=None):
	env = make_maniskill3_vec_env(cfg, num_envs, video_path, is_eval, logger)

	try: # Dict
		cfg.obs_shape = {k: v.shape[1:] for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape[1:]}
	cfg.action_dim = env.action_space.shape[1]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, cfg.num_envs * cfg.episode_length)
	return env
