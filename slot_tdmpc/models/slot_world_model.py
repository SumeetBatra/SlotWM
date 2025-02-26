import torch
import torch.nn as nn

from slot_tdmpc.models.slot_ae import SlotEncoder, SlotDecoder, SlotDecoder, SlotEncoder
from slot_tdmpc.models.wm_components import *
from slot_tdmpc.common import math
from tensordict import TensorDict, TensorDictParams
from tensordict.nn import TensorDictParams
from typing import Dict, Any


class SlotWorldModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super(SlotWorldModel, self).__init__()
        self.cfg = cfg
        if cfg.multitask:
            # TODO: can we use something more informative than random embeddings ?
            # ex. maybe we can use embeddings of the goal image as the task embedding
            # this will potentially help the agent generalize quicker to new tasks in the post-training phase
            # because there is a lot of useful, reusable information in an image
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
            for i in range(len(cfg.tasks)):
                self._action_masks[i, :cfg.action_dims[i]] = 1.

        # latent model components

        self._encoder = SlotEncoder(resolution=(64, 64), hid_dim=64, slot_dim=cfg.slot_dim)
        self._decoder = SlotDecoder(hid_dim=64)

        # TODO: we can do smth better than an mlp dynamics model
        # for example a transformer or RNN to capture short / long range temporal dependencies
        self._dynamics = MLPDynamicsModel(latent_dim=cfg.latent_dim, action_dim=cfg.action_dim, hidden_dim=cfg.hidden_dim)
        self._reward = RewardModel(latent_dim=cfg.latent_dim, action_dim=cfg.action_dim, hidden_dim=cfg.hidden_dim, task_dim=cfg.task_dim)

        # actor critic components
        self._pi = Actor(latent_dim=cfg.latent_dim, action_dim=cfg.action_dim, hidden_dim=cfg.hidden_dim, task_dim=cfg.task_dim)
        self._Qs = Ensemble([
            Critic(latent_dim=cfg.latent_dim, action_dim=cfg.action_dim, hidden_dim=cfg.hidden_dim, task_dim=cfg.task_dim)
            for _ in range(cfg.num_q)
        ])

        self.apply(weight_init)
        zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])
        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)

    def init(self):
        # Create params
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

        # Create modules
        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)

        # Assign params to modules
        # We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params

    def __repr__(self):
        repr = 'Slot World Model\n'
        modules = ['Encoder', 'Dynamics', 'Reward', 'Policy prior', 'Q-functions']
        for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._pi, self._Qs]):
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def task_emb(self, latent: torch.Tensor, task: Any):
        # TODO: more informative representation for task ex. goal image
        '''
        Instead of attaching the task emb to the input, we attach it to the latent variable
        The world model should be task agnostic and build a general "intuitive physics" model of the world,
        regardless of task. Only the actor and critic should be task conditioned
        :param latent: (N x D) latent repr of the input
        :param task: task specifier
        '''
        if isinstance(task, int):
            task = torch.tensor([task], device=latent.device)
        emb = self._task_emb(task.long())
        return torch.cat([latent, emb], dim=-1)

    def encode(self, obs: torch.Tensor):
        if self.cfg.obs == 'rgb' and obs.ndim == 5:
            # encode history of obs and stack
            return torch.stack([self._encoder(o) for o in obs])
        return self._encoder(obs)

    def next(self, z: torch.Tensor, a: torch.Tensor):
        '''
        Predict next latent state given current latent state and action
        '''
        return self._dynamics(action=a, latent=z)

    def reward(self, z: torch.Tensor, a: torch.Tensor, task: Any = None):
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        return self._reward(latent=z, action=a)

    def pi(self, z: torch.Tensor, task: Any):
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        # Gaussian policy prior
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        if self.cfg.multitask:  # Mask out unused action dimensions
            mean = mean * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:  # No masking
            action_dims = None

        log_prob = math.gaussian_logprob(eps, log_std)

        # Scale log probability by action dimensions
        size = eps.shape[-1] if action_dims is None else action_dims
        scaled_log_prob = log_prob * size

        # Reparameterization trick
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = TensorDict({
            "mean": mean,
            "log_std": log_std,
            "action_prob": 1.,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        })
        return action, info

    def Q(self, z: torch.Tensor, a: torch.Tensor, task: Any, return_type='min', target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == 'all':
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2