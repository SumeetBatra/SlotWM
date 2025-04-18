import torch
import torch.nn as nn

from typing import Optional, Optional
from copy import deepcopy
from tensordict import from_modules

def weight_init(m):
	"""Custom weight initialization for TD-MPC2."""
	if isinstance(m, nn.Linear):
		nn.init.trunc_normal_(m.weight, std=0.02)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Embedding):
		nn.init.uniform_(m.weight, -0.02, 0.02)
	elif isinstance(m, nn.ParameterList):
		for i,p in enumerate(m):
			if p.dim() == 3: # Linear
				nn.init.trunc_normal_(p, std=0.02) # Weight
				nn.init.constant_(m[i+1], 0) # Bias


def zero_(params):
	"""Initialize parameters to zero."""
	for p in params:
		p.data.fill_(0)

class MLPDynamicsModel(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int):
        super(MLPDynamicsModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, action: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x = torch.cat((action, latent), dim=-1)
        return self.mlp(x)

class RewardModel(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, task_dim: int = 0):
        super(RewardModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + action_dim + task_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, action: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x = torch.cat([action, latent], dim=-1)
        return self.mlp(x)

class Actor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, task_dim: int = 0):
        super(Actor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + task_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * action_dim)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.mlp(latent)

class Critic(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, task_dim: int = 0):
        super(Critic, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + task_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        if task is not None:
            x = torch.cat((latent, task, action), dim=-1)
        else:
            x = torch.cat((latent, action), dim=-1)
        return self.mlp(x)

class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules. Borrowed from the TD-MPC2 Code
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules[0])
		self._n = len(modules)

	def __len__(self):
		return self._n

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return f'Vectorized {len(self)}x ' + self._repr


class Predictor(nn.Module):
    """Base class for a predictor based on slot_embs."""

    def forward(self, x):
        raise NotImplementedError

    def burnin(self, x):
        pass

    def reset(self):
        pass


class TransformerPredictor(Predictor):
    """Transformer encoder with input/output projection."""

    def __init__(
            self,
            input_dim=518,
            output_dim=518,
            d_model=128,
            num_layers=1,
            num_heads=4,
            ffn_dim=256,
            norm_first=True,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

        transformer_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            norm_first=norm_first,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_enc_layer,
            num_layers=num_layers,
        )

    def forward(self, x):
        # x: (b, 518)
        x = x.unsqueeze(1)  # (b, 1, 518)
        x = self.input_proj(x)  # (b, 1, d_model)
        x = self.transformer_encoder(x)  # (b, 1, d_model)
        x = self.output_proj(x)  # (b, 1, output_dim)
        return x.squeeze(1)  # (b, output_dim)