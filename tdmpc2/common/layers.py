import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy

from einops import rearrange
from typing import Tuple, List, Dict, Any
import numpy as np

def build_grid(resolution: Tuple[int, int]):
	ranges = [np.linspace(0., 1., num=res) for res in resolution]
	grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
	grid = np.stack(grid, axis=-1)
	grid = np.reshape(grid, [resolution[0], resolution[1], -1])
	grid = np.expand_dims(grid, axis=0)
	grid = grid.astype(np.float32)
	return np.concatenate([grid, 1.0 - grid], axis=-1)

class SlotEncoder(nn.Module):
	def __init__(self, resolution: Tuple[int, int], inp_channel: int, hid_dim: int, slot_dim: int):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(inp_channel, hid_dim, 5, padding=2), nn.ReLU(inplace=True),
			nn.Conv2d(hid_dim, hid_dim, 5, padding=2), nn.ReLU(inplace=True),
			nn.Conv2d(hid_dim, hid_dim, 5, padding=2), nn.ReLU(inplace=True),
			nn.Conv2d(hid_dim, hid_dim, 5, padding=2), nn.ReLU(inplace=True),
		)
		self.encoder_pos = SoftPositionEmbed(input_dim=4, hidden_size=hid_dim, resolution=resolution)
		enc_shape = (resolution[0], resolution[1], hid_dim)
		self.dense = nn.Sequential(
			nn.LayerNorm(np.product(enc_shape)),
			nn.Linear(in_features=np.product(enc_shape), out_features=512),
			nn.ReLU(),
			nn.Linear(in_features=512, out_features=slot_dim)
		)
		self.preprocess = PixelPreprocess()

	def slot_encode(self, x: torch.Tensor):
		x = self.preprocess(x)
		x = self.conv(x)
		x = x.permute(0, 2, 3, 1)
		x = self.encoder_pos(x)
		x = torch.flatten(x, 1, 2)
		return x

	def forward(self, x: torch.Tensor):
		x = self.preprocess(x)
		x = self.conv(x)
		x = x.permute(0, 2, 3, 1)
		x = self.encoder_pos(x)
		x = rearrange(x, 'b h w d -> b (h w d)')
		x = self.dense(x)
		return x


class SlotDecoder(nn.Module):
	def __init__(self, hid_dim: int, resolution: Tuple[int, int] = (8, 8), slot_dim: int = 64):
		super().__init__()
		self.conv = nn.Sequential(
			nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(inplace=True),
			nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(inplace=True),
			nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(inplace=True),
			nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)
		)
		self.decoder_initial_size = (8, 8)
		self.decoder_pos = SoftPositionEmbed(input_dim=4, hidden_size=hid_dim, resolution=self.decoder_initial_size)
		self.resolution = resolution

		self.de_dense = nn.Sequential(
			nn.Linear(slot_dim, 512),
			nn.ReLU(),
			nn.Linear(512, hid_dim * resolution[0] * resolution[1])
		)	

	def slot_decode(self, z_hat: torch.Tensor):
		b, n, d = z_hat.shape
		z_hat = z_hat.reshape(b * n, d)[:, None, None, :].repeat(1, *self.resolution, 1)  # (b*n, h, w, d)
		z_hat = self.decoder_pos(z_hat)
		z_hat = rearrange(z_hat, '(b n) h w d -> (b n) d h w', b=b, n=n, d=d)
		outs = self.conv(z_hat)  # (batch * n_slots, channels, h, w)
		recons, masks = rearrange(outs, '(b n) d h w -> b n d h w', b=b, n=n).split([3, 1], dim=2)
		# normalize alpha masks over slots
		masks = F.softmax(masks, dim=1)
		recon_combined = torch.sum(recons * masks, dim=1)  # recombine into image using masks
		return recon_combined, masks

	def forward(self, z_hat: torch.Tensor):
		z_hat = self.de_dense(z_hat)
		z_hat = rearrange(z_hat, '... (h w d) -> ... h w d', h=self.resolution[0], w=self.resolution[1])
		
		print(z_hat.shape)

class SoftPositionEmbed(nn.Module):
	def __init__(self, input_dim: int, hidden_size: int, resolution: Tuple[int, int]):
		super(SoftPositionEmbed, self).__init__()
		self.linear = nn.Linear(input_dim, hidden_size)
		self.grid = torch.from_numpy(build_grid(resolution))
		self.grid = self.grid.requires_grad_(False).cuda()

	def forward(self, inputs: torch.Tensor):
		return inputs + self.linear(self.grid)

class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
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


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
		elif k == 'rgb':
			if cfg.slot_ae:
				res = (cfg.obs_shape[k][1], cfg.obs_shape[k][2])  # (H, W)
				inp_channel = cfg.obs_shape[k][0]
				slot_channels = 8
				out[k] = SlotEncoder(resolution=res, inp_channel=inp_channel, hid_dim=slot_channels, slot_dim=cfg.latent_dim)
			else:
				out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Converts a checkpoint from our old API to the new torch.compile compatible API.
	"""
	# check whether checkpoint is already in the new format
	if "_detach_Qs_params.0.weight" in source_state_dict:
		return source_state_dict

	name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
	new_state_dict = dict()

	# rename keys
	for key, val in list(source_state_dict.items()):
		if key.startswith('_Qs.'):
			num = key[len('_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_Qs.params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
			new_total_key = "_detach_Qs_params." + new_key
			new_state_dict[new_total_key] = val
		elif key.startswith('_target_Qs.'):
			num = key[len('_target_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_target_Qs_params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val

	# add batch_size and device from target_state_dict to new_state_dict
	for prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
		for key in ('__batch_size', '__device'):
			new_key = prefix + 'params.' + key
			new_state_dict[new_key] = target_state_dict[new_key]

	# check that every key in new_state_dict is in target_state_dict
	for key in new_state_dict.keys():
		assert key in target_state_dict, f"key {key} not in target_state_dict"
	# check that all Qs keys in target_state_dict are in new_state_dict
	for key in target_state_dict.keys():
		if 'Qs' in key:
			assert key in new_state_dict, f"key {key} not in new_state_dict"
	# check that source_state_dict contains no Qs keys
	for key in source_state_dict.keys():
		assert 'Qs' not in key, f"key {key} contains 'Qs'"

	# copy log_std_min and log_std_max from target_state_dict to new_state_dict
	new_state_dict['log_std_min'] = target_state_dict['log_std_min']
	new_state_dict['log_std_dif'] = target_state_dict['log_std_dif']

	# copy new_state_dict to source_state_dict
	source_state_dict.update(new_state_dict)

	return source_state_dict




def deconv(out_shape, latent_dim, num_channels, act=None):
	"""
	Transposed convolutional decoder that reconstructs an image
	with shape `out_shape` from a latent vector.
	Supports arbitrary output resolutions.
	"""
	assert len(out_shape) == 3, f"Expected shape (C, H, W), got {out_shape}"
	C, H, W = out_shape

	layers = [
		nn.Linear(latent_dim, num_channels * 8 * 8), nn.ReLU(inplace=False),
		nn.Unflatten(1, (num_channels, 8, 8)),
		nn.ConvTranspose2d(num_channels, num_channels, 3, stride=1), nn.ReLU(inplace=False),
		nn.ConvTranspose2d(num_channels, num_channels, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=False),
		nn.ConvTranspose2d(num_channels, num_channels, 4, stride=2, padding=1), nn.ReLU(inplace=False),
		nn.ConvTranspose2d(num_channels, C, 4, stride=2, padding=1)  # output: (B, C, H?, W?)
	]

	model = nn.Sequential(*layers)

	class ResizeWrapper(nn.Module):
		def __init__(self, model, target_hw):
			super().__init__()
			self.model = model
			self.target_hw = target_hw

		def forward(self, z):
			reshaped = False
			if len(z.shape) > 2:
				# add all previous dimensions to the batch dimension
				b1, b2 = z.shape[0], z.shape[1]
				z = rearrange(z, 'b1 b2 ... -> (b1 b2) ...')
				reshaped = True
			# run through model
			x = self.model(z)
			x = F.interpolate(x, size=self.target_hw, mode='bilinear', align_corners=False)
			if reshaped:
				# reshape back to original shape
				x = rearrange(x, '(b1 b2) ... -> b1 b2 ...', b1=b1, b2=b2)
			return x

	return ResizeWrapper(model, target_hw=(H, W))


def dec(cfg, out={}):
	"""
	Returns a dictionary of decoders that reconstruct original observations from latent states.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.latent_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.obs_shape[k][0])
		elif k == 'rgb':
			if cfg.slot_ae:
				res = (cfg.obs_shape[k][1], cfg.obs_shape[k][2])  # (H, W)
				slot_channels = 8
				out[k] = SlotDecoder(hid_dim=slot_channels, resolution=res, slot_dim=cfg.latent_dim)  # Or change resolution if needed
			else:
				out[k] = deconv(cfg.obs_shape[k], cfg.latent_dim, cfg.num_channels)
		else:
			raise NotImplementedError(f"Decoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)