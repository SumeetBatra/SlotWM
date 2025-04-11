import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from einops import rearrange
from typing import Tuple, List, Dict, Any

# Based off of https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention.py

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min=eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))


def build_grid(resolution: Tuple[int, int]):
	ranges = [np.linspace(0., 1., num=res) for res in resolution]
	grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
	grid = np.stack(grid, axis=-1)
	grid = np.reshape(grid, [resolution[0], resolution[1], -1])
	grid = np.expand_dims(grid, axis=0)
	grid = grid.astype(np.float32)
	return np.concatenate([grid, 1.0 - grid], axis=-1)


def spatial_broadcast(slots: torch.Tensor, resolution: Tuple[int, int]):
	'''
	Broadcast the slot features to a 2D grid and collapse slot dim
	:param slots: (batch_size x num_slots x slot_size) tensor
	'''
	slots = slots.reshape(-1, slots.shape[-1])[:, None, None, :]
	grid = torch.tile(slots, dims=(1, resolution[0], resolution[1], 1))
	# grid has shape (batch_size * n_slots, width_height, slot_size)
	return grid


def gumbel_softmax(logits, temperature = 1.):
    dtype, size = logits.dtype, logits.shape[-1]

    assert temperature > 0

    scaled_logits = logits / temperature

    # gumbel sampling and derive one hot

    noised_logits = scaled_logits + gumbel_noise(scaled_logits)

    indices = noised_logits.argmax(dim = -1)

    hard_one_hot = F.one_hot(indices, size).type(dtype)

    # get soft for gradients

    soft = scaled_logits.softmax(dim = -1)

    # straight through

    hard_one_hot = hard_one_hot + soft - soft.detach()

    # return indices and one hot

    return hard_one_hot, indices


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim: int, num_iters: int = 3, eps: float = 1e-8, hidden_dim: int = 128):
        super(SlotAttention, self).__init__()
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.eps = eps
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.scale = dim ** -0.5

        # slots sampled initially from independent gaussians with mean mu and variance diag(sigma)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor, num_slots=None):
        '''
        :param inputs: (Batch size x Num tokens x input_dim) tensor
        :param num_slots: optional number of slots to use if different from initial num_slots
        :return: (batch_size x num_slots x output_dim) tensor
        '''
        (b, n, d), device, dtype = inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)

        inputs = self.norm_inputs(inputs)
        k, v = self.K(inputs), self.V(inputs)

        for _ in range(self.num_iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.Q(slots)

            dots = torch.einsum('bid, bjd->bij', q, k) * self.scale
            # this is where we normalize across slots dimension to force slots to compete for input tokens
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd, bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, None


class AdaptiveSlotWrapper(nn.Module):
    def __init__(self, slot_attn: SlotAttention, temperature: float = 1.0):
        super(AdaptiveSlotWrapper, self).__init__()
        self.slot_attn = slot_attn
        self.temperature = temperature
        slot_dim = slot_attn.dim
        self.pred_keep_slot = nn.Linear(slot_dim, 2, bias=False)

    def forward(self, x: torch.Tensor):
        slots, _ = self.slot_attn(x)
        keep_slot_logits = self.pred_keep_slot(slots)
        keep_slots, _ = gumbel_softmax(keep_slot_logits, temperature=self.temperature)

        # just use last column for "keep" mask
        keep_slots = keep_slots[..., -1]  # Float["batch num_slots"] of {0., 1.}
        return slots, keep_slots



class SlotEncoder(nn.Module):
	def __init__(self, resolution: Tuple[int, int], inp_channel: int, hid_dim: int):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(inp_channel, hid_dim, 5, padding=2), nn.ReLU(inplace=True),
			nn.Conv2d(hid_dim, hid_dim, 5, padding=2), nn.ReLU(inplace=True),
			nn.Conv2d(hid_dim, hid_dim, 5, padding=2), nn.ReLU(inplace=True),
			nn.Conv2d(hid_dim, hid_dim, 5, padding=2), nn.ReLU(inplace=True),
		)
		self.encoder_pos = SoftPositionEmbed(input_dim=4, hidden_size=hid_dim, resolution=resolution)
		self.preprocess = PixelPreprocess()

	def forward(self, x: torch.Tensor):
		x = self.preprocess(x)
		x = self.conv(x)
		x = x.permute(0, 2, 3, 1)
		x = self.encoder_pos(x)
		x = torch.flatten(x, 1, 2)
		return x



class SlotDecoder(nn.Module):
	def __init__(self, hid_dim: int, out_channel: int, resolution: Tuple[int, int] = (64, 64)):
		super().__init__()
		self.num_images = out_channel // 3
		conv_out = 4 * self.num_images
		self.conv = nn.Sequential(
			nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(inplace=True),
			nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(inplace=True),
			nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(inplace=True),
			nn.ConvTranspose2d(hid_dim, conv_out, 3, stride=(1, 1), padding=1)
		)

		self.decoder_initial_size = (resolution[0]//8, resolution[1]//8)
		self.decoder_pos = SoftPositionEmbed(input_dim=4, hidden_size=hid_dim, resolution=self.decoder_initial_size)

	def forward(self, z_hat: torch.Tensor):
		b, n, d = z_hat.shape
		z_hat = z_hat.reshape(b * n, d)[:, None, None, :].repeat(1, *self.decoder_initial_size, 1)  # (b*n, h, w, d)
		z_hat = self.decoder_pos(z_hat)
		z_hat = rearrange(z_hat, '(b n) h w d -> (b n) d h w', b=b, n=n, d=d)
		outs = self.conv(z_hat)  # (batch * n_slots, channels, h, w)
		outs = rearrange(outs, '(b n) d h w -> b n d h w', b=b, n=n)
		splits = outs.split([3, 1] * self.num_images, dim=2)
		recon_combined = []
		all_masks = []
		for i in range(self.num_images):
			recons, masks = splits[i*2], splits[i*2+1]
			# normalize alpha masks over slots
			masks = F.softmax(masks, dim=1)
			all_masks.append(masks)
			recon_combined.append(torch.sum(recons * masks, dim=1))  # recombine into image using masks
		recon_combined = torch.concatenate(recon_combined, 1)
		all_masks = torch.concatenate(all_masks, 1)
		return recon_combined, all_masks


class SoftPositionEmbed(nn.Module):
	def __init__(self, input_dim: int, hidden_size: int, resolution: Tuple[int, int]):
		super(SoftPositionEmbed, self).__init__()
		self.linear = nn.Linear(input_dim, hidden_size)
		self.grid = torch.from_numpy(build_grid(resolution))
		self.grid = self.grid.requires_grad_(False)

	def forward(self, inputs: torch.Tensor):
		return inputs + self.linear(self.grid.to(inputs.device))


class SlotAutoEncoder(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], num_slots: int, slot_dim: int, adaptive: bool, lambdas: Dict[str, Any]):
        super(SlotAutoEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.num_slots = num_slots
        self.latent_dim = slot_dim
        self.encoder = SlotEncoder(resolution=(64, 64), hid_dim=slot_dim)
        self.decoder = SlotDecoder(hid_dim=slot_dim)
        self.adaptive = adaptive
        slot_model = SlotAttention(num_slots=num_slots, dim=slot_dim)
        if self.adaptive:
            self.latent = AdaptiveSlotWrapper(slot_model)
        else:
            self.latent = slot_model
        self.lambdas = lambdas

    def forward(self, x: torch.Tensor):
        pre_z = self.encoder(x)
        slots, keep_slots = self.latent(pre_z)
        recon, masks = self.decoder(slots)
        outs = {
            'pre_z': pre_z,
            'slots': slots,
            'keep_slots': keep_slots,
            'x_hat_logits': recon,
            'masks': masks
        }
        return outs

    def batched_loss(self, batch):
        outs = self(batch['x'])
        bce_loss = F.binary_cross_entropy_with_logits(outs['x_hat_logits'], target=batch['x'], reduction='none').sum((1, 2, 3)).mean()
        slot_reg = 0.0
        if self.adaptive:
            slot_reg = outs['keep_slots'].sum(1).mean()

        total_loss = self.lambdas['recon'] * bce_loss + self.lambdas['slot_reg'] * slot_reg

        total_loss = total_loss.mean()

        metrics = {
            'loss': total_loss,
            'bce_loss': bce_loss.mean().item(),
            'slot_reg_loss': slot_reg.mean().item() if self.adaptive else slot_reg
        }

        aux = {
            'metrics': metrics,
            'outs': outs
        }

        return total_loss, aux