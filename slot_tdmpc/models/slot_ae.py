import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange
from disentangle.latents.slot_attention import SlotAttention, AdaptiveSlotWrapper
from typing import Tuple, List, Dict, Any


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


class SlotEncoder(nn.Module):
    def __init__(self, resolution: Tuple[int, int], hid_dim: int, slot_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, hid_dim, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(hid_dim, hid_dim, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(hid_dim, hid_dim, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(hid_dim, hid_dim, 5, padding=2), nn.ReLU(inplace=True),
        )
        self.encoder_pos = SoftPositionEmbed(input_dim=4, hidden_size=hid_dim, resolution=resolution)

        self.dense = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=slot_dim)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x


class SlotDecoder(nn.Module):
    def __init__(self, hid_dim: int, resolution: Tuple[int, int] = (8, 8)):
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

    def forward(self, z_hat: torch.Tensor):
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


class SoftPositionEmbed(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, resolution: Tuple[int, int]):
        super(SoftPositionEmbed, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        self.grid = torch.from_numpy(build_grid(resolution))
        self.grid = self.grid.requires_grad_(False).cuda()

    def forward(self, inputs: torch.Tensor):
        return inputs + self.linear(self.grid)


class SlotAutoEncoder(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], num_slots: int, slot_dim: int, adaptive: bool, lambdas: Dict[str, Any]):
        super(SlotAutoEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.num_slots = num_slots
        self.latent_dim = slot_dim
        self.encoder = SlotEncoder(resolution=(64, 64), hid_dim=64, slot_dim=slot_dim)
        self.decoder = SlotDecoder(hid_dim=64)
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