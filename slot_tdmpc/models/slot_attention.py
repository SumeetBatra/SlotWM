import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init


# Based off of https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention.py

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min=eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))


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

