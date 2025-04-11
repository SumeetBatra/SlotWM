import einops
import torch
import wandb
import torch.nn.functional as F


def peak_signal_to_noise_ratio(x_hat_logits, x_true):
    mse = torch.mean(torch.square(F.sigmoid(x_hat_logits) - x_true))
    max_value = torch.Tensor([1.]).to(mse.device)
    return 20 * torch.log10(max_value) - 10 * torch.log10(mse)


def log_reconstruction_metrics(aux, step, use_wandb: bool = False):
    num_samples = 16
    true = einops.rearrange(aux['x_true'][:num_samples], 'b c h w -> h (b w) c')
    predicted = einops.rearrange(F.sigmoid(aux['x_hat_logits'][:num_samples]), 'b c h w -> h (b w) c')
    absolute_diff = torch.abs(true - predicted)
    image = torch.cat((true, predicted, absolute_diff), dim=0)
    psnr = peak_signal_to_noise_ratio(aux['x_hat_logits'], aux['x_true'])

    if use_wandb:
        wandb.log({
            'reconstructions': wandb.Image(image.detach().cpu().numpy()),
            'ae/psnr': psnr.mean().item()
        }, step=step)