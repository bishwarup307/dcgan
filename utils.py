import glob
import os
import shutil

import numpy as np
import torch


def gen_noise(batch_size: int, noise_dim: int, device: str):
    return torch.randn(batch_size, noise_dim, device=device)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class CutTop:
    def __init__(self, cut_top=40):
        self.cut_top = cut_top

    def __call__(self, img):
        img = np.array(img)
        img = img[40:, ...]
        return img


def get_grad(
    crit: torch.nn.Module, real: torch.Tensor, fake: torch.Tensor, epsilon: torch.Tensor
):
    mixed_images = epsilon * real + (1.0 - epsilon) * fake
    mixed_scores = crit(mixed_images)
    grads = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grads


def gradient_penalty(grads):
    grads = grads.reshape(len(grads), -1)
    norms = grads.norm(2, dim=1)
    gp = torch.mean((norms - 1.0) ** 2)
    return gp


def save_checkpoint(model: torch.nn.Module, path: str):
    """
    saves checkpoint for the model
    Args:
        model (nn.Module): a pytorch model object
        path (str): path to save the checkpoint

    Returns:
    """
    torch.save(model.state_dict(), path)


class ModelCheckpoint:
    def __init__(self, path: str, freq: int, keep_n: int = -1):
        self.path = path
        self.freq = freq
        self.keep_n = keep_n
        self._make_path()

    def _make_path(self):
        os.makedirs(self.path, exist_ok=True)

    def _get_all_checkpoints(self):
        return sorted(glob.glob(self.path + os.sep + ".pth"))

    def _delete_prev_checkpoints(self):
        prev_ckpt = self._get_all_checkpoints()
        if self.keep_n > 0:
            prev_ckpt = prev_ckpt[: -self.keep_n]
        for ckpt in prev_ckpt:
            shutil.rmtree(ckpt)

    def save(self, model: torch.nn.Module, step: int):
        if step % self.freq == 0:
            self._delete_prev_checkpoints()
            torch.save(
                model.state_dict(), os.path.join(self.path, f"gen_epoch_{step}.pth")
            )
