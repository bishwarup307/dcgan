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


# class WLOSS_GP(torch.nn.Module):
#     def __init__(self):
