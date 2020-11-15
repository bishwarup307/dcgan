import glob
import os
from typing import Optional, Union, Any

import numpy as np
import scipy.linalg
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader


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
        img = img[self.cut_top :, ...]
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
        return sorted(glob.glob(os.path.join(self.path, "*.pth")))

    def _delete_prev_checkpoints(self):
        prev_ckpt = self._get_all_checkpoints()
        if self.keep_n > 0:
            prev_ckpt = prev_ckpt[: -self.keep_n]
        for ckpt in prev_ckpt:
            os.remove(ckpt)

    def save(self, model: torch.nn.Module, step: int):
        if step % self.freq == 0:
            self._delete_prev_checkpoints()
            torch.save(
                model.state_dict(), os.path.join(self.path, f"gen_epoch_{step}.pth")
            )


def preprocess_inception(img):
    img = torch.nn.functional.interpolate(
        img, size=(299, 299), mode="bilinear", align_corners=False
    )
    return img


class FeatureMap:
    def __init__(self):
        self.output = None

    def __call__(self, module, module_in, module_out):
        self.output = module_out.squeeze()

    def clear(self):
        self.output = []


def load_inception_model(
    pretrained: Optional[bool] = True,
    weights: Optional[str] = None,
    device: str = "cpu",
    # top: bool = False,
    # hooks: Optional[List] = []
):
    if not pretrained and weights is None:
        raise ValueError(f"`weights_path` must be provided when `pretrained == False`")
    inception = torchvision.models.inception_v3(
        pretrained=pretrained, init_weights=False
    )
    if not pretrained:
        inception.load_state_dict(torch.load(weights))
    # if not top:
    #     inception.fc = torch.nn.Identity()

    # if hooks:
    #     for hook in hooks:
    #         inception.
    inception = inception.to(device)
    inception.eval()
    return inception


def get_inception_features(
    model: torch.nn.Module,
    batch: torch.Tensor,
    batch_size: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    hook: Optional[Any] = None,
):
    model = model.to(device)
    model.eval()
    batch = preprocess_inception(batch)
    if batch_size is not None:
        features = []
        int_features = []

        assert batch_size > 0, "invalid batch size"
        ds = TensorDataset(batch)
        loader = DataLoader(
            ds, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False
        )
        for _, sample in enumerate(loader):
            feat = model(sample[0].to(device))
            features.append(feat.detach().cpu())
            if hook is not None:
                int_features.append(hook.output.detach().cpu())

        features = torch.cat(features, dim=0)
        if hook is not None:
            int_features = torch.cat(int_features, dim=0)
    else:
        features = model(batch.to(device))
        if hook is not None:
            int_features = hook.output.detach().cpu()
        features = features.detach().cpu()

    return features, int_features


def get_covariance(features: torch.Tensor):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))


def matrix_sqrt(x):
    """
    Function that takes in a matrix and returns the square root of that matrix.
    For an input matrix A, the output matrix B would be such that B @ B is the matrix A.
    Parameters:
        x: a matrix
    """
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real, device=x.device)


def fid(
    mu_x: torch.Tensor, mu_y: torch.Tensor, sigma_x: torch.Tensor, sigma_y: torch.Tensor
):
    return (mu_x - mu_y).norm(2) + torch.trace(
        sigma_x + sigma_y - 2 * matrix_sqrt(sigma_x @ sigma_y)
    )


def inception_score(batch: torch.Tensor, eps=1e-6):
    py = batch.mean(dim=0)
    kld = batch * (torch.log(batch + eps) - torch.log(py + eps))
    score = kld.sum(dim=1).mean(dim=0).exp()
    return score


def save_tensor(t: torch.Tensor, name: str, save_dir: str = "./"):
    t = t.detach().cpu().numpy()
    np.save(os.path.join(save_dir, name), t)


def load_tensor(name: str, save_dir: str = "./"):
    path = os.path.join(save_dir, f"{name}.npy")
    try:
        t = np.load(path)
        return torch.from_numpy(t).float()
    except FileNotFoundError:
        return None


# def fid(
#     batch_1: torch.Tensor,
#     batch_2: torch.Tensor,
#     cache_dir: Optional[str] = None,
#     pretrained: Optional[bool] = False,
#     weights_path: Optional[str] = None,
#     device: Optional[str] = "cpu",
#     batch_size: Optional[int] = 16,
# ):
#     if not pretrained and weights_path is None:
#         raise ValueError(f"`weights_path` must be provided when `pretrained == False`")
#
#     if cache_dir:
#         os.makedirs(cache_dir, exist_ok=True)
#
#     device = torch.device(device)
#
#     inception = torchvision.models.inception_v3(
#         pretrained=pretrained, init_weights=False
#     )
#     if not pretrained:
#         inception.load_state_dict(torch.load(weights_path))
#     inception = inception.to(device)
#     inception.eval()
#     inception.fc = torch.nn.Identity()
#
#     features_batch_1, features_batch_2 = [], []


if __name__ == "__main__":
    checkpointer = ModelCheckpoint(
        "/home/bishwarup/GAN_experiments/dcgan/2ab2da", freq=1, keep_n=10
    )
    print(checkpointer._get_all_checkpoints()[:-10])
