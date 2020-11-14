from typing import Optional

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        hidden_dim: int = 64,
        im_ch: int = 3,
        upsample_mode: Optional[str] = None,
        use_batchnorm: bool = True,
    ):
        super(Generator, self).__init__()
        if upsample_mode is not None:
            if upsample_mode not in ("nearest", "bilinear"):
                raise ValueError(
                    f"expected `upsample_mode` to be one of `nearest` or `bilinear, got `{upsample_mode}`"
                )
        self.upsample_mode = upsample_mode
        self.use_batchnorm = use_batchnorm
        self.gen = nn.Sequential(
            self._make_block(latent_dim, hidden_dim * 8, first_block=True),
            self._make_block(hidden_dim * 8, hidden_dim * 4),
            self._make_block(hidden_dim * 4, hidden_dim * 2),
            self._make_block(hidden_dim * 2, hidden_dim),
            self._make_block(hidden_dim, im_ch, last_block=True),
        )

    def _make_block(
        self,
        in_ch: int,
        out_ch: int,
        first_block: bool = False,
        last_block: bool = False,
    ):
        layers = []
        activation = nn.Tanh() if last_block else nn.ReLU()
        bias = not self.use_batchnorm
        if self.upsample_mode is not None:
            scale_factor = 4 if first_block else 2
            if self.upsample_mode == "nearest":
                layers.append(nn.UpsamplingNearest2d(scale_factor=scale_factor))
            else:
                layers.append(nn.UpsamplingBilinear2d(scale_factor=scale_factor))
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)
            )
        else:
            padding = 0 if first_block else 1
            layers.append(
                nn.ConvTranspose2d(
                    in_ch, out_ch, kernel_size=4, stride=2, padding=padding, bias=bias
                )
            )
        if self.use_batchnorm and not last_block:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(activation)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.gen(x)


class Critic(nn.Module):
    def __init__(
        self, im_ch: int = 3, hidden_dim: int = 64, use_batchnorm: bool = True
    ):
        super(Critic, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.crit = nn.Sequential(
            self._make_block(im_ch, hidden_dim),
            self._make_block(hidden_dim, hidden_dim * 2),
            self._make_block(hidden_dim * 2, hidden_dim * 4),
            self._make_block(hidden_dim * 4, hidden_dim * 8),
            self._make_block(hidden_dim * 8, 1, last_block=True),
        )

    def _make_block(self, in_ch: int, out_ch: int, last_block: bool = False):
        layers = []
        padding = 0 if last_block else 1
        bias = not self.use_batchnorm
        layers.append(
            nn.Conv2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=padding, bias=bias
            )
        )
        if self.use_batchnorm and not last_block:
            layers.append(nn.BatchNorm2d(out_ch))
        if not last_block:
            layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.crit(x).reshape(-1)


def init_weights(m):
    # print(m)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    else:
        pass


if __name__ == "__main__":
    t = torch.randn(4, 100)
    gen = Generator(im_ch=1, upsample_mode="bilinear")
    output = gen(t)
    print(output.size())
    # critic = Critic()
    # cr_output = critic(output)
    # print(cr_output.size())
    # print(gen)


# def Critic(nn.Module):
# 	def __init__(self, hidden_dim: int = 64):
# 		super(Critic, self).__init__()
#
# 	def _make_block(self):
# 		return nn.Sequential(
#
# 			)
