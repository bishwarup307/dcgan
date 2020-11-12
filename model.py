from typing import Optional

import tocrh
import torch.nn as nn
import numpy as np

def Generator(nn.Module):
	def __init__(self, latent_dim:int = 100, hidden_dim: int = 64, 
		im_ch:int = 1, upsample_mode: Optional[str] = None, use_batchnorm: bool = True):
		super(Generator, self).__init__()
		if upsample mode is not None:
			if upsample_mode not in ('nearest', 'bilinear'):
				raise ValueError(f"expected `upsample_mode` to be one of `nearest` or `bilinear, got {upsample_mode}")
		self.upsample_mode = upsample_mode
		self.use_batchnorm = use_batchnorm
		self.gen = nn.Sequential(
				_make_block(1, hidden_dim * 8, first_block = True),
				_make_block(hidden_dim * 8, hidden_dim * 4)
				_make_block(hidden_dim * 4, hidden_dim * 2)
				_make_block(hidden_dim * 2, hidden_dim)
			)

	def _make_block(self, in_ch:int, out_ch: int, first_block:bool = False, last_block:bool = False):
		layers = []
		if self.upsample_mode is not None:
			scale_factor = 4 if first_block else 2
			if self.upsample_mode == 'nearest':
				layers.append(nn.UpsamplingNearest2d(scale_factor = scale_factor))
			else:
				layers.append(nn.UpsamplingBilinear2d(scale_factor = scale_factor))
			layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
		else:
			padding = 0 if first_block else 1
			layer.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size = 4, stride = 2, padding = padding))
		if self.use_batchnorm and not last_block:
			layers.append(nn.BatchNorm2d(out_ch))
		layers.append(nn.ReLU())
		return nn.Sequential(*layers)

	def forward(self, x):
		x = x.unsqueeze(-1).unsqueeze(-1)
		return self.gen(x)

def Critic(nn.Module):
	def __init__(self, hidden_dim: int = 64):
		super(Critic, self).__init__()

	def _make_block(self):
		return nn.Sequential(

			)