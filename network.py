import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class Interpolate2D(nn.Module):
	def __init__(self, scale_factor, mode='nearest'):
		super(Interpolate2D, self).__init__()
		self.sf = scale_factor
		self.mode = mode

	def forward(self, x):
		return F.interpolate(x, scale_factor=self.sf, mode=self.mode)

class Flatten(nn.Module):
		def forward(self, input):
			return input.view(input.size(0), -1)

class ResidulBlock(nn.Module):
	def __init__(self, inc, outc, sample='none'):
		super(ResidulBlock, self).__init__()
		self.conv1 = spectral_norm(nn.Conv2d(inc, outc, 3, 1, 1))
		self.conv2 = spectral_norm(nn.Conv2d(outc, outc, 3, 1, 1))
		self.conv_sc = spectral_norm(nn.Conv2d(inc, outc, 1, 1, 0)) if inc != outc else False
		
		if sample == 'up':
			self.sample = Interpolate2D(scale_factor=2)
		else:
			self.sample = None

		self.bn1 = nn.BatchNorm2d(inc)
		self.bn2 = nn.BatchNorm2d(outc)
		
		self.act = nn.LeakyReLU(0.2)
		
	def forward(self, x):
		h = self.act(self.bn1(x))

		if self.sample:
			h = self.sample(h)
			x = self.sample(x)
		
		h = self.conv1(h)
		h = self.act(self.bn2(h))
		h = self.conv2(h)
		
		if self.conv_sc:
			x = self.conv_sc(x)
		return x + h

# generator
class Generator(nn.Module):
	def __init__(self, ch_style, ch_content):
		super(Generator, self).__init__()
		ch_input = ch_style + ch_content
		ch_output = 3
		base_dim = 64
		self.style_encoder = nn.Sequential(
			spectral_norm(nn.Conv2d(ch_style, base_dim * 1, 3, 1, 1)),
			nn.BatchNorm2d(base_dim * 1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.AvgPool2d(2),
			spectral_norm(nn.Conv2d(base_dim * 1, base_dim * 2, 3, 1, 1)),
			nn.BatchNorm2d(base_dim * 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.AvgPool2d(2),
			spectral_norm(nn.Conv2d(base_dim * 2, base_dim * 4, 3, 1, 1)),
			nn.BatchNorm2d(base_dim * 4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.AvgPool2d(2),
		)

		self.content_encoder = nn.Sequential(
			spectral_norm(nn.Conv2d(ch_content, base_dim * 1, 3, 1, 1)),
			nn.BatchNorm2d(base_dim * 1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.AvgPool2d(2),
			spectral_norm(nn.Conv2d(base_dim * 1, base_dim * 2, 3, 1, 1)),
			nn.BatchNorm2d(base_dim * 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.AvgPool2d(2),
			spectral_norm(nn.Conv2d(base_dim * 2, base_dim * 4, 3, 1, 1)),
			nn.BatchNorm2d(base_dim * 4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.AvgPool2d(2),
		)

		self.decoder = nn.Sequential(
			spectral_norm(nn.Conv2d(base_dim * 8, base_dim * 4, 3, 1, 1)),
			ResidulBlock(base_dim * 4, base_dim * 4),
			ResidulBlock(base_dim * 4, base_dim * 4),
			ResidulBlock(base_dim * 4, base_dim * 4),
			ResidulBlock(base_dim * 4, base_dim * 4),
			ResidulBlock(base_dim * 4, base_dim * 2, sample='up'),
			ResidulBlock(base_dim * 2, base_dim * 1, sample='up'),
			ResidulBlock(base_dim * 1, base_dim * 1, sample='up'),
			nn.BatchNorm2d(base_dim * 1),
			nn.LeakyReLU(0.2, inplace=True),
			spectral_norm(nn.Conv2d(base_dim * 1, ch_output, 3, 1, 1)),
			nn.Tanh(),
		)

	def forward(self, style, content):
		style_h = self.style_encoder(style)
		content_h = self.content_encoder(content)
		h = torch.cat([style_h, content_h], dim=1)
		return self.decoder(h)

# discriminator
class Discriminator(nn.Module):
	def __init__(self, ch_input):
		super(Discriminator, self).__init__()
		base_dim = 64
		self.net = nn.Sequential(
			spectral_norm(nn.Conv2d(ch_input, base_dim * 1, 3, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),
			spectral_norm(nn.Conv2d(base_dim * 1, base_dim * 2, 3, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),
			spectral_norm(nn.Conv2d(base_dim * 2, base_dim * 4, 3, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),
			spectral_norm(nn.Conv2d(base_dim * 4, base_dim * 8, 3, 1, 1)),
			nn.LeakyReLU(0.2, inplace=True),
			spectral_norm(nn.Conv2d(base_dim * 8, 1, 3, 1, 1)),
		)

	def forward(self, x):
		return self.net(x)
