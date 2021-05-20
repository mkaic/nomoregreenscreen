import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm

class RefinementNetwork(nn.Module):

	def __init__(self, coarse_channels = 37, input_channels = 6, patch_size = 8):
		super().__init__()

		self.input_channels = input_channels
		self.coarse_channels = coarse_channels
		self.concat_channels = self.input_channels + self.coarse_channels

		#subtracting one from the input channels here because we aren't using the error map as an input
		self.conv1 = nn.Conv2d(self.concat_channels, 24, kernel_size = 3)
		self.conv2 = nn.Conv2d(24, 16, kernel_size = 3)
		self.conv3 = nn.Conv2d(16 + self.input_channels, 12, kernel_size = 3)
		self.conv4 = nn.Conv2d(12, 4, kernel_size = 3)

		self.bn1 = nn.BatchNorm2d(24)
		self.bn2 = nn.BatchNorm2d(16)
		self.bn3 = nn.BatchNorm2d(12)

		self.activation = nn.ReLU()


	def forward(self, start_patches, middle_patches):

		z1 = self.conv1(start_patches)
		z1 = self.bn1(z1)
		x1 = self.activation(z1)

		z2 = self.conv2(x1)
		z2 = self.bn2(z2)
		x2 = self.activation(z2)
		x2 = F.interpolate(x2, size = middle_patches.shape[-2:])

		z3 = torch.cat([x2, middle_patches], 1)
		z3 = self.conv3(z3)
		z3 = self.bn3(z3)
		x3 = self.activation(z3)

		z4 = self.conv4(x3)

		return z4