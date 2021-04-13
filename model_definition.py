
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import time
from tqdm import tqdm

class IdentityBlock(nn.Module):
	#Inputs a tensor, convolves/activates it twice, then adds it to the original input version of itself (residual block)

	def __init__(self, channels):
		super().__init__()

		self.activation = nn.LeakyReLU(0.1)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size = 1)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)

	def forward(self, X):

		skipped_X = X

		X = self.conv1(X)
		X = self.activation(X)
		X = self.conv2(X)
	
		X = torch.add(X, skipped_X)
		X = self.activation(X)

		return X


class SkipConnDownChannel(nn.Module):

	def __init__(self, input_channels, output_channels):
		super().__init__()

		self.input_channels = input_channels
		self.activation = nn.LeakyReLU(0.1)

		self.input_down_channel = nn.Conv2d(input_channels, input_channels//2, kernel_size = 1)
		self.skip_down_channel = nn.Conv2d(input_channels, input_channels//2, kernel_size = 1)

		self.concatenated_down_channel = nn.Conv2d(input_channels, input_channels//2, kernel_size = 3, padding = 1)
		self.concatenated_conv = nn.Conv2d(input_channels//2, output_channels, kernel_size = 3, padding = 1)


	def forward(self, X, skipped_X):

		X = self.input_down_channel(X)
		X = self.activation(X)

		skipped_X = self.skip_down_channel(skipped_X)
		skipped_X = self.activation(skipped_X)

		concatenated = torch.cat([X, skipped_X], 1)

		concatenated = self.concatenated_down_channel(concatenated)
		concatenated = self.activation(concatenated)
		concatenated = self.concatenated_conv(concatenated)
		concatenated = self.activation(concatenated)

		return concatenated


class CoarseMatteGenerator(nn.Module):

	def __init__(self, input_channels, output_channels, chan):
		super().__init__()

		self.upchannel1 = nn.Conv2d(input_channels, chan, kernel_size = 3, padding = 1) 
		self.ident1 = IdentityBlock(chan)

		self.upchannel2 = nn.Conv2d(chan, chan*2, kernel_size = 3, padding = 1) 
		self.ident2 = IdentityBlock(chan*2)

		self.upchannel3 = nn.Conv2d(chan*2, chan*4, kernel_size = 3, padding = 1)
		self.ident3 = IdentityBlock(chan*4)

		self.upchannel4 = nn.Conv2d(chan*4, chan*8, kernel_size = 3, padding = 1)
		self.ident4 = IdentityBlock(chan*8)

		self.middle_conv = nn.Conv2d(chan*8, chan*8, kernel_size = 1)
		self.activation = nn.ReLU()

		self.downchannel1 = SkipConnDownChannel(chan*8, chan*4)
		self.downchannel2 = SkipConnDownChannel(chan*4, chan*2)
		self.downchannel3 = SkipConnDownChannel(chan*2, chan)
		self.downchannel4 = SkipConnDownChannel(chan, output_channels)

		


	def forward(self, input_tensor):

		batch = input_tensor.shape[0]
		channels = input_tensor.shape[1]
		height = input_tensor.shape[-2]
		width = input_tensor.shape[-1]
		half_shape = [height//2, width//2]
		quarter_shape = [height//4, width//4]
		eighth_shape = [height//8, width//8]
		sixteenth_shape = [height//16, width//16]
		thirty_second_shape = [height//32, width//32]

		X = F.interpolate(input_tensor, size = quarter_shape)

		X1 = self.upchannel1(X)
		X1 = self.activation(X1)
		X1 = self.ident1(X1)

		X2 = F.interpolate(X1, size = eighth_shape)
		X2 = self.upchannel2(X2)
		X2 = self.activation(X2)
		X2 = self.ident2(X2)

		X3 = F.interpolate(X2, size = sixteenth_shape)
		X3 = self.upchannel3(X3)
		X3 = self.activation(X3)
		X3 = self.ident3(X3)

		X4 = F.interpolate(X3, size = thirty_second_shape)
		X4 = self.upchannel4(X4)
		X4 = self.activation(X4)
		X4 = self.ident4(X4)

		middle = self.middle_conv(X4)
		middle = self.activation(middle)

		X5 = self.downchannel1(middle, X4)
		X5 = F.interpolate(X5, size = sixteenth_shape)

		X6 = self.downchannel2(X5, X3)
		X6 = F.interpolate(X6, size = eighth_shape)


		X7 = self.downchannel3(X6, X2)
		X7 = F.interpolate(X7, size = quarter_shape)


		X8 = self.downchannel4(X7, X1)
		#X8 = torch.sigmoid(X8)

		return X8



class RefinePatches(nn.Module):

	def __init__(self, coarse_channels, input_channels, output_channels = 1, chan = 32, patch_size = 8):
		super().__init__()

		self.input_channels = input_channels
		self.coarse_channels = coarse_channels
		self.concat_channels = self.input_channels + self.coarse_channels

		#subtracting one from the input channels here because we aren't using the error map as an input
		self.conv1 = nn.Conv2d(self.concat_channels - 1, chan, kernel_size = 3)
		self.conv2 = nn.Conv2d(chan, chan*2, kernel_size = 3)
		self.conv3 = nn.Conv2d(chan*2 + self.input_channels, chan, kernel_size = 3)
		self.conv4 = nn.Conv2d(chan, output_channels, kernel_size = 3)

		self.activation = nn.ReLU()


	def forward(self, start_patches, middle_patches):

		z1 = self.conv1(start_patches)
		x1 = self.activation(z1)

		z2 = self.conv2(x1)
		x2 = self.activation(z2)

		z3 = torch.cat([x2, middle_patches], 1)
		z3 = self.conv3(z3)
		x3 = self.activation(z3)

		z4 = self.conv4(x3)

		end_patches = z4

		return end_patches

