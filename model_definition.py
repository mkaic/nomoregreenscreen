
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
from tqdm import tqdm
from train_utils import SkipConnDownChannel, IdentityBlock


class CoarseMatteGenerator(nn.Module):

	def __init__(self, input_channels, output_channels, chan):
		super().__init__()

		self.upchannel1 = nn.Conv2d(input_channels, chan, kernel_size = 3) 
		self.ident1 = IdentityBlock(chan)

		self.upchannel2 = nn.Conv2d(chan, chan*2, kernel_size = 3) 
		self.ident2 = IdentityBlock(chan*2)

		self.upchannel3 = nn.Conv2d(chan*2, chan*4, kernel_size = 3)
		self.ident3 = IdentityBlock(chan*4)


		self.aspp = nn.Sequential(

			nn.Conv2d(chan*4, chan*4, kernel_size = 3, dilation = 3),
			nn.Conv2d(chan*4, chan*4, kernel_size = 3, dilation = 6),
			nn.Conv2d(chan*4, chan*4, kernel_size = 3, dilation = 9)
			)

		self.activation = nn.ReLU()

		self.downchannel1 = SkipConnDownChannel(chan*4, chan*2)
		self.downchannel2 = SkipConnDownChannel(chan*2, chan)
		self.downchannel3 = SkipConnDownChannel(chan, output_channels, final = True)

		


	def forward(self, input_tensor):

		batch = input_tensor.shape[0]
		channels = input_tensor.shape[1]
		height = input_tensor.shape[-2]
		width = input_tensor.shape[-1]
		shape_half = [height//2, width//2]
		shape_4th = [height//4, width//4]
		shape_8th = [height//8, width//8]
		shape_16th = [height//16, width//16]
		shape_32nd = [height//32, width//32]

		
		X1 = self.upchannel1(input_tensor)
		X1 = self.activation(X1)
		X1 = self.ident1(X1)

		X2 = F.interpolate(X1, size = shape_half)
		X2 = self.upchannel2(X2)
		X2 = self.activation(X2)
		X2 = self.ident2(X2)

		X3 = F.interpolate(X2, size = shape_4th)
		X3 = self.upchannel3(X3)
		X3 = self.activation(X3)
		X4 = self.ident3(X3)

		middle = self.aspp(X4)
		middle = self.activation(middle)

		X5 = self.downchannel1(middle, X3)

		X5 = F.interpolate(X5, size = shape_half)
		X6 = self.downchannel2(X5, X2)

		X6 = F.interpolate(X6, size = [height + 8, width + 8])
		X7 = self.downchannel3(X6, X1)

		#X8 = torch.sigmoid(X8)

		return X7



class RefinePatches(nn.Module):

	def __init__(self, coarse_channels, input_channels, output_channels = 1, chan = 32, patch_size = 8):
		super().__init__()

		self.input_channels = input_channels
		self.coarse_channels = coarse_channels
		self.concat_channels = self.input_channels + self.coarse_channels

		#subtracting one from the input channels here because we aren't using the error map as an input
		self.conv1 = nn.Conv2d(self.concat_channels, chan, kernel_size = 3)
		self.conv2 = nn.Conv2d(chan, chan*2, kernel_size = 3)
		self.conv3 = nn.Conv2d(chan*2 + self.input_channels, chan, kernel_size = 3)
		self.conv4 = nn.Conv2d(chan, output_channels, kernel_size = 3)

		self.activation = nn.ReLU()


	def forward(self, start_patches, middle_patches):

		z1 = self.conv1(start_patches)
		x1 = self.activation(z1)

		z2 = self.conv2(x1)
		x2 = self.activation(z2)
		x2 = F.interpolate(x2, size = [middle_patches.shape[-2], middle_patches.shape[-1]])

		z3 = torch.cat([x2, middle_patches], 1)
		z3 = self.conv3(z3)
		x3 = self.activation(z3)

		z4 = self.conv4(x3)

		end_patches = torch.sigmoid(z4)

		return end_patches

