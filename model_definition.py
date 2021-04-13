
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
		X = self.activation(X)

		X = torch.add(X, skipped_X)

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



class MatteRefinementNetwork(nn.Module):

	def __init__(self, coarse_channels, input_channels, output_channels = 1, chan = 32, patch_size = 8, device = 'cpu'):
		super().__init__()

		self.patch_size = patch_size
		self.device = device

		self.input_channels = input_channels
		self.coarse_channels = coarse_channels
		self.concat_channels = self.input_channels + self.coarse_channels

		#subtracting one from the input channels here because we aren't using the error map as an input
		self.conv1 = nn.Conv2d(self.concat_channels - 1, chan, kernel_size = 3)
		self.conv2 = nn.Conv2d(chan, chan*2, kernel_size = 3)
		self.conv3 = nn.Conv2d(chan*2 + self.input_channels, chan, kernel_size = 3)
		self.conv4 = nn.Conv2d(chan, output_channels, kernel_size = 3)

		self.activation = nn.ReLU()

	#This is a simple utility function for grabbing a square patch of an image tensor of dimensions N x C x H x W.
	def get_image_patch(self, image, patch_size, top_left_corner):

		patch = image[:, :, top_left_corner[0] : top_left_corner[0] + patch_size, top_left_corner[1] : top_left_corner[1] + patch_size]
		return patch

	def replace_image_patch(self, image, patch, top_left_corner):

		size = patch.shape[-1]
		image[:, :, top_left_corner[0]:top_left_corner[0]+size, top_left_corner[1]:top_left_corner[1]+size] = patch

		return image

	def forward(self, fake_coarse_alpha, fake_coarse_error, fake_coarse_hidden, input_tensor):

		super_upsampled_coarse_alpha = F.interpolate(fake_coarse_alpha, [input_tensor.shape[-2], input_tensor.shape[-1]])


		#We need a downsampled version of the input for the refinement to take patches out of.
		downsampled_input = F.interpolate(input_tensor, size = [input_tensor.shape[-2]//2, input_tensor.shape[-1]//2])
		#We need to *upsample* the coarse outputs tho.
		upsampled_coarse_alpha = F.interpolate(fake_coarse_alpha, size = [input_tensor.shape[-2]//2, input_tensor.shape[-1]//2])
		upsampled_coarse_hidden = F.interpolate(fake_coarse_hidden, size = [input_tensor.shape[-2]//2, input_tensor.shape[-1]//2])

		upsampled_coarse = torch.cat([upsampled_coarse_alpha, upsampled_coarse_hidden], 1)

		patches = torch.zeros(fake_coarse_alpha.shape[0], self.concat_channels, 8, 8)

		#This part has to be done without batches for simplicity's sake.
		patches = torch.zeros(fake_coarse_error.shape[0] * K)
		for idx in range(fake_coarse_error.shape[0]):

			current_example_to_refine = super_upsampled_coarse_alpha[idx:idx+1, :, :, :]
			values, indices = torch.topk(fake_coarse_error[idx:idx+1, :, :, :].flatten(), k = self.K)
			#print(fake_coarse_error[idx:idx+1, :, :, :].flatten().shape)
			#print(indices.shape)
			low_confidence = np.array(np.unravel_index(indices.cpu().numpy(), fake_coarse_error[idx:idx+1, :, :, :].shape)).T
			print(low_confidence.shape)



			for index in low_confidence:

				top_left_corner = torch.clamp(torch.tensor([(index[-2] * 2) - 4, (index[-1] * 2) - 4]), min = 0, max = upsampled_coarse.shape[-1] - 8).to(self.device)
				#print(top_left_corner)

				coarse_patch = self.get_image_patch(upsampled_coarse[idx:idx+1, :, :, :], 8, top_left_corner)
				#print(coarse_patch.shape)
				input_patch = self.get_image_patch(downsampled_input[idx:idx+1, :, :, :], 8, top_left_corner)
				#print(input_patch.shape)
				start_patch = torch.cat([coarse_patch, input_patch], 1)
				#print(start_patch.shape)



				x1 = self.conv1(start_patch)
				x1 = self.activation(x1)

				x2 = self.conv2(x1)
				x2 = self.activation(x2)
				x2 = F.interpolate(x2, [8,8])

				top_left_corner = [top_left_corner[0] * 2, top_left_corner[1] * 2]

				middle_input_patch = self.get_image_patch(input_tensor[idx:idx+1, :, :, :], 8, top_left_corner)
				middle_concat = torch.cat([x2, middle_input_patch], 1)
				#print(middle_concat.shape)

				x3 = self.conv3(middle_concat)
				x3 = self.activation(x3)
				final_patch = self.conv4(x3)
				#final_patch = torch.sigmoid(final_patch)
				
				top_left_corner = [top_left_corner[0] + 2, top_left_corner[1] + 2]
				current_example_to_refine = self.replace_image_patch(current_example_to_refine, final_patch,top_left_corner)

			super_upsampled_coarse_alpha[idx] = current_example_to_refine

		return super_upsampled_coarse_alpha


"""
class AlphaMatteGenerator(nn.Module):

	def __init__(self):
		super().__init__()

		self.coarse = CoarseMatteGenerator(input_channels = input_channels, output_channels = 2, chan = 64).train().to(device)
		self.refine = MatteRefinementNetwork()

	def forward(self, input_tensor):

		
		real_coarse_alpha = F.interpolate(real_alpha, size = [real_alpha.shape[-2]//4, real_alpha.shape[-1]//4])

		
		fake = coarse(input_tensor)
		fake_coarse_alpha = fake[:,0:1,:,:]
		fake_coarse_error = fake[:,1:2,:,:]
		fake_coarse_hidden = fake[:,2:,:,:]

		real_coarse_error = torch.square(real_coarse_alpha.detach()-fake_coarse_alpha.detach())
		
		fake_refined_alpha = refine(fake_coarse_alpha, fake_coarse_error, input_tensor)

		coarse_opt = torch.optim.Adam(coarse.parameters(), lr = learning_rate)
		refine_opt = torch.optim.Adam(refine.parameters(), lr = learning_rate)

		coarse_loss = bce_loss(fake_coarse_alpha, real_coarse_alpha) + bce_loss(fake_coarse_error, real_coarse_error)
		refine_loss = bce_loss(fake_refined_alpha, real_alpha)

		total_loss = coarse_loss + refine_loss

		coarse_opt.zero_grad()
		total_loss.backward()
		coarse_opt.step()

		coarse_opt.zero_grad()
		total_loss.backward()
		coarse_opt.step()
		

		return
"""