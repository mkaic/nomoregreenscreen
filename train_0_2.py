
print('Loading Libraries...')
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
from torchsummary import summary

from matte_dataset import MatteDataset
from model_definition import *
device = 'cuda'

#PSEUDOCODE

#Given a directory full of directories full of HD short clips rendered out as PNG sequences with alpha channels
#Given a directory full of directories full of HD short clips rendered out as PNG sequences of different possible backgrounds
#Given a defined model architecture with any transfer learning set up in the __init__() function
#Given optimizer(s) for the model



#LOOP some number of training examples.

	#Generate INPUT FRAME PACKET using Dataset and DataLoader
	
	#ZERO GRADIENTS.
	#FEED FORWARD the input frame packet into the MODEL.
	#COMPUTE LOSS(ES) using ALPHA FRAME PACKET and model OUTPUTS, with the accuracy of the middle (target) frame weighted much more than those surrounding it.
	#COMPUTE GRADIENT(S)
	#STEP OPTIMIZERS ONCE


#Once training loop has completed, SAVE STATE DICT of model to drive.

print('Initializing Dataset...')
params = {
	
	'bg_dir':'train_set/backgrounds/',
	'fg_dir':'train_set/foregrounds/',
	'comp_context_depth': 1,
	'comp_context_stride': 2,
	'bprime_context_depth': 1,
	'bprime_context_stride': 2

}

batch_size = 2

#Initialize the dataset and the loader to feed the data to the network.
dataset = MatteDataset(**params)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)


print('Initializing Network...')
#How many channels the network should expect (this has to be calculated because we feed them as one big mega-channeled image block to the network)
input_channels = ((params['comp_context_depth'] * 2 + 1) + (params['bprime_context_depth'] * 2 + 1)) * 3
num_hidden_channels = 8

#Initialize the network which will produce a coarse alpha (1 chan), confidence map (1 chan), and a number of hidden channels (num_hidden_channels chan)...
coarse = CoarseMatteGenerator(input_channels = input_channels, output_channels = num_hidden_channels + 2, chan = 32).train().to(device)
#... which we will then feed to the refinement network to upsample and refine the area of least confidence.
refine = RefinePatches(input_channels = input_channels, coarse_channels = num_hidden_channels + 2, output_channels = 1, chan = 32).train().to(device)

learning_rate = 0.00001

bce_loss = nn.BCEWithLogitsLoss()

#This is a simple utility function for grabbing a square patch of an image tensor of dimensions N x C x H x W.
def get_image_patches(images, error_maps, k, patch_size = 8):

	#store the original shape
	b, c, h, w = error_maps.shape
	#flatten (except batch dim) the error map (this is so that topk works right)
	err = error_maps.view(b, -1)
	#find the highest k values in the error map for each.
	indices = err.topk(k, dim = 1, sorted = False).indices
	print("indices shape", indices.shape)
	#now we make a tensor of zeros shaped like the flattened error maps
	ref = torch.zeros_like(err)
	#we make the values at the indices of all the top k be 1 so that we can recover those indices easily later.
	ref.scatter_(1, indices, 1.)
	print("ref shape", ref.shape)
	#we reshape the 0s and 1s reference tensor back into the original shape of the error maps.
	ref = ref.view(b,1,h,w)
	print("reshaped ref", ref.shape)
	#we find where the ones are, these 
	low_confidence_idx = torch.nonzero(ref.squeeze(1))
	print("low confidence shape", low_confidence_idx.shape)

	#we get separate lists now for the Batch coord, Y coord, and X coord of each low-confidence pixel. Total of batch_size * k in each of those lists
	#importantly, B[###] lines up with Y[###] and X[###], so that's nice. This way we can to 
	B, Y, X = low_confidence_idx[:,0], low_confidence_idx[:,1], low_confidence_idx[:,2]
	print("B shape", B.shape)

	#now that we have the indices of the patches, we need to actually make the patches using .unfold()
	#first, the channels dimension needs to be the last one because... reasons? I'm really not sure tbh why this is necessary.
	permuted = images.permute(0,2,3,1)
	unfolded = permuted.unfold(1, 8, 4).unfold(2, 8, 4)
	patches = unfolded[B, Y, X]

	return patches


lol = get_image_patches(images = torch.rand(4,20,480,640, device = device), error_maps = torch.rand(4,1,480,640, device = device), k = 10000)
print(lol.shape)

def replace_image_patches(images, patches, indices):

	size = patches.shape[-1]
	image[:, :, top_left_corner[0]:top_left_corner[0]+size, top_left_corner[1]:top_left_corner[1]+size] = patch

	return image

#track how many batches have been done for things like periodic outputs and eventually scheduling.
iteration = 0
schedule1 = 2

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled = use_amp)

print('\nTraining...')
for epoch in range(20):
	print(f'Epoch: {epoch}')

	#Get an example from the dataset.
	for input_tensor, real_alpha in loader:

		with torch.cuda.amp.autocast(enabled = use_amp):

			input_tensor = input_tensor.to(device)

			#Generate a fake coarse alpha, along with a guessed error map and some hidden channel data. Oh yeah and the foreground residual
			fake_coarse = coarse(input_tensor)
			fake_coarse_alpha = fake_coarse[:,0:1,:,:]
			fake_coarse_error = fake_coarse[:,1:2,:,:]
			fake_coarse_foreground_residual = fake_coarse[:,2:5,:,:]
			fake_coarse_hidden_channels = torch.relu(fake_coarse[:,5:,:,:])

			real_alpha = real_alpha.to(device)
			#Get a downsampled version of the alpha for grading the coarse network on
			real_coarse_alpha = F.interpolate(real_alpha, size = [real_alpha.shape[-2]//4, real_alpha.shape[-1]//4])
			#The real error map is calculated as the squared difference between the real alpha and the fake alpha.
			real_coarse_error = torch.square(real_coarse_alpha.detach()-fake_coarse_alpha.detach())

			#The loss of the coarse network is the pixel difference between the real and fake coarse alphas and error maps added together.
			coarse_loss = bce_loss(fake_coarse_alpha, real_coarse_alpha) + bce_loss(fake_coarse_error, real_coarse_error)

			#if it's before the training cutoff, then the loss is just for the coarse network.
			loss = coarse_loss
			coarse_opt = torch.optim.Adam(coarse.parameters(), lr = learning_rate)
			coarse_opt.zero_grad()
				

		if(iteration > schedule1):

			downsampled_input_tensor = F.interpolate(input_tensor, [input_tensor.shape[-2]//2, input_tensor.shape[-1]//2])
			start_patch_source = torch.cat([fake_coarse_alpha, fake_coarse_foreground_residual, fake_coarse_hidden_channels, downsampled_input_tensor], 1)

			start_patches = get_image_patches(start_patch_source, fake_coarse_error, 10000)
			middle_patches = get_image_patches(input_tensor, fake_coarse_error)

			#Now, feed the outputs of the coarse generator into the refinement network, which will refine patches.
			fake_refined_patches = refine(start_patches, middle_patches)

			#The loss of the coarse network is the pixel difference between the real and fake coarse alphas and error maps added together.
			coarse_loss = bce_loss(fake_coarse_alpha, real_coarse_alpha) + bce_loss(fake_coarse_error, real_coarse_error)
			#The loss of the refinement network is just the pixel difference between what it made and what it was supposed to make.
			refine_loss = bce_loss(fake_refined_alpha, real_alpha)

			#if it's after the training cutoff, train both modules concurrently
			loss = coarse_loss + refine_loss
			refine_opt = torch.optim.Adam(refine.parameters(), lr = learning_rate)
			refine_opt.zero_grad()


		scaler.scale(loss).backward()

		scaler.step(coarse_opt)

		if(iteration > schedule1):

			scaler.step(refine_opt)

		scaler.update()
		

		#For keeping track of the outputs so I can look through them to see the network is working right.
		iteration += 1

		if(iteration % 100 == 0 and iteration > schedule1):
			image = torch.sigmoid(fake_coarse_alpha[0,:, :, :].clone()).to('cpu')
			image = transforms.ToPILImage()(image)
			image.save(f'outputs6/{iteration}fake_coarse_alpha.jpg')

			image = torch.sigmoid(fake_refined_alpha[0,:,:,:].clone()).to('cpu')
			image = transforms.ToPILImage()(image)
			image.save(f'outputs6/{iteration}fake_refined_alpha.jpg')

			image = torch.sigmoid(fake_coarse_error[0,:, :, :].clone()).to('cpu')
			image = transforms.ToPILImage()(image)
			image.save(f'outputs6/{iteration}fake_coarse_error.jpg')

			image = real_alpha[0]
			image = transforms.ToPILImage()(image)
			image.save(f'outputs6/{iteration}real_alpha.jpg')

		if(iteration % 1000 == 0):
			learning_rate *= 0.95





print('\nTraining completed successfully.')

"""
	

"""
def composite(self, bg_tensor, fg_tensor, alpha_tensor):

		composite = (alpha_tensor * fg_tensor) + ((1 - alpha_tensor) * bg_tensor)

		return composite

"""