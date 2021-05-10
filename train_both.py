
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

from kornia.filters import sobel

from matte_dataset import MatteDataset
from model_definition import *
from train_utils import *
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
dataset_params = {
	
	'bg_dir':'train_set_2/backgrounds/',
	'fg_dir':'train_set_2/foregrounds/',

}

batch_size = 4

#Initialize the dataset and the loader to feed the data to the network.
dataset = MatteDataset(**dataset_params)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 6, pin_memory = True)


print('Initializing Network...')

num_hidden_channels = 32

#Initialize the network which will produce a coarse alpha (1 chan), foreground (3 chan) confidence map (1 chan), and a number of hidden channels (num_hidden_channels chan)...
coarse = torch.load("model_saves/new_coarse.zip").train().to(device)
refine = RefinementNetwork(coarse_channels = 5).train().to(device)

use_amp = False

criterion = nn.L1Loss()

#track how many batches have been done for things like periodic outputs and eventually scheduling.
iteration = 0

coarse_learning_rate = 0.000007
coarse_opt = torch.optim.Adam(coarse.parameters(), lr = coarse_learning_rate)
coarse_scheduler = torch.optim.lr_scheduler.StepLR(coarse_opt, step_size = 2000, gamma = 0.98)

refine_learning_rate = 0.00001
refine_opt = torch.optim.Adam(coarse.parameters(), lr = refine_learning_rate)
refine_scheduler = torch.optim.lr_scheduler.StepLR(refine_opt, step_size = 2000, gamma = 0.98)

print('\nTraining...')
for epoch in range(8):
	print(f'Epoch: {epoch}')

	#Get an example from the dataset.
	for real_foreground, real_background, real_alpha, real_bprime in tqdm(loader):

		with torch.cuda.amp.autocast(enabled = use_amp):

			real_foreground = real_foreground.to(device)
			real_background = real_background.to(device)
			real_alpha = real_alpha.to(device)
			real_bprime = real_bprime.to(device)

			#Composite the augmented foreground onto the augmented background according to the augmented alpha.
			composite_tensor = composite(real_background, real_foreground, real_alpha).view(batch_size, -1, real_foreground.shape[-2], real_foreground.shape[-1])

			#return the input tensor (composite plus b-prime) and the alpha_tensor. The input tensor is just a bunch of channels, the real_alpha is the central (singular) alpha
			#corresponding to the target frame.
			input_tensor = torch.cat([composite_tensor, real_bprime], 1)

			coarse_input = F.interpolate(input_tensor, size = [input_tensor.shape[-2]//4, input_tensor.shape[-1]//4])

			#Get a downsampled version of the alpha for grading the coarse network on
			real_coarse_alpha = F.interpolate(real_alpha, size = [real_alpha.shape[-2]//4, real_alpha.shape[-1]//4])

			#Generate a fake coarse alpha, along with a guessed error map and some hidden channel data. Oh yeah and the foreground residual
			fake_coarse = coarse(coarse_input)
			fake_coarse_alpha = color_ramp(0.1, 0.9, torch.clamp(fake_coarse[:,0:1,:,:], 0, 1))
			fake_coarse_error = torch.sigmoid(fake_coarse[:,1:2,:,:])
			fake_coarse_foreground_residual = fake_coarse[:,2:5,:,:]

			real_coarse_composite = F.interpolate(composite_tensor, size = [composite_tensor.shape[-2]//4, composite_tensor.shape[-1]//4])
			fake_coarse_foreground = torch.clamp(real_coarse_composite + fake_coarse_foreground_residual, 0, 1)

			#if(num_hidden_channels > 0):
				#fake_coarse_hidden_channels = torch.relu(fake_coarse[:,5:,:,:])

			#The real error map is calculated as the squared difference between the real alpha and the fake alpha.
			real_coarse_error = torch.abs(real_coarse_alpha.detach()-fake_coarse_alpha.detach())

			#construct the fake foreground
			#fake_coarse_foreground = torch.clamp(real_coarse_composite[:, dataset_params["comp_context_depth"]*3:dataset_params["comp_context_depth"]*3 + 3] + fake_coarse_foreground_residual, 0, 1)
			foreground_penalty_zone = real_coarse_alpha > 0.1
			real_coarse_foreground = F.interpolate(real_foreground, size = [real_foreground.shape[-2]//4, real_foreground.shape[-1]//4])
		
			coarse_sobel = torch.mean(sobel(fake_coarse_alpha))

		#The loss of the coarse network is L1 loss of coarse alpha, L1 loss of coarse error, and L1 loss (only where real_alpha >0.1) of coarse foreground.
		coarse_loss = criterion(fake_coarse_alpha, real_coarse_alpha) + \
		criterion(fake_coarse_error,real_coarse_error) + \
		torch.mean(torch.abs((real_coarse_foreground - fake_coarse_foreground) * foreground_penalty_zone)) + \
		coarse_sobel/2

		#if it's before the training cutoff, then the loss is just for the coarse network.
		coarse_opt.zero_grad()
		coarse_loss.backward()
		coarse_opt.step()
		coarse_scheduler.step()

		with torch.cuda.amp.autocast(enabled = use_amp):
			#DO PATCH GETTING ON CPU TO SPEED UP TRAINING. THEN LOAD PATCHES ONTO GPU. SAVE MEMORY AND TIME. (OH I THINK I JUST DID IT. HAVEN'T TESTED THOUGH)

			downsampled_input_tensor = F.interpolate(input_tensor, [input_tensor.shape[-2]//2, input_tensor.shape[-1]//2])
			upscaled_coarse_outputs = F.interpolate(fake_coarse, [input_tensor.shape[-2]//2, input_tensor.shape[-1]//2])
			start_patch_source = torch.cat([downsampled_input_tensor, upscaled_coarse_outputs], 1)

			start_patches, indices = get_image_patches(start_patch_source.detach(), fake_coarse_error.detach(), patch_size = 8, stride = 2, k = 10000)
			middle_patches, _ = get_image_patches(input_tensor.detach(), fake_coarse_error.detach(), patch_size = 8, stride = 4, k = 10000)

			#Now, feed the outputs of the coarse generator into the refinement network, which will refine patches.
			fake_refined_patches = refine(start_patches, middle_patches)

			mega_upscaled_fake_coarse = F.interpolate(fake_coarse[:, :4].detach(), size = input_tensor.shape[-2:])
			fake_refined = replace_image_patches(images = mega_upscaled_fake_coarse, patches = fake_refined_patches, indices = indices)
			fake_refined_alpha = fake_refined[:, 0:1]
			fake_refined_foreground = fake_refined[:, 1:4]
		#The loss of the refinement network is just the pixel difference between what it made and what it was supposed to make.
		refine_opt.zero_grad()

		refine_loss = criterion(fake_refined_alpha, real_alpha) + \
		criterion(fake_refined_foreground, real_foreground)

		refine_loss.backward()
		refine_opt.step()
		refine_scheduler.step()

		#For keeping track of the outputs so I can look through them to see the network is working right.
		iteration += 1


		if(iteration % 500 == 0):
			image = fake_coarse_alpha[0]
			image = transforms.ToPILImage()(image)
			image.save(f'outputs7/{iteration}C_fake_coarse_alpha.jpg')

			image = real_coarse_alpha[0]
			image = transforms.ToPILImage()(image)
			image.save(f'outputs7/{iteration}B_real_alpha.jpg')

			image = fake_coarse_foreground[0]
			image = transforms.ToPILImage()(image)
			image.save(f'outputs7/{iteration}A_fake_foreground.jpg')

			image = fake_coarse_error[0]
			image = transforms.ToPILImage()(image)
			image.save(f'outputs7/{iteration}D_fake_error.jpg')

			image = fake_refined_alpha[0]
			image = transforms.ToPILImage()(image)
			image.save(f'outputs7/{iteration}E_refined_alpha.jpg')

			image = fake_refined_foreground[0]
			image = transforms.ToPILImage()(image)
			image.save(f'outputs7/{iteration}F_refined_foreground.jpg')
			

		if(iteration % 500 == 0):

			print(coarse_loss)
			print(refine_loss)	


print('\nTraining completed successfully.')

torch.save(coarse, "./model_saves/coarse_generator_network_save_1.zip")
torch.save(refine, "./model_saves/refinement_network_save_1.zip")
