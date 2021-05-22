
print('Loading Libraries...')
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import time
from tqdm import tqdm
from dali_dataloader import AugmentationPipeline, ImageOpener, MyCustomDataloader
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from kornia.filters import sobel
from coarse_definition import CoarseMatteGenerator
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

batch_size = 3

#Initialize the dataset and the loader to feed the data to the network.

ImageFeeder = ImageOpener(\
	fg_dir = 'dataset/train/fgr/', \
	bg_dir = 'dataset/train/bgr/', \
	alpha_dir = 'dataset/train/pha/',
	batch_size = batch_size)

Pipeline = AugmentationPipeline(\
	dataset = ImageFeeder, \
	num_threads = 4, \
	device_id = 0,
	batch_size = batch_size)

Pipeline.build()
Loader = DALIGenericIterator(pipelines = [Pipeline], output_map = ['fg', 'bg', 'bprime', 'alpha'])
DALIDataloader = MyCustomDataloader(Loader)


print('Initializing Network...')

#Initialize the network which will produce a coarse alpha (1 chan), foreground (3 chan) confidence map (1 chan), and a number of hidden channels (num_hidden_channels chan)...
coarse = CoarseMatteGenerator().train().to(device)

use_amp = False

#track how many batches have been done for things like periodic outputs and eventually scheduling.
iteration = 0

coarse_parameters = [\
	{'params': coarse.Encoder.parameters(), 'lr': 0.0001},\
	{'params': coarse.ASPP.parameters(), 'lr': 0.0005},\
	{'params': coarse.Decoder.parameters(), 'lr': 0.0005}\
	]

coarse_opt = torch.optim.Adam(coarse_parameters, lr = 0.0001)

print('\nTraining...')

L1Loss = nn.L1Loss()
MSELoss = nn.MSELoss()

for iteration in tqdm(range(600000)):

	with torch.cuda.amp.autocast(enabled = use_amp):

		real_background, real_foreground, real_bprime, real_alpha = next(DALIDataloader)

		"""
		real_foreground = real_foreground.to(device)
		real_background = real_background.to(device)
		real_alpha = real_alpha.to(device)
		real_bprime = real_bprime.to(device)
		"""

		#Composite the augmented foreground onto the augmented background according to the augmented alpha.
		composite_tensor = composite(real_background, real_foreground, real_alpha)

		#return the input tensor (composite plus b-prime) and the alpha_tensor. The input tensor is just a bunch of channels, the real_alpha is the central (singular) alpha
		#corresponding to the target frame.
		input_tensor = torch.cat([composite_tensor, real_bprime], 1)

		coarse_input = F.interpolate(input_tensor, size = [input_tensor.shape[-2]//4, input_tensor.shape[-1]//4])

		#Get a downsampled version of the alpha for grading the coarse network on
		real_coarse_alpha = F.interpolate(real_alpha, size = [real_alpha.shape[-2]//4, real_alpha.shape[-1]//4])

		#Generate a fake coarse alpha, along with a guessed error map and some hidden channel data. Oh yeah and the foreground residual
		fake_coarse = coarse(coarse_input)
		fake_coarse_alpha = torch.clamp(fake_coarse[:, 0:1], 0, 1)
		fake_coarse_foreground_residual = fake_coarse[:, 1:4]
		fake_coarse_error = torch.clamp(fake_coarse[:, 4:5], 0, 1)
		fake_coarse_hidden_channels = torch.relu(fake_coarse[:,5:])

		real_coarse_composite = F.interpolate(composite_tensor, size = [composite_tensor.shape[-2]//4, composite_tensor.shape[-1]//4])
		fake_coarse_foreground = torch.clamp(real_coarse_composite + fake_coarse_foreground_residual, 0, 1)
		

		#The real error map is calculated as the squared difference between the real alpha and the fake alpha.
		real_coarse_error = torch.clamp(torch.abs(real_coarse_alpha.detach()-fake_coarse_alpha.detach()), 0, 1)

		#construct the fake foreground
		#fake_coarse_foreground = torch.clamp(real_coarse_composite[:, dataset_params["comp_context_depth"]*3:dataset_params["comp_context_depth"]*3 + 3] + fake_coarse_foreground_residual, 0, 1)
		foreground_penalty_zone = real_coarse_alpha > 0
		real_coarse_foreground = F.interpolate(real_foreground, size = [real_foreground.shape[-2]//4, real_foreground.shape[-1]//4])
	
		coarse_sobel = sobel(fake_coarse_alpha)
		real_sobel = sobel(real_coarse_alpha)

	#The loss of the coarse network is L1 loss of coarse alpha, L1 loss of coarse error, and L1 loss (only where real_alpha >0.1) of coarse foreground.
	coarse_loss = \
	L1Loss(fake_coarse_alpha,real_coarse_alpha) + \
	MSELoss(fake_coarse_error,real_coarse_error) + \
	L1Loss((real_coarse_foreground * foreground_penalty_zone), (fake_coarse_foreground * foreground_penalty_zone)) + \
	L1Loss(coarse_sobel,real_sobel)

	#if it's before the training cutoff, then the loss is just for the coarse network.
	coarse_opt.zero_grad()
	coarse_loss.backward()
	coarse_opt.step()

	#For keeping track of the outputs so I can look through them to see the network is working right.
	iteration += 1


	if(iteration % 1000 == 0):
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

		image = real_coarse_composite[0]
		image = transforms.ToPILImage()(image)
		image.save(f'outputs7/{iteration}E_coarse_composite.jpg')
		

	if(iteration % 1000 == 0):

		print(coarse_loss)	

	if(iteration % 15000 == 0):

		torch.save(coarse, f"./model_saves/proper_coarse_epoch{iteration}.zip")


print('\nTraining completed successfully.')
