
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
refine = MatteRefinementNetwork(input_channels = input_channels, coarse_channels = num_hidden_channels + 2, output_channels = 1, chan = 32, device = device).train().to(device)

learning_rate = 0.00001

bce_loss = nn.BCEWithLogitsLoss()

#track how many batches have been done for things like periodic outputs and eventually scheduling.
iteration = 0

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled = use_amp)

print('\nTraining...')
for epoch in range(20):
	print(f'Epoch: {epoch}')

	#Get an example from the dataset.
	for input_tensor, real_alpha in tqdm(loader):

		with torch.cuda.amp.autocast(enabled = use_amp):

			#transfer the important tensors to GPU
			input_tensor = input_tensor.to(device)
			real_alpha = real_alpha.to(device)

			#Get a downsampled version of the alpha for grading the coarse network on
			real_coarse_alpha = F.interpolate(real_alpha, size = [real_alpha.shape[-2]//4, real_alpha.shape[-1]//4])

			#Generate a fake coarse alpha, along with a guessed error map and some hidden channel data.
			fake_coarse = coarse(input_tensor)
			fake_coarse_alpha = fake_coarse[:,0:1,:,:]
			fake_coarse_error = fake_coarse[:,1:2,:,:]
			fake_coarse_hidden = fake_coarse[:,2:,:,:]

			#The real error map is calculated as the squared difference between the real alpha and the fake alpha.
			real_coarse_error = torch.square(real_coarse_alpha.detach()-fake_coarse_alpha.detach())


			#Do the learning stuff. I've made two separate networks to hopefully make scheduling easier, dunno if that's going to be the case
			#or not though. Any how, it means that there are two optimizers.
			coarse_opt = torch.optim.Adam(coarse.parameters(), lr = learning_rate)

		if(iteration < 500):

			with torch.cuda.amp.autocast(enabled = use_amp):
				#The loss of the coarse network is the pixel difference between the real and fake coarse alphas and error maps added together.
				coarse_loss = bce_loss(fake_coarse_alpha, real_coarse_alpha) + bce_loss(fake_coarse_error, real_coarse_error)
				coarse_opt = torch.optim.Adam(coarse.parameters(), lr = learning_rate)
				
			coarse_opt.zero_grad()
			scaler.scale(coarse_loss).backward()
			scaler.step(coarse_opt)
			scaler.update()

		else:

			with torch.cuda.amp.autocast(enabled = use_amp):
				coarse_opt = torch.optim.Adam(coarse.parameters(), lr = learning_rate)
				refine_opt = torch.optim.Adam(refine.parameters(), lr = learning_rate)
				#Now, feed the outputs of the coarse generator into the refinement network, which will refine patches.
				fake_refined_alpha = refine(fake_coarse_alpha = fake_coarse_alpha, fake_coarse_error = fake_coarse_alpha, fake_coarse_hidden = fake_coarse_hidden, input_tensor = input_tensor)

				#The loss of the coarse network is the pixel difference between the real and fake coarse alphas and error maps added together.
				coarse_loss = bce_loss(fake_coarse_alpha, real_coarse_alpha) + bce_loss(fake_coarse_error, real_coarse_error)
				#The loss of the refinement network is just the pixel difference between what it made and what it was supposed to make.
				refine_loss = bce_loss(fake_refined_alpha, real_alpha)
				total_loss = coarse_loss + refine_loss

			coarse_opt.zero_grad()
			refine_opt.zero_grad()

			scaler.scale(total_loss).backward()

			scaler.step(coarse_opt)
			scaler.step(refine_opt)

			scaler.update()
		

		#For keeping track of the outputs so I can look through them to see the network is working right.
		iteration += 1

		if(iteration % 1 == 0 and not iteration < 502):
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
def composite(self, bg_tensor, fg_tensor, alpha_tensor):

		composite = (alpha_tensor * fg_tensor) + ((1 - alpha_tensor) * bg_tensor)

		return composite

"""