
print("Loading libraries...")
#Torch stuff, plust image transforms, plus torchsummary for getting an idea of how many params my model has, etc.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchsummary import summary

#TQDM for progress bars, PIL for image loading, FFMPEG for video processing, argparse for command line interactivity
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import time
import cv2 as cv
import skimage.transform as sk_transform
from skimage.util import img_as_ubyte

import os
#from coarse_definition import CoarseMatteGenerator
#from refine_definition import RefinementNetwork

from model_definition import CoarseMatteGenerator, RefinementNetwork

from train_utils import get_image_patches, replace_image_patches, color_ramp

device = "cuda"

#Load coarse gen and refinement networks from their saved weights on disks.
print("Loading neural networks...")
Coarse = torch.load('model_saves/coarse_generator_network_epoch_525000.zip').eval().to(device)
Refine = torch.load('model_saves/refinement_network_epoch_525000.zip').eval().to(device)
search_width = 10

#Allows me to run this from the command line with custom inputs/outputs specified there.
parser = argparse.ArgumentParser()
parser.add_argument('--source', '-s', help = 'Path to the video you want to remove the background FROM.')
parser.add_argument('--background', '-b', help = 'Path to the video you want to give the model as reference for what JUST THE BACKGROUND looks like.')
parser.add_argument('--output', '-o', help = 'Path to output the final RGBA PNG sequence to.')
parser.add_argument('--searchdepth', '-w', help = 'how many frames temprally away from the source frame to check in the background video', default = 10)
parser.add_argument('--searchstride', '-j', help = 'how much skimming the algorithm does', default = 2)

args = parser.parse_args()

args.searchdepth = int(args.searchdepth)
args.searchstride = int(args.searchstride)

print("Preprocessing source video...")
#Preprocess the background to a series of jpgs, and the foreground to PNGs (so it's lossless). It ain't pretty but it's necessary.
os.system(f'ffmpeg -loglevel error -i {args.source} -vsync 0 -q:v 2 cached_frames/source/%04d.jpg')
print("Preprocessing background video...")
os.system(f'ffmpeg -loglevel error -i {args.background} -vsync 0 -q:v 2 cached_frames/background/%04d.jpg')

if(not os.path.exists(args.output)):
	os.mkdir(args.output)

print('Detecting features in source video...')

#The main loop. Everything inside this gets executed on every frame of the source video: background
#frame finding, homography, coarse matte generation, and matte refinement.
source_list = sorted(os.listdir('cached_frames/source/'))
source_len = len(source_list)
background_list = sorted(os.listdir('cached_frames/background/'))
background_len = len(background_list)

detector = cv.ORB_create()


#LOOP over all source PNG frames and detect and compute features for them
source_kp_list = []
source_des_list = []

for source_name in tqdm(source_list):

	PILsource = Image.open(f'cached_frames/source/{source_name}')
	
	source = np.asarray(PILsource)
	BWsource = cv.cvtColor(source, cv.COLOR_RGB2GRAY)
	
	source_kp, source_des = detector.detectAndCompute(BWsource, None)

	source_kp_list.append(source_kp)
	source_des_list.append(source_des)

background_kp_list = []
background_des_list = []

print("Detecting features in background video...")
#LOOP over all background PNG frames and detect and compute features for them
for background_name in tqdm(background_list):

	PILbackground = Image.open(f'cached_frames/background/{background_name}')
	
	background = np.asarray(PILbackground)
	BWbackground = cv.cvtColor(background, cv.COLOR_RGB2GRAY)
	
	background_kp, background_des = detector.detectAndCompute(BWbackground, None)

	background_kp_list.append(background_kp)
	background_des_list.append(background_des)

print("Matching background frames to source frames and running inference...")
matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

class UserInputDataset(Dataset):

	def __init__(self, depth = 5, stride = 2):
		super().__init__()

		self.depth = depth
		self.stride = stride

		return

	def __len__(self):

		return source_len

	def __getitem__(self, source_idx):

		source_name = source_list[source_idx]

		source_img = np.asarray(Image.open(f'cached_frames/source/{source_name}'))
		#print(f'src_mean {np.mean(source_img)}')

		#tick = time.time()
		best_score = 10000
		
		#always temporally center our search
		background_start_idx = int(source_idx / source_len * background_len)
		search_start_idx = max(0, background_start_idx - self.depth*self.stride)
		search_end_idx = min(background_len, background_start_idx + self.depth*self.stride)
		best_background = np.zeros_like(source_img)

		#getting strided files is a hassle, because I can't use enumerate to get the indices.

		for background_idx, background_name in zip(\
			list(range(search_start_idx, search_end_idx, self.stride)),\
			background_list[search_start_idx : search_end_idx : self.stride]):
			
			matches = matcher.knnMatch(background_des_list[background_idx], source_des_list[source_idx], k = 2)

			#sort the matches by lowest distance up. the lambda is just a little function that reads like this:
			# function such that for each item x, retrieve the distance property of x.
			top_matches = []

			for match in matches:

				best = match[0]
				worst = match[1]

				if(best.distance < 0.75 * worst.distance):

					top_matches.append(best)

			#print(len(top_matches))

			source_kp = source_kp_list[source_idx]
			source_coords = np.float32([source_kp[match.trainIdx].pt for match in top_matches]).reshape(-1, 2)

			background_kp = background_kp_list[background_idx]
			background_coords = np.float32([background_kp[match.queryIdx].pt for match in top_matches]).reshape(-1, 2)

			poly_tform = sk_transform.estimate_transform(ttype='polynomial', src=source_coords, dst=background_coords, order=2)

			background_img = np.asarray(Image.open(f'cached_frames/background/{background_name}'))

			#Image.fromarray(cv.drawMatches(source_img, source_kp, background_img, background_kp, top_matches, None)).save(f'alignment_test/alignment{source_idx}{background_idx}.jpeg')


			h, w = source_img.shape[:2]
			warped_background = img_as_ubyte(sk_transform.warp(background_img, poly_tform))
			warped_mask = img_as_ubyte(sk_transform.warp(np.ones((h,w)), poly_tform))



			mean_pixel_error = np.mean(np.abs(warped_background - source_img) * np.expand_dims(warped_mask, axis = 2))

			#warped_background[warped_mask != 1] = source_img[warped_mask != 1]
			overall_coverage = np.mean(warped_mask) ** 30 * 100

			score = mean_pixel_error - (overall_coverage * 2)

			#print(int(mean_pixel_error), int(overall_coverage), int(inliers_total))

			if (score < best_score):

				best_unwarped_background = background_img
				best_background = warped_background
				best_score = score
				best_mask = warped_mask
		
		background_PIL = Image.fromarray(best_background)
		source_PIL = Image.fromarray(source_img)
		unwarped_PIL = Image.fromarray(best_unwarped_background)
		mask_PIL = Image.fromarray(best_mask)

		background_tensor = transforms.ToTensor()(background_PIL)
		source_tensor = transforms.ToTensor()(source_PIL)

		
		background_PIL.save(f'alignment_test/{source_idx}background.jpg')
		source_PIL.save(f'alignment_test/{source_idx}source.jpg')
		unwarped_PIL.save(f'alignment_test/{source_idx}unwarped.jpg')
		#mask_PIL.save(f'alignment_test/{source_idx}mask.jpg')

		#print(time.time() - tick)
		

		return source_tensor, background_tensor

dataset = UserInputDataset(depth = args.searchdepth, stride = args.searchstride)
batch_size = 4
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 6, pin_memory = True)

with torch.no_grad():

	batch_number = 0
	for composite_tensor, background_tensor in tqdm(dataloader):

		background_tensor = background_tensor.to(device)
		composite_tensor = composite_tensor.to(device)

		input_tensor = torch.cat([composite_tensor, background_tensor], 1)
		coarse_input = F.interpolate(input_tensor, size = [input_tensor.shape[-2]//4, input_tensor.shape[-1]//4])

		#Generate a fake coarse alpha, along with a guessed error map and some hidden channel data. Oh yeah and the foreground residual
		fake_coarse = Coarse(coarse_input)
		fake_coarse_alpha = torch.clamp(fake_coarse[:, 0:1], 0, 1)
		fake_coarse_foreground_residual = fake_coarse[:, 1:4]
		fake_coarse_error = torch.clamp(fake_coarse[:, 4:5], 0, 1)

		if(fake_coarse.shape[1] > 5):
			fake_coarse_hidden_channels = torch.relu(fake_coarse[:,5:])

		downsampled_input_tensor = F.interpolate(input_tensor, [input_tensor.shape[-2]//2, input_tensor.shape[-1]//2])
		upscaled_coarse_outputs = F.interpolate(fake_coarse, [input_tensor.shape[-2]//2, input_tensor.shape[-1]//2])
		start_patch_source = torch.cat([downsampled_input_tensor, upscaled_coarse_outputs], 1)

		start_patches, indices = get_image_patches(start_patch_source.detach(), fake_coarse_error.detach(), patch_size = 8, stride = 2, k = 10000)
		middle_patches, _ = get_image_patches(input_tensor.detach(), fake_coarse_error.detach(), patch_size = 8, stride = 4, k = 10000)

		#Now, feed the outputs of the coarse generator into the refinement network, which will refine patches.
		fake_refined_patches = Refine(start_patches, middle_patches)

		mega_upscaled_fake_coarse = F.interpolate(fake_coarse[:, :4].detach(), size = input_tensor.shape[-2:])
		fake_refined = replace_image_patches(images = mega_upscaled_fake_coarse, patches = fake_refined_patches, indices = indices)
		fake_refined_alpha = color_ramp(0.05, 0.95, torch.clamp(fake_refined[:, 0:1], 0, 1))
		fake_refined_foreground = torch.clamp(fake_refined[:, 1:4] + composite_tensor, 0, 1)

		RGBA = torch.cat([fake_refined_foreground, fake_refined_alpha], 1)

		for j in range(input_tensor.shape[0]):
			image = transforms.ToPILImage()(RGBA[j])
			output_idx = batch_number*batch_size + j
			output_idx = str(output_idx).zfill(5)
			image.save(f'{args.output}/{output_idx}.png')

		batch_number += 1


print('Deleting cached frames...')

os.system('rm cached_frames/background/*')
os.system('rm cached_frames/source/*')

print('Done.')