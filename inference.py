
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

import os

from model_definition import *
from train_utils import get_image_patches, replace_image_patches, color_ramp

device = "cuda"

#Load coarse gen and refinement networks from their saved weights on disks.
print("Loading neural networks...")
Coarse = torch.load('model_saves/final_coarse_1.zip').eval().to(device)
Refine = torch.load('model_saves/final_refine_1.zip').eval().to(device)
search_width = 10

#Allows me to run this from the command line with custom inputs/outputs specified there.
parser = argparse.ArgumentParser()
parser.add_argument('--source', '-s', help = 'Path to the video you want to remove the background FROM.')
parser.add_argument('--background', '-b', help = 'Path to the video you want to give the model as reference for what JUST THE BACKGROUND looks like.')
parser.add_argument('--output', '-o', help = 'Path to output the final RGBA PNG sequence to.')
parser.add_argument('--searchwidth', '-w', help = 'how many frames temprally away from the source frame to check in the background video')
args = parser.parse_args()

print("Preprocessing source video...")
#Preprocess the background to a series of jpgs, and the foreground to PNGs (so it's lossless). It ain't pretty but it's necessary.
os.system(f'ffmpeg -loglevel error -i {args.source} -vsync 0 cached_frames/source/%d.png')
print("Preprocessing background video...")
os.system(f'ffmpeg -loglevel error -i {args.background} -vsync 0 -q:v 2 cached_frames/background/%d.jpg')

if(args.searchwidth):
	search_width = int(args.searchwidth)

print('Detecting features in source video...')

#The main loop. Everything inside this gets executed on every frame of the source video: background
#frame finding, homography, coarse matte generation, and matte refinement.
source_list = os.listdir('cached_frames/source/')
source_len = len(source_list)
background_list = os.listdir('cached_frames/background/')
background_len = len(background_list)

detector = cv.ORB_create(1000)


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

for source_idx, source_name in enumerate(tqdm(source_list)):

	source_img = np.asarray(Image.open(f'cached_frames/source/{source_name}'))

	#tick = time.time()
	best_score = -1
	
	#always temporally center our search
	background_start_idx = int(source_idx / source_len * background_len)
	search_start_idx = max(0, background_start_idx - search_width)
	search_end_idx = min(background_len, background_start_idx + search_width)
	best_background = np.zeros_like(source_img)

	for background_idx, background_name in enumerate(background_list\
		[search_start_idx: search_end_idx], start = search_start_idx):
		
		matches = matcher.match(background_des_list[background_idx], source_des_list[source_idx], None)

		#sort the matches by lowest distance up. the lambda is just a little function that reads like this:
		# function such that for each item x, retrieve the distance property of x.
		matches.sort(key = lambda x: x.distance, reverse = False)
		top_matches = matches[:int(len(matches)*0.15)]

		source_kp = source_kp_list[source_idx]
		source_coords = np.float32([source_kp[match.trainIdx].pt for match in top_matches]).reshape(-1, 1, 2)

		background_kp = background_kp_list[background_idx]
		background_coords = np.float32([background_kp[match.queryIdx].pt for match in top_matches]).reshape(-1, 1, 2)

		H, inliers = cv.findHomography(background_coords, source_coords, cv.RANSAC, 5.0)

		inliers_total = np.sum(inliers)

		background_img = np.asarray(Image.open(f'cached_frames/background/{background_name}'))

		Image.fromarray(cv.drawMatches(source_img, source_kp, background_img, background_kp, top_matches, None)).save(f'alignment_test/alignment{source_idx}{background_idx}.jpeg')


		h, w = source_img.shape[:2]
		warped_background = cv.warpPerspective(background_img, H, (w,h))
		warped_mask = cv.warpPerspective(np.ones((h,w)), H, (w,h))

		warped_background[warped_mask != 1] = source_img[warped_mask != 1]

		mean_pixel_error = np.mean(np.abs(warped_background - source_img))

		score = inliers_total

		if (score > best_score):

			best_unwarped_background = background_img
			best_background = warped_background
			best_score = score

	background_PIL = Image.fromarray(best_background)
	source_PIL = Image.fromarray(source_img)
	unwarped_PIL = Image.fromarray(best_unwarped_background)

	background_PIL.save(f'alignment_test/{source_idx}background.jpg')
	source_PIL.save(f'alignment_test/{source_idx}source.jpg')
	unwarped_PIL.save(f'alignment_test/{source_idx}unwarped.jpg')

	#print(time.time() - tick)

print('Deleting cached frames...')

os.system('rm cached_frames/background/*')
os.system('rm cached_frames/source/*')

print('Done.')