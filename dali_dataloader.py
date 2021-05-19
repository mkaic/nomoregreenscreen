print('loading libraries...')

import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import math as DALImath
import torchvision.transforms.functional as TF
import os
import random
import itertools
from PIL import Image
import numpy as np
import time
import cupy as cp
import imageio

#print('Initializing Dataset...')
class ImageOpener(object):
	"""
	A function which takes in a directory full of folders which have the exported frames of background clips, a directory full of
	folders which contain the exported frames of foreground clips with alpha channels
	"""

	def __init__(self, bg_dir, fg_dir, alpha_dir, batch_size):
		#_dir --> directory containing folders of images belonging to specific video clips for the background

		self.bg_dir = bg_dir
		self.fg_dir = fg_dir
		self.alpha_dir = alpha_dir
		self.batch_size = batch_size

		num_bg_clips = len(os.listdir(bg_dir))
		num_fg_clips = len(os.listdir(fg_dir))
		self.bg_fg_combos = list(itertools.product(range(num_bg_clips), range(num_fg_clips)))
		random.shuffle(self.bg_fg_combos)

	def __iter__(self):
		self.i = 0
		self.n = len(self.bg_fg_combos)
		return self

	def __next__(self):

		fg_batch = []
		bg_batch = []
		alpha_batch = []

		for _ in range(self.batch_size):
			#Create lists of directories which contain the frames of different videos
			bg_clips_list = os.listdir(self.bg_dir)
			fg_clips_list = os.listdir(self.fg_dir)
			alpha_clips_list = os.listdir(self.alpha_dir)

			#Grab the directory and thus a random background and foreground source video
			bg_clip_dir_idx, fg_clip_dir_idx = self.bg_fg_combos[self.i]

			bg_clip_dir = bg_clips_list[bg_clip_dir_idx] + '/'
			fg_clip_dir = fg_clips_list[fg_clip_dir_idx] + '/'
			alpha_clip_dir = fg_clip_dir

			#Specify where the frames of the background, foreground, and alpha can be found from the root folder.
			bg_clip_dir = self.bg_dir + bg_clip_dir
			fg_clip_dir = self.fg_dir + fg_clip_dir
			alpha_clip_dir = self.alpha_dir + alpha_clip_dir


			#Get the number of frames in the background clip and foreground clip.
			bg_frames_list = os.listdir(bg_clip_dir)
			num_frames_bg = len(bg_frames_list)
			fg_frames_list = os.listdir(fg_clip_dir)
			num_frames_fg = len(fg_frames_list)
			alpha_frames_list = os.listdir(alpha_clip_dir)
			num_frames_alpha = len(alpha_frames_list)

			#Select some random frames to use as background, foreground, and b-prime
			bg_idx = np.random.randint(0, num_frames_bg)
			bprime_idx = bg_idx

			fg_idx = np.random.randint(0, num_frames_fg)
			alpha_idx = fg_idx

			bg_path = bg_clip_dir + bg_frames_list[bg_idx]
			bg = open(bg_path, 'rb')
			bg = np.frombuffer(bg.read(), dtype = np.uint8)
			bg_batch.append(bg)

			fg_path = fg_clip_dir + fg_frames_list[fg_idx]
			fg = open(fg_path, 'rb')
			fg = np.frombuffer(fg.read(), dtype = np.uint8)
			fg_batch.append(fg)

			alpha_path = alpha_clip_dir + alpha_frames_list[alpha_idx]
			alpha = open(alpha_path, 'rb')
			alpha = np.frombuffer(alpha.read(), dtype = np.uint8)
			alpha_batch.append(alpha)

			self.i = (self.i + 1) % self.n

		return fg_batch, bg_batch, alpha_batch

#class

batch_size = 4

Iterator = ImageOpener(\
	fg_dir = 'dataset/train/fgr/', \
	bg_dir = 'dataset/train/bgr/', \
	alpha_dir = 'dataset/train/pha/',
	batch_size = batch_size)

class AugmentationPipeline(Pipeline):

	def __init__(self, dataset, num_threads, device_id, batch_size):

		super().__init__(\
			num_threads = num_threads, \
			device_id = device_id, \
			batch_size = batch_size)

		self.dataset = iter(dataset)
		self.batch_count = batch_size

	def define_graph(self):

		fg_read, bg_read, alpha_read = \
		fn.external_source(\
			source = self.dataset, \
			device = 'cpu', \
			num_outputs = 3)

		fg = fn.decoders.image(fg_read, device = 'mixed')
		bg = fn.decoders.image(bg_read, device = 'mixed')
		bprime = fn.copy(bg)
		alpha = fn.decoders.image(alpha_read, device = 'mixed')

		bg = fn.resize(bg, size = [1920, 1080])
		fg = fn.resize(fg, size = [1920, 1080])
		bprime = fn.resize(bprime, size = [1920, 1080])
		alpha = fn.resize(alpha, size = [1920, 1080])


		bg_rot = fn.random.uniform(range = [-8.0, 9.0])
		bg_brightness = fn.random.uniform(range = [0.85, 1.16])
		bg_contrast = fn.random.uniform(range = [0.85, 1.16])
		bg_saturation = fn.random.uniform(range = [0.85, 1.16])
		bg_hue = fn.random.uniform(range = [-0.05, 0.06])
		bg_blur_chance = fn.random.coin_flip(probability = 0.2)
		bg_blur = fn.random.uniform(range = [0.0, 10.0]) * bg_blur_chance
		bg_flip_chance = fn.random.coin_flip(probability = 0.5)

		bprime_rot = bg_rot + fn.random.uniform(range = [-1.0, 1.0])
		bprime_brightness = bg_brightness + fn.random.uniform(range = [-0.15, 0.15])
		bprime_contrast = bg_contrast + fn.random.uniform(range = [-0.15, 0.15])
		bprime_saturation = bg_saturation + fn.random.uniform(range = [-0.15, 0.15])
		bprime_hue = bg_hue + fn.random.uniform(range = [-0.1, 0.1])
		bprime_blur = bg_blur

		fg_rot = fn.random.uniform(range = [-15.0, 15.0])
		fg_brightness = fn.random.uniform(range = [0.85, 1.16])
		fg_contrast = fn.random.uniform(range = [0.85, 1.16])
		fg_saturation = fn.random.uniform(range = [0.85, 1.16])
		fg_hue = fn.random.uniform(range = [-0.05, 0.06])
		fg_blur_chance = fn.random.coin_flip(probability = 0.2)
		fg_blur = fn.random.uniform(range = [0.0, 10.0]) * fg_blur_chance
		fg_flip_chance = fn.random.coin_flip(probability = 0.5)

		alpha_rot = fg_rot
		alpha_blur = fg_blur

		bg = fn.color_twist(bg, brightness = bg_brightness, saturation = bg_saturation, hue = bg_hue)
		bg = fn.contrast(bg, contrast = bg_contrast)
		bg = fn.rotate(bg, angle = bg_rot, size = [1920, 1080])
		bg = fn.gaussian_blur(bg, sigma = bg_blur, window_size = 3)
		bg = fn.flip(bg, horizontal = bg_flip_chance)
		bg = DALImath.clamp(bg + fn.random.normal(bg) * 5, 0, 256)

		fg = fn.color_twist(fg, brightness = fg_brightness, saturation = fg_saturation, hue = fg_hue)
		fg = fn.contrast(fg, contrast = fg_contrast)
		fg = fn.rotate(fg, angle = fg_rot, size = [1920, 1080])
		fg = fn.gaussian_blur(fg, sigma = fg_blur, window_size = 3)
		fg = fn.flip(fg, horizontal = fg_flip_chance)
		bg = DALImath.clamp(fg + fn.random.normal(fg) * 5, 0, 256)
		
		bprime = fn.color_twist(bprime, brightness = bprime_brightness, saturation = bprime_saturation, hue = bprime_hue)
		bprime = fn.contrast(bprime, contrast = bprime_contrast)
		bprime = fn.rotate(bprime, angle = bprime_rot, size = [1920, 1080])
		bprime = fn.gaussian_blur(bprime, sigma = bprime_blur, window_size = 3)
		bprime = fn.flip(bprime, horizontal = bg_flip_chance)
		bg = DALImath.clamp(bprime + fn.random.normal(bprime) * 5, 0, 256)

		alpha = fn.rotate(alpha, angle = alpha_rot, size = [1920, 1080])
		alpha = fn.gaussian_blur(alpha, sigma = alpha_blur, window_size = 3)
		alpha = fn.flip(alpha, horizontal = fg_flip_chance)

		return (fg, bg, bprime, alpha)

class MyCustomDataloader(object):

	def __init__(self, loader):
		super().__init__()

		self.loader = loader

	def __iter__(self):

		return self

	def __next__(self):

		tensor_dict = next(self.loader)[0]

		fg = tensor_dict['fg'].permute(0,3,2,1)
		bg = tensor_dict['bg'].permute(0,3,2,1)
		bprime = tensor_dict['bprime'].permute(0,3,2,1)
		alpha = tensor_dict['alpha'].permute(0,3,2,1)
		alpha = alpha[:, :1]

		png = torch.cat([fg, alpha], 1)

		bg_trans_x = np.random.randint(-100, 100)
		bg_trans_y = np.random.randint(-100, 100)
		bg_shear_x = np.random.randint(-5, 6)
		bg_shear_y = np.random.randint(-5, 6)
		bg_scale = np.random.randint(8, 13) / 10

		aug_bg_params = {

			'img': bg,
			'angle': 0,
			'translate':[
				bg_trans_x,
				bg_trans_y
			],
			'shear':[
				bg_shear_x,
				bg_shear_y
			],
			'scale': bg_scale

		}
		aug_bg_tensor = TF.affine(**aug_bg_params)

		aug_bprime_params = {

			'img': bprime,
			'angle': 0,
			'translate':[
				bg_trans_x + np.random.randint(-5, 6),
				bg_trans_y + np.random.randint(-5, 6)
			],
			'shear':[
				bg_shear_x + np.random.randint(-2, 3),
				bg_shear_y + np.random.randint(-2, 3)
			],
			'scale': bg_scale + np.random.randint(-1, 2) / 100

		}
		aug_bprime_tensor = TF.affine(**aug_bprime_params)

		aug_png_params = {

			'img': png,
			'angle': 0,
			'translate':[
				np.random.randint(-100, 101),
				np.random.randint(-100, 101)
			],
			'shear':[
				np.random.randint(-15, 16),
				np.random.randint(-15, 16)
			],
			'scale': np.random.randint(3, 15) / 10

		}
		aug_png_tensor = TF.affine(**aug_png_params)
		
		aug_fg_tensor = aug_png_tensor[:, :3, :, :]
		aug_alpha_tensor = aug_png_tensor[:, 3:4, :, :]

		if(np.random.randint(0, 10) > 6):
			shadow_x = np.random.randint(0, 200)
			shadow_y = np.random.randint(0, 200)
			shadow_shear = np.random.randint(-30, 30)
			shadow_rotation = np.random.randint(-30, 30)
			shadow_strength = np.random.randint(10, 90) / 100
			shadow_blur = np.random.randint(2, 16) * 2 + 1

			shadow_stamp = TF.affine(aug_alpha_tensor, translate = [shadow_x, shadow_y], shear = shadow_shear, angle = shadow_rotation, scale = 1)
			shadow_stamp = TF.gaussian_blur(shadow_stamp, shadow_blur)
			shadow_stamp = shadow_stamp * shadow_strength

			aug_bg_tensor = aug_bg_tensor - aug_bg_tensor * shadow_stamp

		return (aug_bg_tensor, aug_fg_tensor, aug_bprime_tensor, aug_alpha_tensor)



