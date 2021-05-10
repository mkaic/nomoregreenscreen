
#print('Loading libraries...')
#torch for obvious reasons, transforms for PIL to Tensor and back, TF for fine control over augmentation,
#Dataset for creating a dataset, os for looping through directories, PIL for I/O of images, numpy for random ints,
#and finally, time for debugging and performance measurement. Itertools for getting all combos of bg and fg
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import time
import itertools

#print('Initializing Dataset...')
class MatteDataset(Dataset):
	"""
	A function which takes in a directory full of folders which have the exported frames of background clips, a directory full of
	folders which contain the exported frames of foreground clips with alpha channels
	"""

	def __init__(self, bg_dir, fg_dir):
		#_dir --> directory containing folders of images belonging to specific video clips for the background

		self.bg_dir = bg_dir
		self.fg_dir = fg_dir

		num_bg_clips = len(os.listdir(bg_dir))
		num_fg_clips = len(os.listdir(fg_dir))
		self.bg_fg_combos = list(itertools.product(range(num_bg_clips), range(num_fg_clips)))

	def augment(self, bg_tensor, fg_tensor):
		#Expects inputs of size C x H x W

		aug_bg_params = {

			'img': bg_tensor,
			'angle': np.random.randint(-3, 4),
			'translate':[
				np.random.randint(-20, 21),
				np.random.randint(-20, 21)
			],
			'shear':[
				np.random.randint(-3, 4),
				np.random.randint(-3, 4)
			],
			'scale':1

		}
		aug_bg_tensor = TF.affine(**aug_bg_params)

		aug_png_params = {

			'img': fg_tensor,
			'angle': np.random.randint(-15, 16),
			'translate':[
				np.random.randint(-100, 101),
				np.random.randint(-100, 101)
			],
			'shear':[
				np.random.randint(-15, 16),
				np.random.randint(-15, 16)
			],
			'scale':1

		}
		aug_png_tensor = TF.affine(**aug_png_params)
		
		aug_fg_tensor = aug_png_tensor[:3, :, :]
		aug_alpha_tensor = aug_png_tensor[3:4, :, :]

		return aug_fg_tensor, aug_bg_tensor, aug_alpha_tensor

	#Does what it says on the box. Takes in a background, foreground, and alpha, and composites them into one image accordingly.

	def shadow_augment(self, bg_tensor, alpha_tensor):

		#randomize a slight affine transform to make sure the shadow is offset from the subject.
		shadow_x = np.random.randint(0, 200)
		shadow_y = np.random.randint(0, 200)
		shadow_shear = np.random.randint(-30, 30)
		shadow_rotation = np.random.randint(-30, 30)
		shadow_strength = np.random.randint(10, 90) / 100
		shadow_blur = np.random.randint(2, 16) * 2 + 1

		shadow_stamp = TF.affine(alpha_tensor, translate = [shadow_x, shadow_y], shear = shadow_shear, angle = shadow_rotation, scale = 1)
		shadow_stamp = TF.gaussian_blur(shadow_stamp, shadow_blur)
		shadow_stamp = shadow_stamp * shadow_strength

		return bg_tensor - bg_tensor * shadow_stamp

	def __getitem__(self, index):

		#Create lists of directories which contain the frames of different videos
		bg_clips_list = os.listdir(self.bg_dir)
		fg_clips_list = os.listdir(self.fg_dir)

		#Grab the directory and thus a random background and foreground source video
		bg_clip_dir_idx, fg_clip_dir_idx = self.bg_fg_combos[index]

		bg_clip_dir = bg_clips_list[bg_clip_dir_idx] + '/'
		fg_clip_dir = fg_clips_list[fg_clip_dir_idx] + '/'

		#Specify where the frames of the background, foreground, and alpha can be found from the root folder.
		bg_clip_dir = self.bg_dir + bg_clip_dir
		fg_clip_dir = self.fg_dir + fg_clip_dir

		#Get the number of frames in the background clip and foreground clip.
		bg_frames_list = os.listdir(bg_clip_dir)
		num_frames_bg = len(bg_frames_list)
		fg_frames_list = os.listdir(fg_clip_dir)
		num_frames_fg = len(fg_frames_list)

		#Select some random frames to use as background, foreground, and b-prime
		bg_idx = np.random.randint(0, num_frames_bg)

		bprime_offset = np.random.randint(-5, 6)
		bprime_idx = min(max(0, bg_idx + bprime_offset), num_frames_bg - 1)

		fg_idx = np.random.randint(0, num_frames_fg)

		bg_tensor = transforms.ToTensor()(Image.open(bg_clip_dir + bg_frames_list[bg_idx]))
		fg_tensor = transforms.ToTensor()(Image.open(fg_clip_dir + fg_frames_list[fg_idx]))

		fg_tensor, bg_tensor, alpha_tensor = self.augment(bg_tensor, fg_tensor)

		bprime_tensor = transforms.ToTensor()(Image.open(bg_clip_dir + bg_frames_list[bprime_idx]))

		if(np.random.rand() > 0.5):
			bg_tensor = TF.hflip(bg_tensor)
			bprime_tensor = TF.hflip(bprime_tensor)

		if(np.random.rand() > 0.5):
			fg_tensor = TF.hflip(fg_tensor)
			alpha_tensor = TF.hflip(alpha_tensor)

		#add shadow augmentation
		do_shadow = np.random.rand() > 0.5

		if do_shadow:
			bg_tensor = self.shadow_augment(bg_tensor, alpha_tensor)

		return fg_tensor, bg_tensor, alpha_tensor, bprime_tensor

	def __len__(self):

		return len(self.bg_fg_combos)

"""
params = {
	
	'bg_dir':'train_set_2/backgrounds/',
	'fg_dir':'train_set_2/foregrounds/',

}


for i in [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]:
	test_dataset = MatteDataset(**params)
	fg, bg, alpha, bprime = test_dataset[i]

	fg = transforms.ToPILImage()(fg)
	fg.save(f'outputs7/{i}fg.jpg')

	bg = transforms.ToPILImage()(bg)
	bg.save(f'outputs7/{i}bg.jpg')

	alpha = transforms.ToPILImage()(alpha)
	alpha.save(f'outputs7/{i}alpha.jpg')

	bprime = transforms.ToPILImage()(bprime)
	bprime.save(f'outputs7/{i}bprime.jpg')
"""