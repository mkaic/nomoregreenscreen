
#print('Loading libraries...')
#torch for obvious reasons, transforms for PIL to Tensor and back, TF for fine control over augmentation,
#Dataset for creating a dataset, os for looping through directories, PIL for I/O of images, numpy for random ints,
#and finally, time for debugging and performance measurement. Itertools for getting all combos of bg and fg
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

	def __init__(self, bg_dir, fg_dir, alpha_dir):
		#_dir --> directory containing folders of images belonging to specific video clips for the background

		self.bg_dir = bg_dir
		self.fg_dir = fg_dir
		self.alpha_dir = alpha_dir

		num_bg_clips = len(os.listdir(bg_dir))
		num_fg_clips = len(os.listdir(fg_dir))
		self.bg_fg_combos = list(itertools.product(range(num_bg_clips), range(num_fg_clips)))

	def augment(self, bg_tensor, fg_tensor, bprime_tensor, alpha_tensor):
		#Expects inputs of size C x H x W

		bg_rot = np.random.randint(-8, 9)
		bg_trans_x = np.random.randint(-100, 100)
		bg_trans_y = np.random.randint(-100, 100)
		bg_shear_x = np.random.randint(-5, 6)
		bg_shear_y = np.random.randint(-5, 6)
		bg_scale = np.random.randint(8, 13) / 10

		bg_brightness = np.random.randint(85, 116) / 100
		bg_contrast = np.random.randint(85, 116) / 100
		bg_saturation = np.random.randint(85, 116) / 100
		bg_hue = np.random.randint(-5, 6) / 100

		bg_blur = np.random.randint(90, 111) / 100
		
		bg_tensor = F.interpolate(bg_tensor.unsqueeze(0), size = fg_tensor.shape[-2:]).squeeze()
		bprime_tensor = F.interpolate(bprime_tensor.unsqueeze(0), size = fg_tensor.shape[-2:]).squeeze()
		

		tick = time.time()
		aug_bg_params = {

			'img': bg_tensor,
			'angle': bg_rot,
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

		aug_bg_tensor = TF.adjust_gamma(aug_bg_tensor, bg_brightness)
		aug_bg_tensor = TF.adjust_contrast(aug_bg_tensor, bg_contrast)
		aug_bg_tensor = TF.adjust_saturation(aug_bg_tensor, bg_saturation)
		aug_bg_tensor = TF.adjust_hue(aug_bg_tensor,  bg_hue)
		bg_gaussian = torch.randn(aug_bg_tensor.shape) * 0.05
		aug_bg_tensor = torch.clamp(aug_bg_tensor + bg_gaussian, 0, 1)
		#aug_bg_tensor = TF.adjust_sharpness(aug_bg_tensor, bg_blur)

		aug_bprime_params = {

			'img': bprime_tensor,
			'angle': bg_rot + np.random.randint(-1, 2),
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

		aug_bprime_tensor = TF.adjust_gamma(aug_bprime_tensor, bg_brightness + np.random.randint(-10, 11) / 100)
		aug_bprime_tensor = TF.adjust_contrast(aug_bprime_tensor, bg_contrast + np.random.randint(-10, 11) / 100)
		aug_bprime_tensor = TF.adjust_saturation(aug_bprime_tensor, bg_saturation + np.random.randint(-10, 11) / 100)
		aug_bprime_tensor = TF.adjust_hue(aug_bprime_tensor, bg_hue + np.random.randint(-3, 4) / 100)
		bprime_gaussian = torch.randn(aug_bprime_tensor.shape) * 0.05
		aug_bprime_tensor = torch.clamp(aug_bprime_tensor + bprime_gaussian, 0, 1)
		#aug_bprime_tensor = TF.adjust_sharpness(aug_bprime_tensor, bg_blur)


		fg_rot = np.random.randint(-8, 9)
		fg_trans_x = np.random.randint(-100, 100)
		fg_trans_y = np.random.randint(-100, 100)
		fg_shear_x = np.random.randint(-5, 6)
		fg_shear_y = np.random.randint(-5, 6)
		fg_scale = np.random.randint(8, 13) / 10

		aug_fg_params = {

			'angle': np.random.randint(-15, 16),
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
		aug_fg_tensor = TF.affine(img=fg_tensor, **aug_fg_params)
		aug_alpha_tensor = TF.affine(img=alpha_tensor, **aug_fg_params)


		aug_fg_tensor = TF.adjust_gamma(aug_fg_tensor, np.random.randint(85, 116) / 100)
		aug_fg_tensor = TF.adjust_contrast(aug_fg_tensor, np.random.randint(85, 116) / 100)
		aug_fg_tensor = TF.adjust_saturation(aug_fg_tensor, np.random.randint(85, 116) / 100)
		aug_fg_tensor = TF.adjust_hue(aug_fg_tensor, np.random.randint(-6, 7) / 100)
		fg_gaussian = torch.randn(aug_fg_tensor.shape) * 0.05
		aug_fg_tensor = torch.clamp(aug_fg_tensor + fg_gaussian, 0, 1)

		return aug_bg_tensor, aug_fg_tensor, aug_bprime_tensor, aug_alpha_tensor

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
		#print(self.bg_fg_combos[index])

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

		#Select some random frames to use as background, foreground, and b-prime
		bg_idx = np.random.randint(0, num_frames_bg)

		fg_idx = np.random.randint(0, num_frames_fg)

		bg_tensor = transforms.ToTensor()(Image.open(bg_clip_dir + bg_frames_list[bg_idx]))
		fg_tensor = transforms.ToTensor()(Image.open(fg_clip_dir + fg_frames_list[fg_idx]))
		alpha_tensor = transforms.ToTensor()(Image.open(alpha_clip_dir + fg_frames_list[fg_idx]))[:1]
		bprime_tensor = bg_tensor.clone()

		bg_tensor, fg_tensor, bprime_tensor, alpha_tensor = self.augment(bg_tensor, fg_tensor, bprime_tensor, alpha_tensor)

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

		if(fg_tensor.shape[-2] < 2160 and fg_tensor.shape[-1] < 3840):
			padding = ((3840 - fg_tensor.shape[-1]) // 2, (3840 - fg_tensor.shape[-1]) // 2, (2160 - fg_tensor.shape[-2]) // 2, (2160 - fg_tensor.shape[-2]) // 2)
			fg_tensor = F.pad(fg_tensor, padding)
			alpha_tensor = F.pad(alpha_tensor, padding)
			bg_tensor = F.pad(bg_tensor, padding)
			bprime_tensor = F.pad(bprime_tensor, padding)

		if(fg_tensor.shape[-2] > 2160 or fg_tensor.shape[-1] > 3840):
			size = (2160, 3840)
			fg_tensor = TF.center_crop(fg_tensor, size)
			alpha_tensor = TF.center_crop(alpha_tensor, size)
			bg_tensor = TF.center_crop(bg_tensor, size)
			bprime_tensor = TF.center_crop(bprime_tensor, size)

		return bg_tensor, fg_tensor, bprime_tensor, alpha_tensor

	def __len__(self):

		return len(self.bg_fg_combos)

"""
params = {
	
	'bg_dir':'dataset/train/bgr/',
	'fg_dir':'dataset/train/fgr/',
	'alpha_dir':'dataset/train/pha/'

}

test_dataset = MatteDataset(**params)



for i in [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]:
	fg, bg, alpha, bprime = test_dataset[i]

	fg = transforms.ToPILImage()(fg)
	fg.save(f'outputs7/{i}fg.jpg')

	bg = transforms.ToPILImage()(bg)
	bg.save(f'outputs7/{i}bg.jpg')

	alpha = transforms.ToPILImage()(alpha)
	alpha.save(f'outputs7/{i}alpha.jpg')

	bprime = transforms.ToPILImage()(bprime)
	bprime.save(f'outputs7/{i}bprime.jpg')


dataloader = DataLoader(test_dataset, num_workers=0, batch_size = 4, pin_memory = True, shuffled = True)

dataloader = iter(dataloader)
tick = time.time()
bg, fg, bprime, alpha = next(dataloader)
tock = time.time()

print(tock-tick)
print(bg.shape, fg.shape, bprime.shape, alpha.shape)
"""