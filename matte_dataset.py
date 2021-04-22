
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

	def __init__(self, bg_dir, fg_dir, comp_context_depth, comp_context_stride, bprime_context_depth, bprime_context_stride):
		#_dir --> directory containing folders of images belonging to specific video clips for the background
		#_context_depth --> how many frames on either side of the center frame to fetch
		#_context_stride --> how many frames to space out the frames being selected for the motion cue packet

		self.bg_dir = bg_dir
		self.fg_dir = fg_dir

		num_bg_clips = len(os.listdir(bg_dir))
		num_fg_clips = len(os.listdir(fg_dir))
		self.bg_fg_combos = list(itertools.product(range(num_bg_clips), range(num_fg_clips)))

		self.comp_context_depth = comp_context_depth
		self.comp_context_stride = comp_context_stride

		self.bprime_context_depth = bprime_context_depth
		self.bprime_context_stride = bprime_context_stride


	def get_frames_tensor(self, clip_dir, center_frame, context_depth, context_stride):

		#Retrieves a center frame plus some number of frames on either side of it temporally dilated
		#by some factor as a tensor of shape N x C x H x W

		#Get a list of the filenames of the frames
		frames_list = os.listdir(clip_dir)
		#Count how many frames are in the list.
		num_frames = len(frames_list)

		#Based on the margin needed for temporal context, calculate a buffer representing the minimum
		#distance the loop has to stay from the start and end of the frames to avoid referencing frames
		#that don't exist
		loop_offset = context_depth * context_stride

		#the random_frame_index references the frame at the center of the packet, this index references 
		#the one to start the collection loop at.
		start_frame_index = center_frame - loop_offset

		#this is the list that will store the names of the frames needed for the context packet to be constructed
		frame_packet = []

		#loop through the number of frames needed for the context packet.
		for frame in range(context_depth * 2 + 1):

			#step at a rate of self.bg_context_stride through the frames
			file = frames_list[start_frame_index + (frame * context_stride)]
			#get the filename of the current JPG frame being added to the cache
			filename = os.fsdecode(file)
			#open the frame as a PIL image
			image = Image.open(clip_dir + filename)
			#Turn the PIL image into a C x H x W tensor
			image = transforms.ToTensor()(image)
			#Add that image to the list we're using.
			frame_packet.append(image)

		#Convert the list of tensors into one big tensor
		frames_tensor = torch.stack(frame_packet, 0)

		return frames_tensor

	def augment(self, bg_tensor, fg_tensor):

		#All input tensors assumed to be single examples with dimensions Time x Channels x Height x Width

		#Now, it's time to take the foreground, background, and time-shifted bprime, and turn them into the composite, background-prime, and alpha
		#Step one is to create a pair of two sets of strongly correlated (but not identical) Affine and Perspective transforms-- one for
		#the background, one for the foreground. This helps correlate the two of them like they would be IRL.

		#Correlated transforms first, methinks. transform comes first, followed by how much it will linearly change per frame in the packet.
		corr_rot = np.random.randint(-10, 11)
		corr_rot_rate = np.random.randint(-3, 4)

		corr_trans_x = np.random.randint(-100, 101)
		corr_trans_x_rate = np.random.randint(-40, 41)

		corr_trans_y = np.random.randint(-100, 101)
		corr_trans_y_rate = np.random.randint(-40, 41)

		corr_shear_x = np.random.randint(-15, 16)
		corr_shear_x_rate = np.random.randint(-3, 4)

		corr_shear_y = np.random.randint(-15, 16)
		corr_shear_y_rate = np.random.randint(-3, 4)

		correlation_factor = ((np.random.random() - 1) * 0.4) + 1

		rand_fg_rot_offset = np.random.randint(-40, 40)

		#initialize empty tensors of the right shape to hold the modified outputs.
		corr_bg_tensor = torch.zeros_like(bg_tensor)
		corr_png_tensor = torch.zeros_like(fg_tensor)

		fg_shape = fg_tensor.shape
		corr_fg_tensor = torch.zeros(fg_shape[0], 3, fg_shape[2], fg_shape[3])
		corr_alpha_tensor = torch.zeros(fg_shape[0], 1, fg_shape[2], fg_shape[3])

		do_horizontal_flip = np.random.rand() > 0.5

		for idx in range(bg_tensor.shape[0]):

			corr_bg_params = {

				'img': bg_tensor[idx, :, :, :],
				'angle': corr_rot + idx * corr_rot_rate,
				'translate':[
					corr_trans_x + idx * corr_trans_x_rate,
					corr_trans_y + idx * corr_trans_y_rate
				],
				'shear':[
					corr_shear_x + idx * corr_shear_x_rate,
					corr_shear_y + idx * corr_shear_y_rate
				],
				'scale':1.3

			}
			corr_bg_tensor[idx, :, :, :] = TF.affine(**corr_bg_params)
			if(do_horizontal_flip):
				corr_bg_tensor[idx, :, :, :] = TF.hflip(corr_bg_tensor[idx, :, :, :])
		


			corr_png_params = {

				'img': fg_tensor[idx, :, :, :],
				'angle': (corr_rot + idx * corr_rot_rate) * correlation_factor + rand_fg_rot_offset,
				'translate':[
					(corr_trans_x + idx * corr_trans_x_rate) * correlation_factor,
					(corr_trans_y + idx * corr_trans_y_rate) * correlation_factor
				],
				'shear':[
					(corr_shear_x + idx * corr_shear_x_rate) * correlation_factor,
					(corr_shear_y + idx * corr_shear_y_rate) * correlation_factor
				],
				'scale':1.2

			}
			corr_png_tensor[idx, :, :, :] = TF.affine(**corr_png_params)
			if(do_horizontal_flip):
				corr_png_tensor[idx, :, :, :] = TF.hflip(corr_png_tensor[idx, :, :, :])
			
			corr_fg_tensor[idx, :, :, :] = corr_png_tensor[idx, :3, :, :]
			corr_alpha_tensor[idx, :, :, :] = corr_png_tensor[idx, 3:4, :, :]



		#add shadow augmentation
		do_shadow = np.random.rand() > 0.6

		if do_shadow:
			corr_bg_tensor = self.shadow_augment(corr_bg_tensor, corr_alpha_tensor)

		"""
		#composite the warped foreground onto the warped background according to the warped alpha
		composite_tensor = self.composite(corr_bg_tensor, corr_fg_tensor, corr_alpha_tensor)

		#generate the "channel-block" version of the input, which no longer contains separate images and is just a blob of channels.
		input_tensor = torch.cat([composite_tensor, bprime_tensor], dim = 0)
		alpha_tensor = corr_alpha_tensor
		"""
	

		return corr_fg_tensor, corr_bg_tensor, corr_alpha_tensor

	#Does what it says on the box. Takes in a background, foreground, and alpha, and composites them into one image accordingly.

	def shadow_augment(self, bg_tensor, alpha_tensor):

		#randomize a slight affine transform to make sure the shadow is offset from the subject.
		shadow_x = np.random.randint(0, 200)
		shadow_y = np.random.randint(0, 200)
		shadow_shear = np.random.randint(-30, 30)
		shadow_rotation = np.random.randint(-30, 30)
		shadow_strength = np.random.randint(40, 90) / 100
		shadow_blur = np.random.randint(2, 10) * 2 + 1

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
		num_frames_bg = len(os.listdir(bg_clip_dir))
		num_frames_fg = len(os.listdir(fg_clip_dir))

		#Based on the margin needed for temporal context, calculate a buffer representing the minimum
		#distance the loop has to stay from the start and end of the frames to avoid referencing frames
		#that don't exist
		comp_loop_offset = self.comp_context_depth * self.comp_context_stride + 1

		#pick random center frame indexes within the range defined by the number of frames and the margin that
		#needs to be kept to prevent errors.
		bg_center = np.random.randint(comp_loop_offset, num_frames_bg - comp_loop_offset)

		#how much bprime should be offset in either direction temporally from b.
		rand_bprime_offset = np.random.randint(1, 5)
		bprime_loop_offset = comp_loop_offset + rand_bprime_offset

		#Deal with the edge case where the bg center frame is too close to the margin, in which case, only temporally shift to one side.
		#Otherwise, select bprime as a set of frames nearby but not identical to b.
		if(bg_center <= bprime_loop_offset):
			bprime_center = np.random.randint(bprime_loop_offset, bprime_loop_offset + rand_bprime_offset)
		elif(bg_center >= num_frames_bg - bprime_loop_offset):
			bprime_center = np.random.randint(num_frames_bg - bprime_loop_offset - rand_bprime_offset, num_frames_bg - bprime_loop_offset)
		else:
			bprime_center = np.random.randint(bg_center - rand_bprime_offset, bg_center + rand_bprime_offset)

		#Select a random frame to act as the center of the foreground frame packet.
		fg_center = np.random.randint(comp_loop_offset, num_frames_fg - comp_loop_offset)


		#retrieve frame packets for the background and background-prime with the context depth and stride as passed to __init__()
		bg_tensor = self.get_frames_tensor(bg_clip_dir, bg_center, self.comp_context_depth, self.comp_context_stride)
		bprime_tensor = self.get_frames_tensor(bg_clip_dir, bprime_center, self.bprime_context_depth, self.bprime_context_stride)

		#retrieve frame packets for the foreground and alpha with the context depth and stride as passed to __init__()
		fg_tensor = self.get_frames_tensor(fg_clip_dir, fg_center, self.comp_context_depth, self.comp_context_stride)

		fg_tensor, bg_tensor, alpha_tensor = self.augment(bg_tensor, fg_tensor)

		

		return fg_tensor, bg_tensor, alpha_tensor, bprime_tensor

	def __len__(self):

		return len(self.bg_fg_combos)


"""
params = {
	
	'bg_dir':'train_set/backgrounds/',
	'fg_dir':'train_set/foregrounds/',
	'comp_context_depth': 1,
	'comp_context_stride': 2,
	'bprime_context_depth': 1,
	'bprime_context_stride': 2

}

test_dataset = MatteDataset(**params)
test_example, test_alpha = test_dataset[400]

for i in range(test_example.shape[0]):
	image = test_example[i, :, :, :]
	image = transforms.ToPILImage()(image)
	image.save(f'outputs/in{i}.jpg')

for i in range(test_alpha.shape[0]):
	image = test_alpha[i, :, :, :]
	image = transforms.ToPILImage()(image)
	image.save(f'outputs/alpha{i}.jpg')


"""