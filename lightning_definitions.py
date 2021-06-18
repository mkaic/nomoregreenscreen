
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import os
from tqdm import tqdm
from coarse_definition import CoarseMatteGenerator
from refine_definition import RefinementNetwork
from dali_dataloader import AugmentationPipeline4K, ImageOpener
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import pytorch_lightning as pl
from train_utils import *
from kornia.filters import sobel
import numpy as np



class LightningModel(pl.core.lightning.LightningModule):

	def __init__(self, train_refine = False, batch_size = 2):

		super().__init__()

		self.coarse = CoarseMatteGenerator()
		self.batch_size = batch_size

		self.L1Loss = nn.L1Loss()
		self.MSELoss = nn.MSELoss()

		self.train_refine = train_refine
		self.refine = RefinementNetwork()

	def forward(self, coarse_input):

		#THIS basically acts as a model for what the model should do at INFERENCE time

		#Generate a fake coarse alpha, along with a guessed error map and some hidden channel data. Oh yeah and the foreground residual

		fake_coarse = self.coarse(coarse_input)

		fake_coarse_alpha = torch.clamp(fake_coarse[:, 0:1], 0, 1)
		fake_coarse_foreground_residual = fake_coarse[:, 1:4]
		fake_coarse_error = torch.clamp(fake_coarse[:, 4:5], 0, 1)
		fake_coarse_hidden_channels = torch.relu(fake_coarse[:,5:])

		downsampled_input_tensor = F.interpolate(input_tensor, [input_tensor.shape[-2]//2, input_tensor.shape[-1]//2])
		upscaled_coarse_outputs = F.interpolate(fake_coarse, [input_tensor.shape[-2]//2, input_tensor.shape[-1]//2])
		start_patch_source = torch.cat([downsampled_input_tensor, upscaled_coarse_outputs], 1)

		start_patches, indices = get_image_patches(start_patch_source.detach(), fake_coarse_error.detach(), patch_size = 8, stride = 2, k = 5000)
		middle_patches, _ = get_image_patches(input_tensor.detach(), fake_coarse_error.detach(), patch_size = 8, stride = 4, k = 5000)

		#Now, feed the outputs of the coarse generator into the refinement network, which will refine patches.
		fake_refined_patches = self.refine(start_patches, middle_patches)

		mega_upscaled_fake_coarse = F.interpolate(fake_coarse[:, :4].detach(), size = input_tensor.shape[-2:], mode = 'bilinear', align_corners = True)
		fake_refined = replace_image_patches(images = mega_upscaled_fake_coarse, patches = fake_refined_patches, indices = indices)
		fake_refined_alpha = color_ramp(0.05, 0.95, torch.clamp(fake_refined[:, 0:1], 0, 1))
		fake_refined_foreground = torch.clamp(fake_refined[:, 1:4] + composite_tensor, 0, 1)

		return (fake_refined_alpha, fake_refined_foreground)

	def configure_optimizers(self):

		if(not self.train_refine):

			coarse_parameters = [\
				{'params': self.coarse.Encoder.parameters(), 'lr': 0.0001},\
				{'params': self.coarse.ASPP.parameters(), 'lr': 0.0005},\
				{'params': self.coarse.Decoder.parameters(), 'lr': 0.0005}\
				]


			coarse_opt = torch.optim.Adam(coarse_parameters, lr = 0.0001)

			return coarse_opt

		if(self.train_refine):

			parameters = [\
				{'params': self.coarse.Encoder.parameters(), 'lr': 0.00005},\
				{'params': self.coarse.ASPP.parameters(), 'lr': 0.00005},\
				{'params': self.coarse.Decoder.parameters(), 'lr': 0.0001},\
				{'params': self.refine.parameters(), 'lr': 0.0003}\
				]


			opt = torch.optim.Adam(parameters, lr = 0.0001)

			return opt


	def training_step(self, batch, batch_idx):

		real_background, real_foreground, real_bprime, real_alpha = batch

		#Composite the augmented foreground onto the augmented background according to the augmented alpha.
		composite_tensor = composite(real_background, real_foreground, real_alpha)

		#return the input tensor (composite plus b-prime) and the alpha_tensor. The input tensor is just a bunch of channels, the real_alpha is the central (singular) alpha
		#corresponding to the target frame.
		input_tensor = torch.cat([composite_tensor, real_bprime], 1)

		coarse_input = F.interpolate(input_tensor, size = [input_tensor.shape[-2]//4, input_tensor.shape[-1]//4])

		#Get a downsampled version of the alpha for grading the coarse network on
		real_coarse_alpha = F.interpolate(real_alpha, size = [real_alpha.shape[-2]//4, real_alpha.shape[-1]//4])

		#Generate a fake coarse alpha, along with a guessed error map and some hidden channel data. Oh yeah and the foreground residual

		fake_coarse = self.coarse(coarse_input)

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
		self.L1Loss(fake_coarse_alpha,real_coarse_alpha) + \
		self.MSELoss(fake_coarse_error,real_coarse_error) + \
		self.L1Loss((real_coarse_foreground * foreground_penalty_zone), (fake_coarse_foreground * foreground_penalty_zone)) + \
		self.L1Loss(coarse_sobel,real_sobel)

		if(not self.train_refine):

			loss = coarse_loss

		if(self.train_refine):

			downsampled_input_tensor = F.interpolate(input_tensor, [input_tensor.shape[-2]//2, input_tensor.shape[-1]//2])
			upscaled_coarse_outputs = F.interpolate(fake_coarse, [input_tensor.shape[-2]//2, input_tensor.shape[-1]//2])
			start_patch_source = torch.cat([downsampled_input_tensor, upscaled_coarse_outputs], 1)

			start_patches, indices = get_image_patches(start_patch_source.detach(), fake_coarse_error.detach(), patch_size = 8, stride = 2, k = 5000)
			middle_patches, _ = get_image_patches(input_tensor.detach(), fake_coarse_error.detach(), patch_size = 8, stride = 4, k = 5000)

			#Now, feed the outputs of the coarse generator into the refinement network, which will refine patches.
			fake_refined_patches = self.refine(start_patches, middle_patches)

			mega_upscaled_fake_coarse = F.interpolate(fake_coarse[:, :4].detach(), size = input_tensor.shape[-2:], mode = 'bilinear', align_corners = True)
			fake_refined = replace_image_patches(images = mega_upscaled_fake_coarse, patches = fake_refined_patches, indices = indices)
			fake_refined_alpha = color_ramp(0.05, 0.95, torch.clamp(fake_refined[:, 0:1], 0, 1))
			fake_refined_foreground = torch.clamp(fake_refined[:, 1:4] + composite_tensor, 0, 1)

			refine_loss = \
			torch.mean(torch.abs(fake_refined_alpha - real_alpha.detach())) + \
			torch.mean(torch.abs(fake_refined_foreground - real_foreground.detach()))

			loss = coarse_loss + refine_loss

		if(batch_idx % 1000 == 0):
			image = fake_coarse_alpha[0]
			image = transforms.ToPILImage()(image)
			image.save(f'outputs7/{batch_idx}C_fake_coarse_alpha.jpg')

			image = real_coarse_alpha[0]
			image = transforms.ToPILImage()(image)
			image.save(f'outputs7/{batch_idx}B_real_alpha.jpg')

			image = fake_coarse_foreground[0]
			image = transforms.ToPILImage()(image)
			image.save(f'outputs7/{batch_idx}A_fake_foreground.jpg')

			image = fake_coarse_error[0]
			image = transforms.ToPILImage()(image)
			image.save(f'outputs7/{batch_idx}D_fake_error.jpg')

			image = real_coarse_composite[0]
			image = transforms.ToPILImage()(image)
			image.save(f'outputs7/{batch_idx}G_real_coarse_composite.jpg')


			if(self.train_refine):

				image = fake_refined_alpha[0]
				image = transforms.ToPILImage()(image)
				image.save(f'outputs7/{batch_idx}E_refined_alpha.jpg')

				image = fake_refined_foreground[0]
				image = transforms.ToPILImage()(image)
				image.save(f'outputs7/{batch_idx}F_refined_foreground.jpg')


		return loss


	def prepare_data(self):

		batch_size = self.batch_size

		#Initialize the dataset and the loader to feed the data to the network.

		ImageFeeder = ImageOpener(\
			fg_dir = 'dataset/train/fgr/', \
			bg_dir = 'dataset/train/bgr/', \
			alpha_dir = 'dataset/train/pha/',
			batch_size = batch_size)

		Pipeline4K = AugmentationPipeline4K(\
			dataset = ImageFeeder, \
			num_threads = 4, \
			device_id = 0, \
			batch_size = batch_size)

		Pipeline4K.build()

		class LightningWrapper(DALIGenericIterator):

			def __init__(self, *kargs, **kvargs):
				super().__init__(*kargs, **kvargs)

			def __next__(self):

				tensor_dict = super().__next__()[0]

				fg = tensor_dict['fg'].permute(0,3,1,2)
				bg = tensor_dict['bg'].permute(0,3,1,2)
				bprime = tensor_dict['bprime'].permute(0,3,1,2)
				alpha = tensor_dict['alpha'].permute(0,3,1,2)
				alpha = alpha[:, :1]

				bg = bg.float()/256
				fg = fg.float()/256
				bprime = bprime.float()/256
				alpha = alpha.float()/256

				if(np.random.randint(0, 101) < 30):

					bg = F.interpolate(bg, size = [bg.shape[-2]//2,bg.shape[-1]//2])
					fg = F.interpolate(fg, size = [fg.shape[-2]//2,fg.shape[-1]//2])
					bprime = F.interpolate(bprime, size = [bprime.shape[-2]//2,bprime.shape[-1]//2])
					alpha = F.interpolate(alpha, size = [alpha.shape[-2]//2,alpha.shape[-1]//2])


				crop_chance = np.random.randint(0, 101) > 25

				crop_w = 3840
				crop_h = 2160

				if(crop_chance):
					crop_w = np.random.randint(bg.shape[-1]//2.5, bg.shape[-1])
					crop_h = np.random.randint(bg.shape[-2]*2//3, bg.shape[-2])

				bg = TF.center_crop(bg, output_size = [crop_h, crop_w])
				fg = TF.center_crop(fg, output_size = [crop_h, crop_w])
				bprime = TF.center_crop(bprime, output_size = [crop_h, crop_w])
				alpha = TF.center_crop(alpha, output_size = [crop_h, crop_w])

				png = torch.cat([fg, alpha], dim = 1)

				bg_trans_x = np.random.randint(-100, 100)
				bg_trans_y = np.random.randint(-100, 100)
				bg_shear_x = np.random.randint(-5, 6)
				bg_shear_y = np.random.randint(-5, 6)
				bg_scale = np.random.randint(10, 13) / 10

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
						bg_trans_x + np.random.randint(-10, 11),
						bg_trans_y + np.random.randint(-10, 11)
					],
					'shear':[
						bg_shear_x + np.random.randint(-5, 6),
						bg_shear_y + np.random.randint(-5, 6)
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
				
				aug_fg_tensor = aug_png_tensor[:, :3]
				aug_alpha_tensor = aug_png_tensor[:, 3:4]


				if(np.random.randint(0, 10) > 6):
					shadow_x = np.random.randint(0, 200)
					shadow_y = np.random.randint(0, 200)
					shadow_shear = np.random.randint(-30, 30)
					shadow_rotation = np.random.randint(-30, 30)
					shadow_strength = np.random.randint(20, 80) / 100
					shadow_blur = np.random.randint(2, 16) * 2 + 1

					shadow_stamp = TF.affine(aug_alpha_tensor, translate = [shadow_x, shadow_y], shear = shadow_shear, angle = shadow_rotation, scale = 1)
					shadow_stamp = TF.gaussian_blur(shadow_stamp, shadow_blur)
					shadow_stamp = shadow_stamp * shadow_strength

					aug_bg_tensor = aug_bg_tensor - (aug_bg_tensor * shadow_stamp)

				return(aug_bg_tensor, aug_fg_tensor, aug_bprime_tensor, aug_alpha_tensor)

			def __len__(self):

				return(25000)

		self.DALIDataloader = LightningWrapper(pipelines = [Pipeline4K], output_map = ['bg', 'fg', 'bprime', 'alpha'])

		return self.DALIDataloader

	def train_dataloader(self):

		return self.DALIDataloader


if 'PL_TRAINER_GPUS' in os.environ:
	os.environ.pop('PL_TRAINER_GPUS')

model = LightningModel(train_refine = False, batch_size = 1)

trainer = pl.Trainer(gpus=1, distributed_backend='ddp', default_root_dir='./model_saves', max_epochs = 100)

trainer.fit(model)





