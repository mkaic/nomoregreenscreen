
import torch
import torch.nn as nn
import torch.nn.functional as F

def augment(bg_tensor, fg_tensor, bprime_tensor):
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

	_, bg_h, bg_w = bg_tensor.shape

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
	#aug_bprime_tensor = TF.adjust_sharpness(aug_bprime_tensor, bg_blur)


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
		'scale': np.random.randint(3, 15) / 10

	}
	aug_png_tensor = TF.affine(**aug_png_params)
	
	aug_fg_tensor = aug_png_tensor[:3, :, :]
	aug_alpha_tensor = aug_png_tensor[3:4, :, :]

	bg_gaussian = torch.cuda.FloatTensor(aug_bg_tensor.shape).normal_() * 0.05
	fg_gaussian = torch.cuda.FloatTensor(aug_fg_tensor.shape).normal_() * 0.05
	bprime_gaussian = torch.cuda.FloatTensor(aug_bprime_tensor.shape).normal_() * 0.05

	aug_fg_tensor = torch.clamp(aug_fg_tensor + fg_gaussian, 0, 1)
	aug_bg_tensor = torch.clamp(aug_bg_tensor + bg_gaussian, 0, 1)
	aug_bprime_tensor = torch.clamp(aug_bprime_tensor + bprime_gaussian, 0, 1)

	return aug_fg_tensor, aug_bg_tensor, aug_alpha_tensor, aug_bprime_tensor



#This is a simple utility function for grabbing a square patch of an image tensor of dimensions N x C x H x W.
def get_image_patches(images, error_maps, k, patch_size = 6, stride = 4):

	#store the original shape
	b, c, h, w = error_maps.shape
	#flatten (except batch dim) the error map (this is so that topk works right)
	err = error_maps.view(b, -1)
	#find the highest k values in the error map for each.
	indices = err.topk(k, dim = 1, sorted = False).indices
	##DEBUG##print("indices shape", indices.shape)
	#now we make a tensor of zeros shaped like the flattened error maps
	ref = torch.zeros_like(err)
	#we make the values at the indices of all the top k be 1 so that we can recover those indices easily later.
	ref.scatter_(1, indices, 1.)
	##DEBUG##print("ref shape", ref.shape)
	#we reshape the 0s and 1s reference tensor back into the original shape of the error maps.
	ref = ref.view(b,1,h,w)
	##DEBUG##print("reshaped ref", ref.shape)
	#we find where the ones are, these 
	low_confidence_idx = torch.nonzero(ref.squeeze(1))
	##DEBUG##print("low confidence shape", low_confidence_idx.shape)

	#we get separate lists now for the Batch coord, Y coord, and X coord of each low-confidence pixel. Total of batch_size * k in each of those lists
	#importantly, B[###] lines up with Y[###] and X[###], so that's nice. This way we can to 
	B, Y, X = low_confidence_idx[:,0], low_confidence_idx[:,1], low_confidence_idx[:,2]
	##DEBUG##print("B shape", B.shape)

	#now that we have the indices of the patches, we need to actually make the patches using .unfold()
	#first, the channels dimension needs to be the last one because... reasons? I'm really not sure tbh why this is necessary.
	padding = (patch_size - stride) // 2
	padded = F.pad(images, (padding, padding, padding, padding))
	permuted = padded.permute(0,2,3,1)
	unfolded = permuted.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
	patches = unfolded[B, Y, X]

	return patches, low_confidence_idx


#lol = get_image_patches(images = torch.rand(4,20,480,640, device = device), error_maps = torch.rand(4,1,480,640, device = device), k = 10000)
#print(lol.shape)

def replace_image_patches(images, patches, indices):


	B, Y, X = indices[:,0], indices[:,1], indices[:,2]
	imageB, imageC, imageY, imageX = images.shape
	##DEBUG##print(images.shape)
	patchesP, patchesC, patchesX, patchesY = patches.shape
	##DEBUG##print(patches.shape)
	##DEBUG##print(indices.shape)
	##DEBUG##print(B.shape, Y.shape, X.shape)

	#Now we do some wild reshaping. First, the image is turned from N x C x H x W into N x #VertPatches x PatchSize x #HorizPatches x PatchSize x C...
	#This turns the image into a bunch of PatchSize x PatchSize patches organized by nearly the same indices as were used to get the patches.
	images = images.view(imageB, imageC, (imageY//patchesY), patchesY, (imageX//patchesX), patchesX)
	##DEBUG##print('\n')
	##DEBUG##print(images.shape)
	#This permutation gets the patches organized into N x PatchY x PatchX x PatchHeight x PatchWidth x Channels
	images = images.permute(0,2,4,1,3,5)
	##DEBUG##print(images.shape)

	images[B, Y, X] = patches

	#Now we undo the permutation...
	images = images.permute(0,3,1,4,2,5)
	#And reshape the patches back into an image (undoing the original expansion into patches)
	patched_image = images.view(imageB, imageC, imageY, imageX)

	return patched_image

def color_ramp(a, b, image):

	return torch.clamp(((1/(b - a)) * image) + (a/(a-b)), 0, 1)

def composite(bg_tensor, fg_tensor, alpha_tensor):

		composite = (alpha_tensor * fg_tensor) + ((1 - alpha_tensor) * bg_tensor)

		return composite

class ResidualBlock(nn.Module):

	def __init__(self, \
		input_channels, bottleneck_channels, output_channels,\
		dilation = None, projection = False, downsample = False):

		super().__init__()


		self.intake = nn.Conv2d(input_channels, bottleneck_channels, kernel_size = 1, stride = 2 if downsample else 1)
		self.bn1 = nn.BatchNorm2d(bottleneck_channels)

		if(dilation == None):
			self.crunch = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size = 3, padding = 1)
		else:
			self.crunch = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size = 3,\
			dilation = dilation, padding = dilation)
		
		self.bn2 = nn.BatchNorm2d(bottleneck_channels)
		self.outlet = nn.Conv2d(bottleneck_channels, output_channels, kernel_size = 1)
		self.bn3 = nn.BatchNorm2d(output_channels)

		self.projection = projection
		if(self.projection):
			self.projection_conv = nn.Conv2d(input_channels, output_channels, kernel_size = 1, stride = 2 if downsample else 1)

		self.activation = nn.ReLU()

	def forward(self, X):

		skipX = X

		X = self.intake(X)
		X = self.bn1(X)
		X = self.activation(X)

		X = self.crunch(X)
		X = self.bn2(X)
		X = self.activation(X)

		X = self.outlet(X)
		X = self.bn3(X)

		if(self.projection):
			skipX = self.projection_conv(skipX)

		X = X + skipX

		X = self.activation(X)

		return X


class DecoderBlock(nn.Module):

	def __init__(self, input_channels, output_channels):
		super().__init__()

		self.conv = nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1, bias = False)
		self.bn = nn.BatchNorm2d(output_channels)
		self.activation = nn.ReLU()

	def forward(self, X):

		X = self.conv(X)
		X = self.bn(X)
		X = self.activation(X)

		return X