
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class IdentityBlock(nn.Module):
	#Inputs a tensor, convolves/activates it twice, then adds it to the original input version of itself (residual block)

	def __init__(self, channels):
		super().__init__()

		self.activation = nn.ReLU()
		self.conv1 = nn.Conv2d(channels, channels, kernel_size = 1)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size = 5)
		self.skipconv = nn.Conv2d(channels, channels, kernel_size = 5)

		self.bn = nn.BatchNorm2d(channels)

	def forward(self, X):

		skipped_X = X

		X = self.conv1(X)
		X = self.activation(X)
		X = self.conv2(X)

		skipped_X = self.skipconv(skipped_X)
	
		X = torch.add(X, skipped_X)
		X = self.bn(X)
		X = self.activation(X)

		return X


class SkipConnDownChannel(nn.Module):

	def __init__(self, input_channels, output_channels, final = False):
		super().__init__()

		self.input_channels = input_channels
		self.activation = nn.ReLU()
		self.final = final

		self.input_down_channel = nn.Conv2d(input_channels, input_channels//2, kernel_size = 1)
		self.skip_down_channel = nn.Conv2d(input_channels, input_channels//2, kernel_size = 1)

		self.concatenated_down_channel = nn.Conv2d(input_channels, input_channels//2, kernel_size = 5)
		self.concatenated_conv = nn.Conv2d(input_channels//2, output_channels, kernel_size = 5)
		self.bn = nn.BatchNorm2d(output_channels)


	def forward(self, X, skipped_X):

		X = self.input_down_channel(X)
		X = self.activation(X)

		skipped_X = self.skip_down_channel(skipped_X)
		skipped_X = self.activation(skipped_X)

		skipped_X = F.interpolate(skipped_X, size = [X.shape[-2], X.shape[-1]])
		concatenated = torch.cat([X, skipped_X], 1)

		concatenated = self.concatenated_down_channel(concatenated)
		concatenated = self.activation(concatenated)
		concatenated = self.concatenated_conv(concatenated)
		concatenated = self.bn(concatenated)
		

		if(not self.final):
			
			concatenated = self.activation(concatenated)

		return concatenated