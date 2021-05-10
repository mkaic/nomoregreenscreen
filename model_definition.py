
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
from tqdm import tqdm
from train_utils import ResidualBlock, DecoderBlock


class CoarseMatteGenerator(nn.Module):

	def __init__(self, input_channels = 6, output_channels = 37):
		super().__init__()
		#enters at 540 x 960
		self.conv1 = nn.Conv2d(input_channels, 64, kernel_size = 7, stride = 2, padding = 3)
		self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

		#now 270 x 480
		self.res1 = ResidualBlock(64, 64, 256, projection = True, downsample = True)
		self.res2 = ResidualBlock(256, 64, 256)
		self.res3 = ResidualBlock(256, 64, 256)

		#135 x 240
		self.res4 = ResidualBlock(256, 128, 512, projection = True, downsample = True)
		self.res5 = ResidualBlock(512, 128, 512)
		self.res6 = ResidualBlock(512, 128, 512)
		self.res7 = ResidualBlock(512, 128, 512)

		#68 x 120
		self.res8 = ResidualBlock(512, 256, 1024, projection = True, downsample = True)
		self.res9 = ResidualBlock(1024, 256, 1024)
		self.res10 = ResidualBlock(1024, 256, 1024)
		self.res11 = ResidualBlock(1024, 256, 1024)
		self.res12 = ResidualBlock(1024, 256, 1024)
		self.res13 = ResidualBlock(1024, 256, 1024)

		#34 x 60
		self.res14 = ResidualBlock(1024, 512, 1024, dilation = 2, projection = True, downsample = True)
		self.res15 = ResidualBlock(1024, 512, 1024, dilation = 2)
		self.res16 = ResidualBlock(1024, 512, 1024, dilation = 2)

		self.aspp1 = nn.Conv2d(1024, 256, kernel_size = 1)
		self.bn1 = nn.BatchNorm2d(256)
		self.aspp2 = nn.Conv2d(1024, 256, kernel_size = 3, dilation = 3, padding = 3)
		self.bn2 = nn.BatchNorm2d(256)
		self.aspp3 = nn.Conv2d(1024, 256, kernel_size = 3, dilation = 6, padding = 6)
		self.bn3 = nn.BatchNorm2d(256)
		self.aspp4 = nn.Conv2d(1024, 256, kernel_size = 3, dilation = 9, padding = 9)
		self.bn4 = nn.BatchNorm2d(256)

		self.activation = nn.ReLU()
		
		self.global_average = nn.AvgPool2d(kernel_size = [5, 8])
		self.global_feature_conv = nn.Conv2d(1024, 256, kernel_size = 1)
		self.bn5 = nn.BatchNorm2d(256)

		self.concat_crunch = nn.Conv2d(1280, 256, kernel_size = 1)

		#We'll upsample first back up to 68 x 120
		self.decode1 = DecoderBlock(256 + 512, 128)
		#then 135 x 240
		self.decode2 = DecoderBlock(128 + 256, 64)
		#then 270 x 480
		self.decode3 = DecoderBlock(64 + 64, 48)

		#and finally to 540 x 960
		self.final_conv = nn.Conv2d(48 + 6, output_channels, kernel_size = 3, padding = 1)


	def forward(self, input_tensor):

		size4 = input_tensor.shape[:-2]
		size8 = [x // 2 for x in size4]
		size16 = [x // 2 for x in size8]
		size32 = [x // 2 for x in size16]
		size64 = [x // 2 for x in size32]

		X = self.conv1(input_tensor)
		X = self.max_pool(X)

		X1 = self.res1(X)
		X1 = self.res2(X1)
		X1 = self.res3(X1)

		X2 = self.res4(X1)
		X2 = self.res5(X2)
		X2 = self.res6(X2)
		X2 = self.res7(X2)

		X3 = self.res8(X2)
		X3 = self.res9(X3)
		X3 = self.res10(X3)
		X3 = self.res11(X3)
		X3 = self.res12(X3)
		X3 = self.res13(X3)

		X4 = self.res14(X3)
		X4 = self.res15(X4)
		X4 = self.res16(X4)

		aspp1 = self.aspp1(X4)
		aspp2 = self.aspp2(X4)
		aspp3 = self.aspp3(X4)
		aspp4 = self.aspp4(X4)

		aspp1 = self.bn1(aspp1)
		aspp2 = self.bn2(aspp2)
		aspp3 = self.bn3(aspp3)
		aspp4 = self.bn4(aspp4)

		aspp1 = self.activation(aspp1)
		aspp2 = self.activation(aspp2)
		aspp3 = self.activation(aspp3)
		aspp4 = self.activation(aspp4)

		global_features = self.global_average(X4)
		global_features = F.interpolate(global_features, size = aspp1.shape[-2:], mode = 'bilinear', align_corners = True)
		global_features = self.global_feature_conv(global_features)
		global_features = self.bn5(global_features)

		aspp_concat = torch.cat([aspp1, aspp2, aspp3, aspp4, global_features], dim = 1)
		aspp_result = self.concat_crunch(aspp_concat)
		aspp_result = self.activation(aspp_result)

		decode1 = F.interpolate(aspp_result, size = X2.shape[-2:], mode = 'bilinear', align_corners = True)
		decode1 = torch.cat([decode1, X2], dim = 1)
		decode1 = self.decode1(decode1)

		decode2 = F.interpolate(decode1, size = X1.shape[-2:], mode = 'bilinear', align_corners = True)
		decode2 = torch.cat([decode2, X1], dim = 1)
		decode2 = self.decode2(decode2)

		decode3 = F.interpolate(decode2, size = X.shape[-2:], mode = 'bilinear', align_corners = True)
		decode3 = torch.cat([decode3, X], dim = 1)
		decode3 = self.decode3(decode3)

		final_logits = F.interpolate(decode3, size = input_tensor.shape[-2:], mode = 'bilinear', align_corners = True)
		final_logits = torch.cat([final_logits, input_tensor], dim = 1)
		final_logits = self.final_conv(final_logits)
	
		return final_logits



class RefinementNetwork(nn.Module):

	def __init__(self, coarse_channels = 37, input_channels = 6, patch_size = 8):
		super().__init__()

		self.input_channels = input_channels
		self.coarse_channels = coarse_channels
		self.concat_channels = self.input_channels + self.coarse_channels

		#subtracting one from the input channels here because we aren't using the error map as an input
		self.conv1 = nn.Conv2d(self.concat_channels, 24, kernel_size = 3)
		self.conv2 = nn.Conv2d(24, 16, kernel_size = 3)
		self.conv3 = nn.Conv2d(16 + self.input_channels, 12, kernel_size = 3)
		self.conv4 = nn.Conv2d(12, 4, kernel_size = 3)

		self.activation = nn.ReLU()


	def forward(self, start_patches, middle_patches):

		z1 = self.conv1(start_patches)
		x1 = self.activation(z1)

		z2 = self.conv2(x1)
		x2 = self.activation(z2)
		x2 = F.interpolate(x2, size = middle_patches.shape[-2:])

		z3 = torch.cat([x2, middle_patches], 1)
		z3 = self.conv3(z3)
		x3 = self.activation(z3)

		z4 = self.conv4(x3)

		end_patches = torch.sigmoid(z4)

		return end_patches

