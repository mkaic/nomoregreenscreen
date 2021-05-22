
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseMatteGenerator(nn.Module):

	def __init__(self):
		super().__init__()

		self.Encoder = Resnet50()

		self.activation = nn.ReLU()
	
		self.ASPP = ASPP()

		self.Decoder = Decoder()


	def forward(self, input_tensor):


		X4, X2, X1, X, input_tensor = self.Encoder(input_tensor)		

		aspp_result = self.ASPP(X4)

		final_logits = self.Decoder(aspp_result, X2, X1, X, input_tensor)
	
		return final_logits


class Resnet50(nn.Module):

	def __init__(self):
		super().__init__()
		#enters at 540 x 960
		self.conv1 = nn.Conv2d(6, 64, kernel_size = 7, stride = 2, padding = 3)
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

	def forward(self, input_tensor):

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

		return(X4, X2, X1, X, input_tensor)


class ASPP(nn.Module):		

	def __init__(self):
		super().__init__()

		self.aspp1 = nn.Conv2d(1024, 256, kernel_size = 1)
		self.bn1 = nn.BatchNorm2d(256)
		self.aspp2 = nn.Conv2d(1024, 256, kernel_size = 3, dilation = 3, padding = 3)
		self.bn2 = nn.BatchNorm2d(256)
		self.aspp3 = nn.Conv2d(1024, 256, kernel_size = 3, dilation = 6, padding = 6)
		self.bn3 = nn.BatchNorm2d(256)
		self.aspp4 = nn.Conv2d(1024, 256, kernel_size = 3, dilation = 9, padding = 9)
		self.bn4 = nn.BatchNorm2d(256)

		self.global_average = nn.AvgPool2d(kernel_size = [5, 5])
		self.global_feature_conv = nn.Conv2d(1024, 256, kernel_size = 1)
		self.bn5 = nn.BatchNorm2d(256)

		self.concat_crunch = nn.Conv2d(1280, 256, kernel_size = 1)

		self.activation = nn.ReLU()

	def forward(self, X):

		aspp1 = self.aspp1(X)
		aspp2 = self.aspp2(X)
		aspp3 = self.aspp3(X)
		aspp4 = self.aspp4(X)

		aspp1 = self.bn1(aspp1)
		aspp2 = self.bn2(aspp2)
		aspp3 = self.bn3(aspp3)
		aspp4 = self.bn4(aspp4)

		aspp1 = self.activation(aspp1)
		aspp2 = self.activation(aspp2)
		aspp3 = self.activation(aspp3)
		aspp4 = self.activation(aspp4)

		global_features = self.global_average(X)
		global_features = F.interpolate(global_features, size = aspp1.shape[-2:], mode = 'bilinear', align_corners = True)
		global_features = self.global_feature_conv(global_features)
		global_features = self.bn5(global_features)

		aspp_concat = torch.cat([aspp1, aspp2, aspp3, aspp4, global_features], dim = 1)
		aspp_result = self.concat_crunch(aspp_concat)
		aspp_result = self.activation(aspp_result)

		return aspp_result


class Decoder(nn.Module):

	def __init__(self):
		super().__init__()

		#We'll upsample first back up to 68 x 120
		self.decode1 = DecoderBlock(256 + 512, 128)
		#then 135 x 240
		self.decode2 = DecoderBlock(128 + 256, 64)
		#then 270 x 480
		self.decode3 = DecoderBlock(64 + 64, 48)

		#and finally to 540 x 960
		self.final_conv = nn.Conv2d(48 + 6, 37, kernel_size = 3, padding = 1)

	def forward(self, aspp_result, X2, X1, X, input_tensor):

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

