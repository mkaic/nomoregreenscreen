
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
import ffmpeg
import argparse

import os

#Define hyperparameters. Context depths and strides refer to the number and spacing of temporal context to use on
#the two input videos. Not really adjustable (without retraining model), but good to have notated here. Might
#try to make them adjustable somehow TODO?

target_context_depth = 2
target_context_stride = 5
target_loop_buffer = target_context_depth * target_context_stride

background_context_depth = 3
background_context_stride = 5
background_loop_buffer = background_context_depth * background_context_stride


#Allows PyTorch to take full advantage of the fancy stuff in my GPU for convolutions.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"

#Allows me to run this from the command line with custom inputs/outputs specified there.
parser = argparse.ArgumentParser()
parser.add_argument('--target', '-t', help = 'Path to the video you want to remove the background FROM.')
parser.add_argument('--background', '-b', help = 'Path to the video you want to give the model as reference for what JUST THE BACKGROUND looks like.')
parser.add_argument('--output', '-o', help = 'Path to output the final video file/still image frame file to.')

args = parser.parse_args()

#Preprocess the background to a series of still frames. It ain't pretty but it's necessary.

ffmpeg.input(args.background).output('cached_frames/background/%d.jpg', vsync = 0, qscale = 2).overwrite_output().run(quiet = True)

#Given the path to a video, a central frame index, the number of frames before and after it to grab, 
#the spacing between those frames, and a unique cache identifier for later, return a tensor of the right images.
def get_frame_packet(path, index, context_depth, context_stride, cache_id):

	#Import the video and use ffprobe to find out the frame count
	video = ffmpeg.input(path)

	start_index = index - (context_stride * context_depth)
	selection_string = ''

	#construct the selection query for ffmpeg to use to find the right frames.
	for context_index in range(2 * context_depth + 1):
		selection_string = selection_string + 'eq(n,' + str(start_index + (context_index * context_stride)) + ')+'

	selection_string = selection_string[:-1]	
	print(selection_string)

	#output the correct frames
	video.filter_('select', selection_string).output('cached_frames/' + cache_id + '/%d.jpg', vsync = 0, qscale = 2).overwrite_output().run(quiet = True)

	frame_packet =  []

	#loop through the frame packet we just made
	for image in os.listdir('cached_frames/' + cache_id):
		filename = os.fsdecode(image)
		#find the right ones
			
		frame_packet.append(transforms.ToTensor()(Image.open('cached_frames/' + cache_id + '/' + filename)))

	#Since whatever packet we're getting is gonna be fed into the neural net, we'll just concatenate along the channels axis, so it's like one THICC image.
	frame_packet_tensor = torch.stack(frame_packet, 0)
	frame_packet_tensor = frame_packet_tensor.view(-1, frame_packet_tensor.shape[-2], frame_packet_tensor.shape[-1])
	return frame_packet_tensor


def get_frame_embedding(target_frame):

	return

def generate_input_tensor(index):

	return

#Note to self -- it might be better to just pre-convert both clips to image sequences. It ain't ideal but it might be necessary.

#Inference time!
#First we'll grab the frame counts of both input vids, they'll be useful later.
target_probe = ffmpeg.probe(args.target)
target_frame_count = int(target_probe['streams'][0]['nb_frames'])

background_probe = ffmpeg.probe(args.background)
background_frame_count = int(background_probe['streams'][0]['nb_frames'])

for target_frame in range(target_loop_buffer - 1, target_frame_count - target_loop_buffer):

	input_tensor = generate_input_tensor(target_frame)



#Main inference loop. Loop over frames of target video (minus a margin on either side for getting frame packets),
#and use The Fancy Model on each of them, then save the result to an outputs directory 









"""
#Reconstruct a video from the alpha png frames produced.

color_space = target_probe['streams'][0]['color_space']
color_transfer = target_probe['streams'][0]['color_transfer']
color_primaries = target_probe['streams'][0]['color_primaries']
target_framerate = target_probe['streams'][0]['avg_frame_rate']

try:
	output = ffmpeg.input('./test_footage/bg_frames/%04d.jpg', framerate = target_framerate)
	output.output('colorSpaceCorrect.mov', vcodec = 'qtrle', color_primaries = color_primaries, color_trc = color_transfer, colorspace = color_space, qscale = 2).overwrite_output().run(quiet = True, capture_stderr = True)

except ffmpeg.Error as e:

	print('stderr:', e.stderr.decode('utf8'))
	raise e

"""