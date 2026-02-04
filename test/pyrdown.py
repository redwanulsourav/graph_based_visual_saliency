import numpy as np
import torch
import torch.nn.functional as F
import cv2

def legacy(x):
	out = cv2.pyrDown(x)
	return out

def torch_pyrdown(img):
	kernel = torch.tensor(
		 [[1,  4,  6,  4, 1],
		  [4, 16, 24, 16, 4],
		  [6, 24, 36, 24, 6],
		  [4, 16, 24, 16, 4],
		  [1,  4,  6,  4, 1]],
		 dtype = img.dtype,
		 device = img.device)

	kernel = kernel / kernel.sum()
	kernel = kernel.view(1, 1, 5, 5).repeat(img.shape[1], 1, 1, 1)
	
	img_pad = F.pad(img, (2, 2, 2, 2), mode = 'reflect')

	blurred = F.conv2d(img_pad, kernel, stride = 2, groups = img.shape[1])
	return blurred

if __name__ == '__main__':
	img = np.random.rand(32, 32, 3)

	out1 = legacy(img)

	img0 = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
	out2 = torch_pyrdown(img0).squeeze(0).permute(1, 2, 0).numpy()

	print(out1.shape)
	print(out2.shape)
	
	print((out1-out2).sum())

