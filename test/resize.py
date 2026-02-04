import numpy as np
import torch
import torch.nn.functional as F
import cv2

def legacy(x, dim):
	return cv2.resize(x, dim)

def torch_resize(x, dim):
	return F.interpolate(x, size = (dim[1], dim[0]), mode = 'bilinear', align_corners = False)

if __name__ == '__main__':
	img0 = np.random.rand(64, 64, 3).astype(np.float64)
	dim = (32, 32)
	out0 = legacy(img0, dim)

	img1 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0)

	out1 = torch_resize(img1, dim).squeeze().permute(1, 2, 0).numpy()

	print((out0 - out1).sum())
