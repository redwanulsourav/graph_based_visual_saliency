import numpy as np
import torch
import cv2
import torch.nn.functional as F
import math
from scipy.signal import convolve2d
from filter_2d import torch_filter2D

def fsdLaplacian(img: np.ndarray, n: int):
	torch_img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

	lpf = np.zeros((5,1), dtype=np.float64)
	lpf[2, 0] = 3.0/8.0      # Middle
	lpf[1, 0] = 0.25
	lpf[3, 0] = 0.25
	lpf[0, 0] = 1.0/16
	lpf[4, 0] = 1.0/16
	lpf = np.matmul(lpf, lpf.transpose())
	

	torch_lpf = torch.zeros((5,)).double()
	torch_lpf[2] = 3.0/8.0
	torch_lpf[1] = 0.25
	torch_lpf[3] = 0.25
	torch_lpf[0] = 1.0 / 16.0
	torch_lpf[4] = 1.0 / 16.0
	torch_lpf = torch.outer(torch_lpf, torch_lpf).unsqueeze(0).unsqueeze(0)

	fsdLowPassedPyr = {}
	fsdLaplacianPyr = {}
	fsdLowPassedPyr[0] = img   # At scale 0, we have the original image.

	torch_fsd_low_passed_pyr = {}
	torch_fsd_laplacian_pyr = {}
	torch_fsd_low_passed_pyr[0] = torch_img

	for i in range(1, n+1):
		g0 = cv2.filter2D(fsdLowPassedPyr[i-1], -1, lpf, borderType = cv2.BORDER_REFLECT)
		fsdLaplacianPyr[i-1] = fsdLowPassedPyr[i-1] - g0  # The difference is the laplacian at previous scale.
		newDim = (fsdLaplacianPyr[i-1].shape[1] // 2, fsdLaplacianPyr[i-1].shape[0] // 2) # The gaussian at current scale will be downsampled to this dim.
		fsdLowPassedPyr[i] = cv2.resize(g0, newDim)   # Downsample and store.

		torch_g0 = torch_filter2D(torch_fsd_low_passed_pyr[i-1], torch_lpf)
		torch_fsd_laplacian_pyr[i - 1] = torch_fsd_low_passed_pyr[i - 1] - torch_g0
		torch_fsd_low_passed_pyr[i] = F.interpolate(torch_g0, size =  (newDim[1], newDim[0]), mode = 'bilinear')

		print(f'[DEBUG] Point 1')
		print(f'[DEBUG] g0.shape: {g0.shape} torch_g0.shape: {torch_g0.shape}')
		print(f'[DEBUG] torch_g0.squeeze().shape: {torch_g0.squeeze().shape}')
		print(f'[DEBUG] {(torch_g0.squeeze().numpy() - g0).sum()}')
		print(f'[DEBUG] {(torch_fsd_laplacian_pyr[i - 1].squeeze().unsqueeze(2).numpy() - fsdLaplacianPyr[i-1]).sum()}')
		print(f'[DEBUG] {(torch_fsd_low_passed_pyr[i].squeeze().unsqueeze(2).numpy() - fsdLowPassedPyr[i]).sum()}')
		print('') 
	return (fsdLowPassedPyr, fsdLaplacianPyr), (torch_fsd_low_passed_pyr, torch_fsd_laplacian_pyr)

def symmetric_pad_2d(x, pad_h, pad_w):
    # symmetric padding includes edge values
    # works for real tensors; apply to real/imag separately for complex
    N, C, H, W = x.shape
    device = x.device

    def idx_symmetric(n, pad):
        idx = torch.arange(-pad, n + pad, device=device)
        # map by symmetric reflection INCLUDING edge
        while True:
            neg = idx < 0
            if not neg.any(): break
            idx[neg] = -idx[neg] - 1
        while True:
            over = idx >= n
            if not over.any(): break
            idx[over] = 2*n - idx[over] - 1
        return idx

    iy = idx_symmetric(H, pad_h)
    ix = idx_symmetric(W, pad_w)
    return x[:, :, iy][:, :, :, ix]

def extractOrientationFeatures(img: np.ndarray, anglesN): 
	torch_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
	# torch_img = torch_img[:, [2, 1, 0], :]

	lpf = np.zeros((5,1), dtype=np.float64)
	lpf[2, 0] = 3.0/8.0      # Middle
	lpf[1, 0] = 0.25
	lpf[3, 0] = 0.25
	lpf[0, 0] = 1.0/16.0
	lpf[4, 0] = 1.0/16.0
	lpf = np.matmul(lpf, lpf.transpose())
	
	torch_lpf = torch.zeros((5,)).double()
	torch_lpf[2] = 3.0/8.0
	torch_lpf[1] = 0.25
	torch_lpf[3] = 0.25
	torch_lpf[0] = 1.0 / 16.0
	torch_lpf[4] = 1.0 / 16.0
	torch_lpf = torch.outer(torch_lpf, torch_lpf).unsqueeze(0).unsqueeze(0)


	r = img[:, :, 2].astype(np.float64)
	g = img[:, :, 1].astype(np.float64)
	b = img[:, :, 0].astype(np.float64)
	intensity = (r + g + b) / 3

	torch_inten = torch.sum(torch_img, dim = 1) / 3
	
	print(f'[DEBUG] Point 2')
	print(f'{(torch_inten.squeeze().unsqueeze(0).numpy() - intensity).sum()}')
	print('')

	orientedFeatures = {}

	(fsdLowPassed, laplacian), (torch_fsd_low_passed, torch_laplacian) = fsdLaplacian(intensity, 4)
	torch_oriented_features = {}	
	for p, img in laplacian.items():
		orientedFeatures[p] = {}
		torch_oriented_features[p] = {}
		# torch_img = torch.from_numpy(img).unsqueeze(2)
		torch_img = torch_laplacian[p]
		for alpha in range(1, anglesN + 1):
			imgI = img.astype(np.complex128)
			H, W = img.shape[0], img.shape[1]
			xx = np.arange(W) - W // 2
			yy = np.arange(H) - H // 2
			X, Y = np.meshgrid(xx, yy)

			torch_H, torch_W = torch_img.shape[2], torch_img.shape[3]
			torch_xx = torch.arange(torch_W, dtype = torch.int64, device = torch_img.device)  - torch_W // 2
			torch_yy = torch.arange(torch_H, dtype = torch.int64, device = torch_img.device) - torch_H // 2
			torch_X, torch_Y = torch.meshgrid(torch_xx, torch_yy, indexing = 'xy')
			
			print(f'[DEBUG] Point 3')
			print(f'xx.shape: {xx.shape}, yy.shape: {yy.shape}')
			print(f'torch_xx.shape: {torch_xx.shape}, torch_yy.shape: {torch_yy.shape}')
			print(f'X.shape: {X.shape}, Y.shape: {Y.shape}')
			print(f'torch_X.shape: {torch_X.shape}, torch_Y.shape: {torch_Y.shape}')
			print("max |X - torch_X|:", np.max(np.abs(X - torch_X.cpu().numpy())))
			print("max |Y - torch_Y|:", np.max(np.abs(Y - torch_Y.cpu().numpy())))

			print('')

			theta = np.pi / 4 * (alpha - 1)
			torch_theta = math.pi / 4 * (alpha - 1)

			k = (np.pi / 2) * np.array([np.cos(theta), np.sin(theta)])
			torch_k = (math.pi / 2) * torch.tensor(
					[math.cos(torch_theta), math.sin(torch_theta)],
					device = torch_img.device,
					dtype = torch.float64)

			multiplier = k[0] * X + k[1] * Y
			torch_multiplier = torch_k[0] * torch_X + torch_k[1] * torch_Y
			print("max |multiplier - torch_multiplier|:",
      		np.max(np.abs(multiplier - torch_multiplier.cpu().numpy())))
			imgI = img * np.exp(1j * multiplier)
			torch_img_c = torch_img * torch.exp(1j * torch_multiplier)
			print(f'[DEBUG] Point 5')
			print(f'[DEBUG] imgI.shape: {imgI.shape}')
			print(f'[DEBUG] torch_img_c.shape: {torch_img_c.shape}')

			convolved = convolve2d(imgI, lpf, mode='same', boundary= 'symm')
			
			real_pad = symmetric_pad_2d(torch_img_c.real, 2, 2)
			imag_pad = symmetric_pad_2d(torch_img_c.imag, 2, 2)
			img_i_pad = torch.complex(real_pad, imag_pad)

			imgM = np.abs(convolved)
			img_m = F.conv2d(img_i_pad, torch_lpf.to(torch_img_c.dtype))
			img_m = torch.abs(img_m)
		
			diff = img_m - imgM

			num = np.linalg.norm(diff.ravel())
			den = np.linalg.norm(imgM.ravel()) + 1e-12
			
			print("relative L2:", num / den)

			print(f'')
			print(f'[DEBUG] Point 4')
			print(f'[DEBUG] imgM.shape: {imgM.shape}, img_m.shape: {img_m.shape}')
# print(f'[DEBUG] {np.max(np.abs((img_m.squeeze().numpy() - imgM))}')
			print(f'')

			orientedFeatures[p][alpha] = imgM

	return orientedFeatures

if __name__ == '__main__':
	img = np.random.rand(64, 64, 3).astype(np.float64)
	extractOrientationFeatures(img, 4)
