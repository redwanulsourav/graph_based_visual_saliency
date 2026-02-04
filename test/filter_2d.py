import cv2
import numpy as np
import torch
import torch.nn.functional as F

def opencv_border_reflect_pad(x, pad_h, pad_w):
    # OpenCV BORDER_REFLECT:
    # fedcba|abcdefgh|hgfedcb  (includes edge pixel)
    B, C, H, W = x.shape
    device = x.device

    def reflect_indices(n, pad):
        # indices for length n padded by pad on both sides
        # left: pad..1 then 0..n-1 then n-1..n-pad
        idx = torch.arange(-pad, n + pad, device=device)
        idx = idx.clone()
        # map negatives and overflow with reflection INCLUDING edge
        # while loops handle pad > n too
        while True:
            neg = idx < 0
            if not neg.any():
                break
            idx[neg] = -idx[neg] - 1

        while True:
            over = idx >= n
            if not over.any():
                break
            idx[over] = 2*n - idx[over] - 1
        return idx

    iy = reflect_indices(H, pad_h)
    ix = reflect_indices(W, pad_w)

    # advanced indexing: (B,C,H,W) -> (B,C,H+2p,W+2p)
    return x[:, :, iy][:, :, :, ix]

def legacy(x):
	k = np.zeros((5, 5), dtype = np.float64)
	k[0, 0] = k[0, 2] = k[2, 0] = k[2, 2] = 1
	k[0, 1] = k[2, 1] = k[1, 0] = k[1, 2] = 3
	k[2, 2] = 5
	k[0, 3] = k[0, 4] = 2
	k[1, 3] = k[1, 4] = 3
	k[2, 3] = k[2, 4] = 1
	k[3, 3] = k[3, 4] = 4
	k[4, 0] = 1
	k[4, 1] = 2
	k[4, 2] = 3
	k[4, 3] = 4
	k[4, 4] = 5

	return cv2.filter2D(x, -1, k, borderType = cv2.BORDER_REFLECT)

def torch_filter2D(x, k):
	B, C, H, W = x.shape
	

	kH = k.shape[2] // 2
	# x_pad = F.pad(x, pad = (kH, kH, kH, kH), mode = 'reflect')
	x_pad = opencv_border_reflect_pad(x, kH, kH)
	print(x_pad.shape)	
	k = k.repeat(C, 1, 1, 1)

	return F.conv2d(x_pad, k, groups = C)

if __name__ == '__main__':
	
	img = np.random.rand(64, 64, 3).astype(np.float64)

	out0 = legacy(img)

	img1 = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

	out1 = torch_filter2D(img1, None).squeeze().permute(1, 2, 0).numpy()

	print(out0.shape)
	print(out1.shape)

	print((out0 - out1).sum())



	
