import numpy as np
import cv2
from scipy.signal import convolve2d
import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import math

class GBVS():
    def __init__(self, dim = (64, 64)):
        self.dim = dim


    def forward(self, x):
        """
            Input: 
                x (torch.Tensor) -> An torch image in RGB format
            Output:
                Saliency map
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError()

        if len(x.shape) != 4:
            raise ValueError()
        # Extract visual and orientation features.
        v_feats = self.visual_features(x.clone())
        orientation = self.oriented_features(x.clone(), 4)
        result_map = torch.zeros((1, 1, self.dim[1], self.dim[0]), dtype = torch.float64)
        
        for feat_name, scaled_features in v_feats.items():
            activation = self.calc_activation(scaled_features[0])
            tmp = self.normalize(activation)
            result_map += tmp

        # for scale, angleMap in orientation.items():
        for angle_idx, feature in orientation[0].items():
            # feature = cv2.resize(feature, self.dim)
            tmp = self.normalize(self.calc_activation(feature))
            result_map += tmp

        result_map = (result_map - result_map.min()) / (result_map.max() - result_map.min())
        # TODO: Probably need to some resize / rescale here.
        return result_map

    
    def calc_activation(self, x, max_iter = 2000, eps = 1e-12, sigma2 = 6.4, alpha = 0.05, tol = 1e-9):
        if not isinstance(x, torch.Tensor):
            raise ValueError()
            
        B, _, H, W = x.shape
        N = H * W
        device, dtype = x.device, x.dtype

        xf = x.view(B, N).clamp_min(eps)
        
        idx = torch.arange(N, device = device)
        row = idx // W
        col = idx % W

        dr = row[:, None] - row[None, :]
        dc = col[:, None] - col[None, :]

        d2 = (dr * dr + dc * dc).to(dtype)

        spatial = torch.exp(-d2/ (2.0 * sigma2)) 
        
        log_ratio_abs = torch.abs(
                torch.log(xf[:, :, None] / xf[:, None, :])
        )

        adj = log_ratio_abs * spatial

        row_sum = adj.sum(dim = -1 , keepdim = True)
        transition_mat = adj / row_sum
        
        dead = (row_sum.squeeze(-1) <= eps)

        if dead.any():
            transition_mat = transition_mat.clone()
            b_idx, i_idx = dead.nonzero(as_tuple = True)
            transition_mat[b_idx, i_idx, :] = 0
            transition_mat[b_idx, i_idx, i_idx] = 1.0

        
        pi = torch.full((B, N), 1.0 / N, dtype = dtype, device = device)
        uniform = pi

        for _ in range(max_iter):
            pi_next = pi @ transition_mat
            pi_next = (1.0 - alpha) * pi_next + alpha * uniform

            pi_next = pi_next.clamp_min(0)
            pi_next = pi_next / (pi_next.sum(dim = -1, keepdim = True) + eps)

            if (pi_next - pi).abs().sum(dim = -1).max().item() < tol:
                pi = pi_next
                break

            pi = pi_next
        
        pi = pi.view(B, 1, H, W)
        return pi

    
    def normalize(
        self,
        x: torch.Tensor,              # (B, 1, H, W)
        sigma2: float = 6.4,
        eps: float = 1e-5,
        alpha: float = 0.0,           # set e.g. 0.05 for PageRank teleportation
        max_iter: int = 2000,
        tol: float = 1e-8,
    ):
        """
        Batched equivalent of your NumPy normalize():
          A_ij = x[j] * exp(-dist(i,j)^2/(2*sigma2))
          P = row_normalize(A)
          pi = stationary distribution via power iteration
        Returns: (B, 1, H, W)
        """
        B, _, H, W = x.shape
        N = H * W
        device, dtype = x.device, x.dtype

        # Match: x[x==0] = 1e-5 (avoid in-place modifying caller)
        x_safe = x.clone()
        x_safe[x_safe == 0] = eps

        # Flatten destination weights: (B, N)
        xf = x_safe.view(B, N)

        # Coordinates for each node index: (N,)
        idx = torch.arange(N, device=device)
        row = idx // W
        col = idx % W

        # Pairwise squared distances: (N, N)
        dr = row[:, None] - row[None, :]
        dc = col[:, None] - col[None, :]
        d2 = (dr * dr + dc * dc).to(dtype)

        # Spatial kernel (shared across batch): (N, N)
        spatial = torch.exp(-d2 / (2.0 * sigma2))

        # Adjacency: A[b, i, j] = xf[b, j] * spatial[i, j]
        adj = spatial[None, :, :] * xf[:, None, :]         # (B, N, N)

        # Row-normalize -> transition matrix P
        row_sum = adj.sum(dim=-1, keepdim=True)            # (B, N, 1)
        P = adj / (row_sum + 1e-12)                        # (B, N, N)

        # Fix dead rows (rare here unless spatial underflows everywhere)
        dead = (row_sum.squeeze(-1) <= 1e-12)              # (B, N)
        if dead.any():
            P = P.clone()
            b_idx, i_idx = dead.nonzero(as_tuple=True)
            P[b_idx, i_idx, :] = 0
            P[b_idx, i_idx, i_idx] = 1.0

        # Power iteration for stationary distribution (row-vector convention)
        pi = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
        uniform = pi.clone()                                # (B, N)

        for _ in range(max_iter):
            pi_next = pi @ P                                # (B, N)

            # PageRank trick (teleportation): guaranteed unique stationary dist if alpha>0
            if alpha > 0:
                pi_next = (1.0 - alpha) * pi_next + alpha * uniform

            # Renormalize for numeric stability
            pi_next = pi_next.clamp_min(0)
            pi_next = pi_next / (pi_next.sum(dim=-1, keepdim=True) + 1e-12)

            # Convergence check
            if (pi_next - pi).abs().max().item() < tol:
                pi = pi_next
                break
            pi = pi_next

        return pi.view(B, 1, H, W)  
    
    def torch_filter2D(self, x, k):
        B, C, H, W = x.shape
        # x_pad = F.pad(x, pad = (1, 1, 1, 1), mode = 'replicate')
        x_pad = self._opencv_border_reflect_pad(x, k.shape[-1]//2, k.shape[-1]//2) 
        k = k.repeat(C, 1, 1, 1)

        return F.conv2d(x_pad, k, groups = C)
    
    def _opencv_border_reflect_pad(self, x, pad_h, pad_w):
        # OpenCV BORDER_REFLECT:
        # fedcba | abcdefgh | hgfedcb 
        B, C, H, W = x.shape

        def _reflect_indices(n, pad):
            idx = torch.arange(-pad, n + pad, device = x.device)
            idx = idx.clone()

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
        iy = _reflect_indices(H, pad_h)
        ix = _reflect_indices(W, pad_w)
        return x[:, :, iy[:, None], ix[None, :]]

    def fsd_laplacian(self, img, n: int):
        if not isinstance(img, torch.Tensor):
            raise TypeError()

        if len(img.shape) != 4:
            raise ValueError(f'{img.shape}')
        
        lpf = torch.zeros((5,)).double()
        lpf[2] = 3.0/8.0      # Middle
        lpf[1] = 0.25
        lpf[3] = 0.25
        lpf[0] = 1.0/16.0
        lpf[4] = 1.0/16.0
        lpf = torch.outer(lpf, lpf)

        lpf = lpf.unsqueeze(0).unsqueeze(0)
        fsd_low_passed_pyr = {}
        fsd_laplacian_pyr = {}
        fsd_low_passed_pyr[0] = img   # At scale 0, we have the original image.

        for i in range(1, n+1):
            # g0 = cv2.filter2D(fsdLowPassedPyr[i-1], -1, lpf, borderType = cv2.BORDER_REFLECT)
            g0 = self.torch_filter2D(fsd_low_passed_pyr[i - 1], lpf)
            fsd_laplacian_pyr[i-1] = fsd_low_passed_pyr[i-1] - g0  # The difference is the laplacian at previous scale.
            new_dim = (fsd_laplacian_pyr[i-1].shape[-2] // 2, fsd_laplacian_pyr[i-1].shape[-1] // 2) # The gaussian at current scale will be downsampled to this dim.
            fsd_low_passed_pyr[i] = F.interpolate(g0, size = (new_dim[1], new_dim[0]), mode = 'bilinear', align_corners = False) # cv2.resize(g0, newDim)   # Downsample and store.

        
        return (fsd_low_passed_pyr, fsd_laplacian_pyr)
    
    def _symmetric_pad_2d(self, x, pad_h, pad_w):
        B, C, H, W = x.shape

        def idx_symmetric(n, pad):
            idx = torch.arange(-pad, n + pad, device = x.device)
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
        iy = idx_symmetric(H, pad_h)
        ix = idx_symmetric(W, pad_w)
        return x[:, :, iy][:, :, :, ix]
              
    def oriented_features(self, img, anglesN):
        # TESTED 
        if not isinstance(img, torch.Tensor):
            raise TypeError()

        if len(img.shape) != 4:
            raise ValueError()
        
        B, C, H, W = img.shape
        lpf = torch.zeros((5,)).double()
        lpf[2] = 3.0/8.0      # Middle
        lpf[1] = 0.25
        lpf[3] = 0.25
        lpf[0] = 1.0/16.0
        lpf[4] = 1.0/16.0
        lpf = torch.outer(lpf, lpf)

        lpf = lpf.unsqueeze(0).unsqueeze(0)
        inten = torch.sum(img, dim = 1)                 # (B, H, W)
        inten = inten / 3

        inten = inten.unsqueeze(1)                              # (B, 1, H, W)
        inten[inten < 0] = 0
        oriented_feat = {}
        fsd_low_passed, laplacian = self.fsd_laplacian(inten, 4)
        
        for p, img in laplacian.items():
            oriented_feat[p] = {}
            for alpha in range(1, anglesN + 1):
                # imgI = img.astype(np.complex128)
                img_c = torch.complex(img, torch.zeros_like(img))
                
                H, W = img.shape[2], img.shape[3]
                # xx = np.arange(W) - W // 2
                xx = torch.arange(W, dtype = torch.int64, device = img.device) - W // 2
                # yy = np.arange(H) - H // 2
                yy = torch.arange(H, dtype = torch.int64, device = img.device) - H // 2

                # X, Y = np.meshgrid(xx, yy)
                X, Y = torch.meshgrid(xx, yy, indexing = 'xy')
                
                # theta = np.pi / 4 * (alpha - 1)
                theta = math.pi / 4 * (alpha - 1)

                # k = (np.pi / 2) * np.array([np.cos(theta), np.sin(theta)])
                k = (math.pi / 2) * torch.tensor(
                        [math.cos(theta), math.sin(theta)],
                        device = img.device,
                        dtype = torch.float64)
                

                multiplier = k[0] * X + k[1] * Y
                img_c = img * torch.exp(1j * multiplier)

                # convolved = convolve2d(imgI, lpf, mode='same', boundary= 'symm')
                # real_pad = F.pad(img_c.real, (2, 2, 2, 2), mode = 'reflect')
                real_pad = self._symmetric_pad_2d(img_c.real, 2, 2)
                imag_pad = self._symmetric_pad_2d(img_c.imag, 2, 2)
                # imag_pad = F.pad(img_c.imag, (2, 2, 2, 2), mode = 'reflect')
                img_i_pad = torch.complex(F.conv2d(real_pad, lpf), F.conv2d(imag_pad, lpf))

                img_m = torch.abs(img_i_pad).real # np.abs(convolved)
                oriented_feat[p][alpha] = img_m


        return oriented_feat

    def torch_pyrdown(self, img):
        # TESTED
        if not isinstance(img, torch.Tensor):
            raise TypeError()

        if len(img.shape) != 4:
            raise ValueError(f'{img.shape}')

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
        img_pad = self._opencv_border_reflect_pad(img, 2, 2)

        blurred = F.conv2d(img_pad, kernel, stride = 2, groups = img.shape[1])
        return blurred
        

    def visual_features(self, img):
        # TESTED

        if not isinstance(img, torch.Tensor):
            raise TypeError()

        if len(img.shape) != 4:
            raise ValueError()
        
        B, C, H, W = img.shape
        
        inten = torch.sum(img, dim = 1)                 # (B, H, W)
        inten = inten / 3

        max_inten = inten.amax(dim = (1, 2), keepdim = True)    # (B, 1, 1)
        max_inten /= 10                                         # (B, 1, 1)
        mask = (inten > max_inten).bool()                       # (B, H, W)
        
        mask = mask.unsqueeze(0)                                # (B, 1, H, W)
        inten = inten.unsqueeze(1)                              # (B, 1, H, W)

        mask3 = mask.repeat(1, 3, 1, 1)
        inten3 = inten.repeat(1, 3, 1, 1)
        
        img[mask3] = img[mask3] / inten3[mask3]
        
        red = img[:, 0, :, :] - (img[:, 1, :, :] + img[:, 2, :, :]) / 2
        green = img[:, 1, :, :] - (img[:, 0, :, :] + img[:, 2, :, :]) / 2
        blue = img[:, 2, :, :] - (img[:, 0, :, :] + img[:, 1, :, :]) / 2
        yellow = (img[:, 0, :, :] + img[:, 1, :, :]) / 2 - torch.abs(img[:, 0, :, :] - img[:, 1, :, :]) / 2 - img[:, 2, :, :]
        inten[inten < 0] = 0
        red[red < 0] = 0
        green[green < 0] = 0
        blue[blue < 0] = 0
        yellow[yellow < 0] = 0
        
        feat_names = ['intensity', 'red', 'green', 'blue', 'yellow']
        feat = {}

        for name in feat_names:
            feat[name] = {}
        
        feat['intensity'][0] = inten
        feat['red'][0] = red.unsqueeze(1)
        feat['green'][0] = green.unsqueeze(1)
        feat['blue'][0] = blue.unsqueeze(1)
        feat['yellow'][0] = yellow.unsqueeze(1)
        
        for name in feat_names:
            for i in range(0, 4):
                if i > 0: 
                    feat[name][i] = self.torch_pyrdown(feat[name][i - 1])

        return feat


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input')

    ap = ap.parse_args()
    img = cv2.imread(ap.input)
    model = GBVS()
    saliencyMap = model.forward(img)
    cv2.imwrite('out.jpg', saliencyMap)
