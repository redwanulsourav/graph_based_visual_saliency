import numpy as np
import cv2
from scipy.signal import convolve2d
import argparse

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
        visualFeatures = self.visual_features(x)
        orientation = self.extractOrientationFeatures(x, 4)
        resultMap = np.zeros(self.dim, np.float32)
        
        for featureName, scaleFeatures in visualFeatures.items():
            tempMap = self.normalize(self.calcActivation(scaleFeatures[0]))
            resultMap += tempMap

        # for scale, angleMap in orientation.items():
        for angleIdx, feature in orientation[0].items():
            # feature = cv2.resize(feature, self.dim)
            tempMap = self.normalize(self.calcActivation(feature))
            resultMap += tempMap

        resultMap = (resultMap - resultMap.min()) / (resultMap.max() - resultMap.min())
        resultMap = resultMap * 255
        resultMap = resultMap.astype(np.uint8)
        resultMap = cv2.resize(resultMap, (xOrig.shape[1], xOrig.shape[0]))
        return resultMap

    
    def calcActivation(self, x):
        x[x == 0] = 1e-5
        nNodes = x.shape[0] * x.shape[1]
        adjMatrix = np.zeros((nNodes, nNodes), dtype = np.float32)
        for index, value in np.ndenumerate(adjMatrix):
            i, j = index
            rowI = i // self.dim[1] # Y
            colI = i % self.dim[1] # X

            rowJ = j // self.dim[1] # Y
            colJ = j % self.dim[1] # X
            # print(i, j, rowI, colI, rowJ, colJ)
            adjMatrix[i, j] = adjMatrix[j, i] = np.abs(np.log(x[rowI, colI] / x[rowJ, colJ])) * \
                                np.exp(-(np.square(rowI - rowJ) + np.square(colI - colJ))/(2 * 6.4))        

        rowSums = adjMatrix.sum(axis = 1, keepdims = True)
        transitionMatrix = adjMatrix / rowSums
        pwrTransitionMatrix = transitionMatrix.copy()

        pi = np.full(nNodes, 1.0 / nNodes)
        tolerance = 1e-8
        maxiter = 1000

        for i in range(maxiter):
            pwrTransitionMatrix = pwrTransitionMatrix @ pwrTransitionMatrix
            nextPi = pi @ pwrTransitionMatrix
            nextPi1 = nextPi @ transitionMatrix
            print(np.abs(nextPi - pi).max())
            if np.allclose(nextPi, pi, atol = tolerance):
                break
            # pi = nextPi
        print('activation done')
        pi = pi @ pwrTransitionMatrix
        return pi.reshape(self.dim[0], self.dim[1])
    def normalize(self, x):
        x[x == 0] = 1e-5
        nNodes = x.shape[0] * x.shape[1]
        adjMatrix = np.zeros((nNodes, nNodes), dtype = np.float32)
        for index, value in np.ndenumerate(adjMatrix):
            # 2D coordinates of node i.
            i, j = index
            rowI = i // self.dim[1] # Y
            colI = i % self.dim[1] # X

            rowJ = j // self.dim[1] # Y
            colJ = j % self.dim[1] # X

            adjMatrix[i, j] = x[rowJ, colJ] * \
                    np.exp(-(np.square(rowI - rowJ) + np.square(colI - colJ))/(2 * 6.4))

        rowSums = adjMatrix.sum(axis = 1, keepdims = True)
        transitionMatrix = adjMatrix / rowSums

        pi = np.full(nNodes, 1.0 / nNodes)
        tolerance = 1e-8
        maxiter = 100000
        pwrTransitionMatrix = transitionMatrix.copy()
        
        
        for i in range(maxiter):
            pwrTransitionMatrix = pwrTransitionMatrix @ pwrTransitionMatrix
            nextPi0 = pi @ pwrTransitionMatrix
            nextPi1 = nextPi0 @ transitionMatrix
            print(np.abs(nextPi - pi).max())
            if np.allclose(nextPi, pi, atol = tolerance):
                break
            # pi = nextPi
        print('normalization done')
        pi = pi @ pwrTransitionMatrix
        return pi.reshape(self.dim[0], self.dim[1])
	
	def torch_filter2D(x, k):
		B, C, H, W = x.shape
		# x_pad = F.pad(x, pad = (1, 1, 1, 1), mode = 'replicate')
		x_pad = self._reflect_indices(x, k.shape[-1]//2, k.shape[1]//2)	
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
		iy = reflect_indices(H, pad_h)
		ix = reflect_indices(W, pad_w)

		return x[:, :, iy][:, :, :, ix]


    def fsd_laplacian(self, img, n: int):
		if not isinstance(img, torch.Tensor):
			raise TypeError()

		if len(img.shape) != 4:
			raise ValueError()
		
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
           	g0 = torch_filter2D(fsd_low_passed_pyr[i - 1], lpf)

			fsd_laplacian_pyr[i-1] = fsd_low_passed_pyr[i-1] - g0  # The difference is the laplacian at previous scale.
            new_dim = (fsd_laplacian_pyr[i-1].shape[1] // 2, fsd_laplacian_pyr[i-1].shape[0] // 2) # The gaussian at current scale will be downsampled to this dim.
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
       	
		"""
        r = img[:, :, 2].astype(np.float64)
        g = img[:, :, 1].astype(np.float64)
        b = img[:, :, 0].astype(np.float64)
        intensity = (r + g + b) / 3
		"""

		inten = torch.sum(img, dim = 1) / 3

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
				real_pad = self._symmetric_pad_2d(torch_img_c.real, 2, 2)
				imag_pad = self._symmetric_pad_2d(torch_img_c.imag, 2, 2)
				# imag_pad = F.pad(img_c.imag, (2, 2, 2, 2), mode = 'reflect')
				img_i_pad = torch.complex(real_pad, imag_pad)

                img_m = F.conv2d(img_i_pad, lpf.to(torch_img_c.dtype)) # np.abs(convolved)
                oriented_feat[p][alpha] = img_m

        return oriented_feat

	def torch_pyrdown(self, img):
		# TESTED
		if not isinstance(img, torch.Tensor):
			raise TypeError()

		if len(img.shape) != 4:
			raise ValueError()

		kernel = torch.tensor(
				[[1,  4,  6,  4, 1],
				 [4, 16, 24, 16, 4],
				 [6, 24, 36, 24, 6],
				 [4, 16, 24, 16, 4],
				 [1,  4,  6,  4, 1]],
				dtype = x.dtype,
				device = x.device)

		kernel = kernel / kernel.sum()
		kernel = kernel.view(1, 1, 5, 5).repeat(x.shape[1], 1, 1, 1)
		
		img_pad = F.pad(img, (2, 2, 2, 2), mode = 'reflect')

		blurred = F.conv2d(x, kernel, stride = 2, groups = img.shape[1])
		return blurred
		

    def visual_features(self, img):
		# TESTED

		if not isinstance(img, torch.Tensor):
			raise TypeError()

		if len(img.shape) != 4:
			raise ValueError()
		
		B, C, H, W = img.shape
		
		inten = torch.sum(img, dim = 1) 				# (B, H, W)
		inten = inten / 3

		max_inten =	inten.amax(dim = (1, 2), keepdim = True) 	# (B, 1, 1)
		max_inten /= 10											# (B, 1, 1)
		mask = (inten > max_inten).bool()						# (B, H, W)
		
		mask = mask.unsqueeze(0)								# (B, 1, H, W)
		inten = inten.unsqueeze(0)								# (B, 1, H, W)

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

        features['intensity'][0] = inten
        features['red'][0] = red
        features['green'][0] = green
        features['blue'][0] = blue
        features['yellow'][0] = yellow

        for name in feat_names:
            for i in range(1, 4):
                features[name][i] = self.torch_pyrdown(features[name][i - 1])

        return features


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input')

    ap = ap.parse_args()
    img = cv2.imread(ap.input)
    model = GBVS()
    saliencyMap = model.forward(img)
    cv2.imwrite('out.jpg', saliencyMap)
