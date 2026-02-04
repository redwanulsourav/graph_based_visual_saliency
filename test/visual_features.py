import numpy as np
import torch

def legacy(img):
	r = img[:, :, 2].astype(np.float64)
	g = img[:, :, 1].astype(np.float64)
	b = img[:, :, 0].astype(np.float64)
	intensity = (r + g + b) / 3
	
	maxI = intensity.max()
	maxI = maxI / 10
	mask = intensity > maxI
	
	r[mask] = r[mask] / intensity[mask]
	g[mask] = g[mask] / intensity[mask]
	b[mask] = b[mask] / intensity[mask]

	red = (r - (g + b) / 2)
	green = (g - (r + b) / 2)
	blue = (b - (r + g) / 2)
	yellow = (r + g) / 2 - np.abs(r - g) / 2 - b
	
	intensity[intensity < 0] = 0
	red[red < 0] = 0
	green[green < 0] = 0
	blue[blue < 0] = 0
	yellow[yellow < 0] = 0

	return intensity, red, green, blue, yellow

def new_func(img):
	img = torch.from_numpy(img)	# (H, W, C)
	img = img.unsqueeze(0)		# (1, H, W, C)
	img = img.permute(0, 3, 1, 2) # (B, C, H, W)
# img = img[:, [2, 1, 0], :, :]
	B, C, H, W = img.shape

	inten = torch.sum(img, dim = 1)                 # (B, H, W)
	inten = inten / 3
	max_inten = inten.amax(dim = (1, 2), keepdim = True)    # (B, 1, 1)
	max_inten /= 10                                         # (B, 1, 1)
	mask = (inten > max_inten).bool()                       # (B, H, W)
	
	mask = mask.unsqueeze(0)                                # (B, 1, H, W)
	mask3 = mask.repeat(1, 3, 1, 1)							# (B, 3, H, W)
	inten = inten.unsqueeze(0)                              # (B, 1, H, W)
	inten3 = inten.repeat(1, 3, 1, 1)						# (B, 3, H, W)

	img[mask3] = img[mask3] / inten3[mask3]

	red = img[:, 2, :, :] - (img[:, 1, :, :] + img[:, 0, :, :]) / 2
	green = img[:, 1, :, :] - (img[:, 2, :, :] + img[:, 0, :, :]) / 2
	blue = img[:, 0, :, :] - (img[:, 2, :, :] + img[:, 1, :, :]) / 2
	yellow = (img[:, 2, :, :] + img[:, 1, :, :]) / 2 - torch.abs(img[:, 2, :, :] - img[:, 1, :, :]) / 2 - img[:, 0, :, :]

	inten[inten < 0] = 0
	red[red < 0] = 0
	green[green < 0] = 0
	blue[blue < 0] = 0
	yellow[yellow < 0] = 0

	return inten.squeeze().numpy(), red.squeeze().numpy(), green.squeeze().numpy(), blue.squeeze().numpy(), yellow.squeeze().numpy()



if __name__ == '__main__':
	img = np.random.rand(32, 32, 3)

	i_old, r_old, g_old, b_old, y_old = legacy(img)
	print(i_old.shape)

	i_new, r_new, g_new, b_new, y_new = new_func(img)
	print(i_new.shape)

	print((i_old - i_new).sum())
	print((r_old - r_new).sum())
	print((g_old - g_new).sum())
	print((b_old - b_new).sum())
	print((y_old - y_new).sum())


