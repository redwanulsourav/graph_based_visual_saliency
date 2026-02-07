import argparse
import os
import cv2
import torch

from gbvs import GBVS

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input_dir', required = True)
    ap.add_argument('-o', '--output_dir', required = True)
    ap = ap.parse_args()

    files = os.listdir(ap.input_dir)

    for file in files:
        filename = file.split('.')[0]
        full_path = f'{ap.input_dir}/{file}'
        print(full_path)
        img = cv2.imread(full_path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        torch_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float64) / 255
        model = GBVS()
        model.forward(torch_img)

        print(f'{file} processed')

