"""
draw each cluster for verification
./output
| - group_0000.png
| - group_0001.png
| ...
"""

import os
from argparse import ArgumentParser
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils

from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

img_root = os.path.join(args.dataset, 'img')
labels = torch.load(os.path.join(args.dataset, 'labels.pt'))

tt = transforms.ToTensor()

output_path = os.path.join(args.output, 'group_{:04d}.png')
for gid, g in enumerate(tqdm(labels['groups'], ascii=True, ncols=100)):
	imgs = []
	for idx in g:
		img = Image.open(os.path.join(img_root, '{:06d}.png'.format(idx)))
		img = tt(img)
		imgs.append(img)
	imgs = torch.stack(imgs)
	vutils.save_image(imgs, output_path.format(gid))
