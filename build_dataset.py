import os
import shutil
from argparse import ArgumentParser

import torch
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage import feature, color
from PIL import Image, ImageOps
from tqdm import tqdm

"""
assuming all icon are square and stored as the following structure
./dataset
| - icon1.png
| - icon2.png
| ...

Outputs:
./preprocessed_data
| - contour
| | - 00000.png
| | - 00001.png
| | ...
| - img
| | - 00000.png
| | - 00001.png
| | ...
| - labels.pt
"""

EXTS = ['.png', '.jpg', '.jpeg', '.bmp']
def allowed_extension(path):
	target = os.path.splitext(path)[1]
	return any(ext == target for ext in EXTS)

def open_img(path):
	"""
	Typically an icon image contains Alpha channel
	We paste the icon to a white background image to eliminate the alpha channel
	if we directly discard alpha channel there'll be some artifact.
	"""
	img = Image.open(path).convert('RGBA')
	assert img.size[0] == img.size[1], "Cannot handle rectangular icon image"
	nimg = Image.new(mode='RGBA', size=img.size, color=(255,255,255,255))
	nimg.paste(img, mask=img.split()[3])
	return nimg.convert('RGB')

def get_Lab_3dhist(img):
	"""
	return 3d Lab histogram 
	shape = (bins, bins, bins)
	"""
	sample = np.array(img)
	non_white_pixels = color.rgb2lab(sample)[np.any(sample != [255, 255, 255], axis=2), :]
	# L: 0 to 100, a: -127 to 128, b: -128 to 127.
	H, _ = np.histogramdd(non_white_pixels, bins=(8, 8, 8), range=[(0, 100), (-127, 128), (-128, 127)])
	return H

def get_resized_and_contour(img, args):
	"""
	return the contour and icon in resized target resolution
	(image padding included)
	"""
	pad = img.size[0] // args.pad_ratio
	img = ImageOps.expand(img, (pad, pad, pad, pad), fill=(255, 255, 255))

	contour = feature.canny(np.array(img.convert('L')), sigma=args.canny_sigma)
	contour = contour.astype('float32') * 255
	k = np.ones((args.contour_k, args.contour_k))
	contour = ndimage.convolve(contour, k, mode='constant', cval=0.0)
	contour = Image.fromarray(contour.astype('uint8'), 'L')

	resolution = args.resolution + (args.resolution // args.pad_ratio) * 2
	img = img.resize((resolution, resolution), Image.BICUBIC)

	contour = contour.resize((resolution, resolution), Image.BICUBIC)
	return img, contour

def main():
	parser = ArgumentParser()
	parser.add_argument('--dataset', type=str, required=True)
	parser.add_argument('--save_path', type=str, default='preprocessed_data')
	parser.add_argument('--n_clusters', type=int, default=500)
	parser.add_argument('--resolution', type=int, default=128)
	parser.add_argument('--pad_ratio', type=int, default=8)
	parser.add_argument('--contour_k', type=int, default=5)
	parser.add_argument('--bins', type=int, default=8)
	parser.add_argument('--canny_sigma', type=float, default=1)
	args = parser.parse_args()
	print(vars(args))

	files = [os.path.join(args.dataset, f) for f in sorted(filter(allowed_extension, os.listdir(args.dataset)))]

	img_root = os.path.join(args.save_path, 'img')
	contour_root = os.path.join(args.save_path, 'contour')

	if os.path.exists(args.save_path):
		shutil.rmtree(args.save_path)
	os.mkdir(args.save_path)
	os.mkdir(img_root)
	os.mkdir(contour_root)

	histograms = []
	for idx, f in enumerate(tqdm(files, ascii=True, ncols=100)):
		img = open_img(f)

		# calc histogram
		H = get_Lab_3dhist(img)
		H = H.flatten() / H.sum()
		histograms.append(H)

		# save processed image
		resized_img, contour = get_resized_and_contour(img, args)

		img_path = os.path.join(img_root, '{:06d}.png'.format(idx))
		contour_path = os.path.join(contour_root, '{:06d}.png'.format(idx))
		resized_img.save(img_path)
		contour.save(contour_path)
	histograms = np.stack(histograms)

	# kmeans
	print('Kmeans n_clusters =', args.n_clusters)
	kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(histograms)
	labels = kmeans.labels_
	groups = [[] for _ in range(args.n_clusters)]
	for idx, label in enumerate(labels):
		groups[label].append(idx)
	torch.save({
		'labels': labels,
		'groups': groups
	}, os.path.join(args.save_path, 'labels.pt'))

if __name__ == '__main__':
	main()
	print('done.')
