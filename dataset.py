import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as tvF

class RandomResizedCrop(transforms.RandomResizedCrop):
	def __init__(self, size, scale=(0.84, 0.9), ratio=(1.0, 1.0), interpolation=Image.BICUBIC):
		super(RandomResizedCrop, self).__init__(size, scale, ratio, interpolation)

	def __call__(self, img1, img2):
		assert img1.size == img2.size
		# fix parameter
		i, j, h, w = self.get_params(img1, self.scale, self.ratio)
		# return the image with the same transformation

		img1 = tvF.resized_crop(img1, i, j, h, w, self.size, self.interpolation)
		img2 = tvF.resized_crop(img2, i, j, h, w, self.size, self.interpolation)
		return img1, img2

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
	def __call__(self, img1, img2):
		assert img1.size == img2.size

		p = random.random()
		if p < 0.5:
				img1 = tvF.hflip(img1)
		if p < 0.5:
				img2 = tvF.hflip(img2)
		return img1, img2

class RandomVerticalFlip(transforms.RandomVerticalFlip):
	def __call__(self, img1, img2):
		assert img1.size == img2.size

		p = random.random()
		if p < 0.5:
				img1 = tvF.vflip(img1)
		if p < 0.5:
				img2 = tvF.vflip(img2)
		return img1, img2


class IconDataset(Dataset):
	def __init__(self, root, resolution=128, pad_ratio=8):
		self.resolution = resolution
		self.min_crop_area = ((pad_ratio+1) / (pad_ratio+2)) ** 2

		self.root = root
		self.contour_dir = os.path.join(root, 'contour')
		self.img_dir = os.path.join(root, 'img')

		labels = torch.load(os.path.join(root, 'labels.pt'))
		self.labels = labels['labels']
		self.groups = labels['groups']

		self.idxs = list(range(len(self.labels)))
		self.size = len(self.idxs)
		self.icon_paths = [os.path.join(self.img_dir, '{:06d}.png'.format(i)) for i in self.idxs]
		self.contour_paths = [os.path.join(self.contour_dir, '{:06d}.png'.format(i)) for i in self.idxs]

		self.norm = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])

		self.style_img_aug = transforms.Compose([
			transforms.RandomResizedCrop(resolution, scale=(self.min_crop_area, 1.0), ratio=(1.0, 1.0)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
		])

		self.paired_aug = [
			RandomResizedCrop(resolution, scale=(self.min_crop_area, 1.0), ratio=(1.0, 1.0)),
			RandomHorizontalFlip(),
			RandomVerticalFlip(),
		]

	def __len__(self):
		return self.size

	def __getitem__(self, idx1):
		"""
		returns s1, s2, s3, contour such that
		s1, s2 are in the same cluster
		s3, contour are paired icon and it's contour
		note that s3 can be in different cluster
		"""
		label = self.labels[idx1]
		group = self.groups[label]

		# pick the icon in the same color cluster
		idx2 = random.choice(group)
		idx3 = random.choice(self.idxs)

		s1 = Image.open(self.icon_paths[idx1]).convert('RGB')
		s2 = Image.open(self.icon_paths[idx2]).convert('RGB')
		s3 = Image.open(self.icon_paths[idx3]).convert('RGB')
		contour = Image.open(self.contour_paths[idx3]).convert('RGB')

		s1 = self.style_img_aug(s1)
		s2 = self.style_img_aug(s2)

		for aug in self.paired_aug:
			s3, contour = aug(s3, contour)

		s1 = self.norm(s1)
		s2 = self.norm(s2)
		s3 = self.norm(s3)
		contour = self.norm(contour)

		return s1, s2, s3, contour[:1, :, :]
