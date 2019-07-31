import os
import sys
import time
import shutil
from argparse import ArgumentParser
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from network import Generator, Discriminator
from dataset import IconDataset

# argparse
parser = ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--lrG', type=float, default=1e-4)
parser.add_argument('--lrD', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--resolution', type=int, default=128)
parser.add_argument('--pad_ratio', type=int, default=8)
parser.add_argument('--log_iter', type=int, default=20)
parser.add_argument('--save_iter', type=int, default=2000)
parser.add_argument('--save_path', type=str, default='weights')
parser.add_argument('--sample_path', type=str, default='samples')
parser.add_argument('--dummy_input', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--cuda', action='store_true', default=True)
args = parser.parse_args()
print(vars(args))

# helpers

def clear_dir(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.mkdir(path)

# hinge loss
def dis_loss(D, real, fake):
	d_out_real = D(real)
	d_out_fake = D(fake.detach())
	loss_real = F.relu(1.0 - d_out_real).mean()
	loss_fake = F.relu(1.0 + d_out_fake).mean()
	return loss_real + loss_fake

def gen_loss(D, fake):
	d_out = D(fake)
	return -(d_out).mean()

# device
if torch.cuda.is_available() and args.cuda:
	device = torch.device('cuda')
	torch.backends.cudnn.benchmark = True
	print('cudnn benchmark enabled')
else:
	device = torch.device('cpu')
print('Device:', device)

# load dataset
# construct networks
G = Generator(ch_style=3, ch_content=1).to(device)
Ds = Discriminator(3+3).to(device)
Dc = Discriminator(3+1).to(device)

if args.dummy_input: # debug purpose
	BATCH_SIZE = 16
	s1 = torch.randn(BATCH_SIZE, 3, 128, 128).to(device)
	contour = torch.randn(BATCH_SIZE, 1, 128, 128).to(device)
	fake = G(s1, contour)
	print('fake.shape', fake.shape)
	ds_out = Ds(torch.cat([fake, s1]))
	dc_out = Dc(torch.cat([fake, contour]))
	print('Ds_out.shape', ds_out.shape)
	print('Dc_out.shape', dc_out.shape)
	sys.exit(0)

optimG = optim.Adam(G.parameters(), lr=args.lrG, betas=(0, 0.999))
optimDc = optim.Adam(Dc.parameters(), lr=args.lrD, betas=(0, 0.999))
optimDs = optim.Adam(Ds.parameters(), lr=args.lrD, betas=(0, 0.999))

# prepare dataset
dataset = IconDataset(root=args.dataset, resolution=args.resolution, pad_ratio=args.pad_ratio)
dataloader = torch.utils.data.DataLoader(dataset,
	batch_size=args.batch_size,
	shuffle=True,
	drop_last=True,
	num_workers=4,
	pin_memory=True,
)

# sample fixed inputs
for s1, s2, s3, contour in dataloader:
	break

fixed_s1 = s1.to(device)
fixed_contour = contour.to(device)
vutils.save_image(fixed_s1.detach().cpu(), 's1.png', normalize=True, range=(-1, 1), padding=0)
vutils.save_image(s2.detach().cpu(), 's2.png', normalize=True, range=(-1, 1), padding=0)
vutils.save_image(s3.detach().cpu(), 's3.png', normalize=True, range=(-1, 1), padding=0)
vutils.save_image(fixed_contour.detach().cpu(), 'contour.png', normalize=True, range=(-1, 1), padding=0)

# training loop
print('training...')
writer = SummaryWriter()
global_step = 0
start_epoch = 0
timestamp = time.time()
# check resume

if args.resume:
	save_path = os.path.join(
		args.save_path,
		'latest_{}.pth'
	)

	G.load_state_dict(torch.load(save_path.format('G')))
	Ds.load_state_dict(torch.load(save_path.format('Ds')))
	Dc.load_state_dict(torch.load(save_path.format('Dc')))
	optimG.load_state_dict(torch.load(save_path.format('optimG')))
	optimDs.load_state_dict(torch.load(save_path.format('optimDs')))
	optimDc.load_state_dict(torch.load(save_path.format('optimDc')))
	global_step, start_epoch = torch.load(save_path.format('state'))
	print('resumed from Epoch: {:04d} Step: {:07d}'.format(start_epoch, global_step))
else:
	clear_dir(args.sample_path)
	clear_dir(args.save_path)

for epoch in range(start_epoch, args.epochs):
	for s1, s2, s3, contour in dataloader:
		# s1 s2 are in same cluster in lab space
		# s3 contour are paired icon and it's contour
		global_step += 1

		s1 = s1.to(device)
		s2 = s2.to(device)
		s3 = s3.to(device)
		contour = contour.to(device)

		fake = G(s1, contour)
		style_fake = torch.cat([fake, s2], dim=1)
		style_real = torch.cat([s1, s2], dim=1)
		content_fake = torch.cat([fake, contour], dim=1)
		content_real = torch.cat([s3, contour], dim=1)
		
		# update style discriminator
		optimDs.zero_grad()
		Ds_loss = dis_loss(Ds, style_real, style_fake)
		Ds_loss.backward()
		optimDs.step()

		# update content discriminator
		optimDc.zero_grad()
		Dc_loss = dis_loss(Dc, content_real, content_fake)
		Dc_loss.backward()
		optimDc.step()

		# update generator
		optimG.zero_grad()
		Gs_loss = gen_loss(Ds, style_fake)
		Gc_loss = gen_loss(Dc, content_fake)
		G_loss = Gs_loss + Gc_loss
		G_loss.backward()
		optimG.step()

		# log losses
		writer.add_scalars('loss', {
			'D_style_loss': Ds_loss.item(),
			'D_content_loss': Dc_loss.item(),
			'G_style_loss': Gs_loss.item(),
			'G_content_loss': Gc_loss.item(),
			'G_loss': G_loss.item(),
		}, global_step=global_step)

		if global_step % args.log_iter == 0:
			curTime = time.time()
			print('Epoch: {:04d} Step: {:07d} Elapsed Time: {:.3f}s Ds: {:.5f} Dc: {:.5f} G: {:.5f}'.format(
				epoch, global_step,
				curTime - timestamp,
				Ds_loss.item(),
				Dc_loss.item(),
				G_loss.item(),
			))
			timestamp = curTime

		if global_step % args.save_iter == 0:
			save_idx = global_step // args.save_iter
			for prefix in ['{:05d}'.format(save_idx), 'latest']:
				save_path = os.path.join(
					args.save_path,
					prefix
				)
				save_path += '_{}.pth'
				torch.save(G.state_dict(), save_path.format('G'))
				torch.save(Ds.state_dict(), save_path.format('Ds'))
				torch.save(Dc.state_dict(), save_path.format('Dc'))
				torch.save(optimG.state_dict(), save_path.format('optimG'))
				torch.save(optimDs.state_dict(), save_path.format('optimDs'))
				torch.save(optimDc.state_dict(), save_path.format('optimDc'))
				torch.save((global_step, epoch), save_path.format('state'))
			
			G.eval()
			with torch.no_grad():
				fixed_fake = G(fixed_s1, fixed_contour)
			G.train()
			save_path = os.path.join(args.sample_path, '{:05d}.png'.format(save_idx))
			vutils.save_image(fixed_fake.detach().cpu(), save_path, normalize=True, range=(-1, 1), padding=0)
			
			print('log {:05d} saved'.format(save_idx))
