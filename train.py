import argparse
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import opt
from evaluation import evaluate
from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor
from dataloader import MaskDataset
from dataloader import SpecificValDataset
from util.io import load_ckpt
from util.io import save_ckpt
import config as cfg

import torch.multiprocessing as mp
#mp.set_start_method('spawn')

class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


cfg.set_train_args()

if not os.path.exists(cfg.save_dir):
    os.makedirs('{:s}/images'.format(cfg.save_dir))
    os.makedirs('{:s}/ckpt'.format(cfg.save_dir))

if not os.path.exists(cfg.log_dir):
    os.makedirs(cfg.log_dir)
writer = SummaryWriter(log_dir=cfg.log_dir)

size = (cfg.image_size, cfg.image_size)
#size = (256, 256)
img_tf = transforms.Compose(
    [transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_tf = transforms.Compose(
    [transforms.ToTensor()])

if cfg.depth:
    depth = True
else:
    depth = False

dataset_train = MaskDataset(depth, cfg.in_channels, cfg.mask_year, cfg.im_year, mode='train')
dataset_val = MaskDataset(depth, cfg.in_channels, cfg.mask_year, cfg.im_year, mode='val')

iterator_train = iter(data.DataLoader(dataset_train, 
    batch_size= cfg.batch_size, sampler=InfiniteSampler(len(dataset_train)),
    num_workers= cfg.n_threads))
print(len(dataset_train))

model = PConvUNet().to(cfg.device)

if cfg.finetune:
    lr = cfg.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = cfg.lr

start_iter = 0
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = InpaintingLoss(VGG16FeatureExtractor()).to(cfg.device)

if cfg.resume_iter:
    start_iter = load_ckpt(
        '{}/ckpt/{}.pth'.format(cfg.save_dir, cfg.resume_iter), [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

for i in tqdm(range(start_iter, cfg.max_iter)):
    model.train()
    image, mask, gt = [x.to(cfg.device) for x in next(iterator_train)]
    output, _ = model(image, mask)
    loss_dict = criterion(image, mask, output, gt)

    loss = 0.0
    for key, coef in opt.LAMBDA_DICT.items():
        value = coef * loss_dict[key]
        loss += value
        if (i + 1) % cfg.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % cfg.save_model_interval == 0 or (i + 1) == cfg.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format( cfg.save_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)

    if (i + 1) % cfg.vis_interval == 0:
        model.eval()
        evaluate(model, dataset_val, cfg.device,
                 '{:s}/images/{:s}/test_{:d}'.format( cfg.save_dir, cfg.save_part, i + 1))

writer.close()
