import argparse
import os

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets

import torch

from gan import Gan
from nets.bgan_net import BGanGenerator, BGanDiscriminator

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
# parser.add_argument('-f') # uncomment to run on colab
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)


bGan_G = BGanGenerator(img_shape, opt.latent_dim)
bGan_D = BGanDiscriminator(img_shape)

bGan = Gan(bGan_G, bGan_D, opt.lr)
bGan.train(dataloader, opt.n_epochs, opt.clip_value, opt.n_critic, opt.sample_interval)
