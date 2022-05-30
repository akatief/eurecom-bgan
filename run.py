import argparse
import os

import pandas as pd

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets

import torch

from gan import Gan
from nets.bgan_net import BGanGenerator, BGanDiscriminator
from nets.wasserstein_net import WassersteinGenerator, WassersteinDiscriminator

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
# parser.add_argument('-f') # uncomment to run on colab
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=10, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

os.makedirs("data/mnist", exist_ok=True)
mnist_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)


#%%

os.makedirs("data/celebA", exist_ok=True)
''' celeba_dataloader = torch.utils.data.DataLoader(
    datasets.CelebA(
        "data/celebA",
        #split="training",
        download=True, #True gives a Badzip error
        # download the file from
        # https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
        # and save in the above folder
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
'''


dataloader = mnist_dataloader #or celeba_dataloader

bGan_G = BGanGenerator(img_shape, opt.latent_dim)
bGan_D = BGanDiscriminator(img_shape)

bGan = Gan(bGan_G, bGan_D, opt.lr, dataset_name="mnist",loss_name="bgan") #mnist or celeba
b_list_loss_G,b_list_loss_D,b_list_Frech_dist = bGan.train(dataloader, opt.n_epochs, opt.clip_value, opt.n_critic, opt.sample_interval)



wasserstein_G = WassersteinGenerator(img_shape, opt.latent_dim)
wasserstein_D = WassersteinDiscriminator(img_shape)

wasserstein_Gan = Gan(wasserstein_G, wasserstein_D, opt.lr,dataset_name="mnist",loss_name="wgan") #mnist or celeba
w_list_loss_G,w_list_loss_D,w_list_Frech_dist = wasserstein_Gan.train(dataloader, opt.n_epochs, opt.clip_value, opt.n_critic, opt.sample_interval)



b_loss_dict = {'b_loss_G': b_list_loss_G, 'b_loss_D' : b_list_loss_D,  }
df_b = pd.DataFrame(b_loss_dict)
df_b.to_csv(path_or_buf='results/bGAN.csv')

wass_loss_dict = {'wass_loss_G': w_list_loss_G, 'wass_loss_D' : w_list_loss_D,  }
df_w = pd.DataFrame(wass_loss_dict)
df_w.to_csv(path_or_buf='results/WGAN.csv')

dict_frech_dist = {'b_frech_dist':b_list_Frech_dist,'wass_frech_dist':w_list_Frech_dist}
df_frech_dist = pd.DataFrame(dict_frech_dist)
df_frech_dist.to_csv(path_or_buf='results/frech_dist.csv')

df = pd.read_csv('results/frech_dist.csv')
fig = px.line(df, y = ['wass_frech_dist','b_frech_dist'], title='Frechet Distance')
fig.show()

df1 = pd.read_csv('results/bGAN.csv')
df2= pd.read_csv('results/WGAN.csv')
df = pd.concat([df1, df2], axis=1)
print(df)
fig = px.line(df, y = ["b_loss_D","b_loss_G","wass_loss_G","wass_loss_D"], title='Wass Loss')
fig.show()


