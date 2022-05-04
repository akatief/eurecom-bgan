import argparse
import os
import numpy as np

from scipy.linalg import sqrtm

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch


def _sqrtm(input):
    np_matrix = input.detach().cpu().numpy().astype(np.float_)
    output = torch.from_numpy(sqrtm(np_matrix).real).to(input)
    return output


def frechet_inception_distance(generated, world):
    '''
    Computes FID between generated samples and real-world ones

    :param generated: generated samples
    :param world: real samples
    :return:
    '''
    mu_g = torch.mean(generated, 0).flatten()
    mu_w = torch.mean(world, 0).flatten()
    cov_g = torch.cov(generated.flatten(start_dim=1, end_dim=3).T)
    cov_w = torch.cov(world.flatten(start_dim=1, end_dim=3).T)

    return (mu_g - mu_w).dot(mu_g - mu_w) + torch.trace(cov_w) + torch.trace(cov_g) - 2 * torch.trace(
        _sqrtm(torch.linalg.matmul(cov_w, cov_g)))


class Gan:
    def __init__(self, generator, discriminator, lr, dataset_name = "mnist", loss_name = ''):
        self.G = generator
        self.D = discriminator
        # Optimizers
        self.optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
        self.optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
        self.dataset_name = dataset_name
        self.loss_name = loss_name
        cuda = torch.cuda.is_available()
        if cuda:
            self.G.cuda()
            self.D.cuda()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def train(self, dataloader, n_epochs, clip_value, n_critic, sample_interval):
        batches_done = 0
        list_loss_G = []
        list_loss_D = []
        list_Frech_dist = []

        for epoch in range(n_epochs):

            for i, (imgs, _) in enumerate(dataloader):

                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.G.latent_dim))))

                # Generate a batch of images
                fake_imgs = self.G(z).detach()
                # Adversarial loss
                scores_real = self.D(real_imgs)
                scores_fake = self.D(fake_imgs)

                loss_D = self.D.loss(scores_real, scores_fake)

                loss_D.backward()
                self.optimizer_D.step()

                # Clip weights of discriminator
                for p in self.D.parameters():
                    p.data.clamp_(-clip_value, clip_value)

                # Train the generator every n_critic iterations
                if i % n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    self.optimizer_G.zero_grad()

                    # Generate a batch of images
                    gen_imgs = self.G(z)

                    loss_G = self.G.loss(self.D(gen_imgs))

                    loss_G.backward()
                    self.optimizer_G.step()
                    if batches_done % 50 == 0:
                        print(f"[Epoch {epoch}/{n_epochs}] [Batch {batches_done % len(dataloader)}/{len(dataloader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")
                    list_loss_G.append(loss_G.item())
                    list_loss_D.append(loss_D.item())


                if batches_done % sample_interval == 0:
                    # Frechet Inception Distance
                    fid = frechet_inception_distance(fake_imgs, real_imgs)
                    print('[FID %f]' % fid.item())
                    list_Frech_dist.append(fid.item())
                    os.makedirs(f"images/{self.dataset_name}/{self.loss_name}", exist_ok=True)
                    save_image(fake_imgs.data[:25], f"images/{self.dataset_name}/{batches_done}.png", nrow=5, normalize=True)
                batches_done += 1

        return list_loss_G,list_loss_D,list_Frech_dist