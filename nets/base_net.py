from abc import abstractmethod

import numpy as np
import torch.nn as nn


class BaseDiscriminator(nn.Module):
    def __init__(self, img_shape):
        super(BaseDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Sigmoid added in accordance to bGAN paper
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

    @abstractmethod
    def loss(self, scores_real, scores_fake):
        pass


class BaseGenerator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(BaseGenerator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

        self.img_shape = img_shape
        self.latent_dim = latent_dim

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

    @abstractmethod
    def loss(self, scores):
        pass
