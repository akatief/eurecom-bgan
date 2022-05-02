import torch
from nets.base_net import BaseDiscriminator, BaseGenerator


class WassersteinDiscriminator(BaseDiscriminator):
    def __init__(self, img_shape):
        super.__init__(img_shape)

    def loss(self, scores_real, scores_fake):
        return -torch.mean(scores_real) + torch.mean(scores_fake)


class WassersteinGenerator(BaseGenerator):
    def __init__(self, img_shape, latent_dim):
        super.__init__(img_shape, latent_dim)

    def loss(self, scores):
        return -torch.mean(scores)
