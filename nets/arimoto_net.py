import torch
from nets.base_net import BaseDiscriminator, BaseGenerator


class ArimotoDiscriminator(BaseDiscriminator):
    def __init__(self, img_shape, alpha=.5):
        super().__init__(img_shape)
        self.alpha = alpha

    def loss(self, scores_real, scores_fake):
        k = self.alpha / (self.alpha - 1)
        k_inv = k**-1
        return torch.mean(k * (1 - scores_real**k_inv)) + torch.mean(k * (1 - (1 - scores_fake)**k_inv))


class ArimotoGenerator(BaseGenerator):
    def __init__(self, img_shape, latent_dim, alpha=1):
        super().__init__(img_shape, latent_dim)
        self.alpha = alpha

    def loss(self, scores):
        k = self.alpha / (self.alpha - 1)
        k_inv = k**-1
        return torch.mean(k * (1 - scores)**k_inv)
