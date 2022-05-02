import torch
from nets.base_net import BaseDiscriminator, BaseGenerator


class BGanDiscriminator(BaseDiscriminator):
    def __init__(self, img_shape):
        super().__init__(img_shape)

    def loss(self, scores_real, scores_fake):
        return torch.mean(0.5 * torch.square(scores_fake) - 0.5) - torch.mean(scores_real - 1)


class BGanGenerator(BaseGenerator):
    def __init__(self, img_shape, latent_dim):
        super().__init__(img_shape, latent_dim)

    def loss(self, scores):
        return torch.mean(0.5 * torch.square(scores - 1))
