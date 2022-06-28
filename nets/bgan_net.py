import torch
from nets.base_net import BaseDiscriminator, BaseGenerator


def f(x, alpha=1):
    if alpha == 1 or alpha == -1:
        return x * torch.log(x) - x + 1
    elif alpha == -1:
        return -torch.log(x) + x - 1
    else:
        return 4 / (1 - alpha ** 2) * (1 - torch.pow(x, (1 + alpha) / 2)) + 2 / (1 - alpha) * (x - 1)


def df(x, alpha=1):
    if alpha == 1:
        return torch.log(x)
    elif alpha == -1:
        return (x - 1) / x
    else:
        return 2 / (x * (alpha - 1)) * (torch.pow(x, (alpha + 1) / 2) - x)


class BGanDiscriminator(BaseDiscriminator):
    def __init__(self, img_shape, alpha=1):
        super().__init__(img_shape)
        self.alpha = alpha

    def loss(self, scores_real, scores_fake):
        #scores_real, scores_fake = scores_real * 2, scores_fake * 2
        # return torch.mean(0.5 * torch.square(scores_fake) - 0.5) - torch.mean(scores_real - 1)
        # return torch.mean(scores_fake - 1) - torch.mean(torch.log(scores_real))
        return torch.mean(df(scores_fake, self.alpha) * scores_fake) - torch.mean(
            f(scores_fake, alpha=self.alpha)) - torch.mean(df(scores_real, alpha=self.alpha))


class BGanGenerator(BaseGenerator):
    def __init__(self, img_shape, latent_dim, alpha=1):
        super().__init__(img_shape, latent_dim)
        self.alpha = alpha

    def loss(self, scores):
        #scores = scores * 2
        # return torch.mean(0.5 * torch.square(scores - 1))
        # return torch.mean(scores * torch.log(scores) - scores + 1)

        return torch.mean(f(scores, alpha=self.alpha))
