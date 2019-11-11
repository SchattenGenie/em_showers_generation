import torch
import torch.nn as nn
import numpy as np
import pyro
import pyro.distributions as dist


def logsumexp(x, dim):
    x_max, _ = x.max(dim=dim,keepdim=True)
    x_max_expand = x_max.expand(x.size())
    res = x_max + torch.log((x-x_max_expand).exp().sum(dim=dim, keepdim=True))
    return res


class GaussianMixtureNetwork(nn.Module):
    def __init__(self, input_dim, mixture_size, targets, hidden_dim=32):
        super(GaussianMixtureNetwork, self).__init__()
        self.input_dim = input_dim
        self.mixture_size = mixture_size
        self.targets = targets
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.alphas = nn.Linear(hidden_dim, mixture_size)
        self.sigmas = nn.Linear(hidden_dim, mixture_size * targets)
        self.means = nn.Linear(hidden_dim, mixture_size * targets)

    def forward(self, inputs):
        hidden = self.nn(inputs)
        alphas = self.alphas(hidden).view(-1, self.mixture_size)
        log_sigmas = self.sigmas(hidden).view(-1, self.targets, self.mixture_size)
        means = self.means(hidden).view(-1, self.targets, self.mixture_size)
        return alphas, means, torch.clamp_min(log_sigmas, -5)

    def logits(self, inputs, target):
        """
        inputs = [N, input_dim]
        target = [N, K]
        """
        # alphas, means, sigmas = [N, K, mixture_size]
        alphas, means, log_sigmas = self.forward(inputs)

        log_alphas = self.logsoftmax(alphas)
        log_pdf = -0.5 * np.log(np.pi * 2) - log_sigmas - ((target.unsqueeze(-1) - means) / log_sigmas.exp()).pow(2) / 2.
        logits = logsumexp(log_alphas + log_pdf.sum(dim=1), dim=-1).view(-1)
        return logits

    def generate(self, inputs):
        # alphas, means, sigmas = [N, K, mixture_size]
        alphas, means, log_sigmas = self.forward(inputs)
        alphas = self.softmax(alphas)
        # alphas_picked = [N, K]
        alphas_sampled = pyro.sample("alphas", dist.Categorical(probs=alphas))
        # print(alphas_sampled)
        sigmas = log_sigmas.exp()
        result = pyro.sample("preds", dist.Normal( # Laplace
            torch.gather(
                input=means,
                dim=2,
                index=alphas_sampled.view(-1, 1).repeat(1, self.targets).view(-1, self.targets, 1)
            ).view(-1, self.targets),
            torch.gather(
                input=sigmas,
                dim=2,
                index=alphas_sampled.view(-1, 1).repeat(1, self.targets).view(-1, self.targets, 1)
            ).view(-1, self.targets)
        ))
        return result

    def generate_mll(self, inputs):
        # alphas, means, sigmas = [N, K, mixture_size]
        alphas, means, log_sigmas = self.forward(inputs)
        alphas = self.softmax(alphas)
        # alphas_picked = [N, K]
        alphas_sampled = pyro.sample("alphas", dist.Categorical(probs=alphas))
        # print(alphas_sampled)
        sigmas = log_sigmas.exp()
        result = torch.gather(
                input=means,
                dim=2,
                index=alphas_sampled.view(-1, 1).repeat(1, self.targets).view(-1, self.targets, 1)
            ).view(-1, self.targets)
        return result
