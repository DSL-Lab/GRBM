import torch
import torch.nn as nn
import numpy as np
from utils import cosine_schedule


class GMMDataset(torch.utils.data.Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx, :], torch.ones(1).to(self.samples.device)


class GMM(nn.Module):
    """ Gaussian Mixture Models 
        N.B.: covariance is assumed to be diagonal
    """

    def __init__(self, w, mu, sigma):
        """
          p(x) = sum_i w[i] N(mu[i], sigma[i]^2 * I)

          config:
            w: shape K X 1, mixture coefficients, must sum to 1
            mu: shape K X D, mean
            sigma: shape K X D, (diagonal) variance 
        """
        super().__init__()
        self.register_buffer('w', w)
        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)
        self.K = w.shape[0]
        self.D = mu.shape[1]

    @torch.no_grad()
    def log_gaussian(self, x, mu, sigma):
        """ log density of single (diagonal-covariance) multivariate Gaussian"""
        return -0.5 * ((x - mu)**2 / sigma**2).sum(dim=1) - 0.5 * (
            self.D * np.log(2 * np.pi) + torch.log(torch.prod(sigma**2)))

    @torch.no_grad()
    def log_prob(self, x):
        return torch.logsumexp(
            torch.stack([
                torch.log(self.w[kk]) +
                self.log_gaussian(x, self.mu[kk], self.sigma[kk])
                for kk in range(self.K)
            ]), 0)

    @torch.no_grad()
    def sampling(self, num_samples):
        m = torch.distributions.Categorical(self.w)
        idx = m.sample((num_samples,))
        return self.mu[idx, :] + torch.randn(num_samples, self.D).to(
            self.w.device) * self.sigma[idx, :]

    @torch.no_grad()
    def langevin_sampling(self, x, num_steps=10, eta=1.0e+0, is_anneal=False):
        eta_list = cosine_schedule(eta_max=eta, T=num_steps)
        for ii in range(num_steps):
            eta_ii = eta_list[ii] if is_anneal else eta
            x = x.detach()
            x.requires_grad = True
            eng = -self.log_prob(x).sum()
            grad = torch.autograd.grad(eng, x)[0]
            x = x - eta_ii * grad + torch.randn_like(x) * np.sqrt(eta_ii * 2)

        return x.detach()
