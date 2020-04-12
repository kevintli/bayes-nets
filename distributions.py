import numpy as np
import random
from scipy.stats import multivariate_normal

import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

class DiscreteDistribution:
    """
    A discrete, tabular probability distribution.
    All possible values and their corresponding probabilties must be manually specified.
    """
    def __init__(self, probabilities, needs_normalize=False):
        """
        Params:
        - probabilities (dict): keys are arbitrary objects representing the values this RV can take on,
            and values are their corresponding probabilities
        
        - needs_normalize (bool): if set to True, the values in the probabilities dictionary will be 
            normalized to ensure they sum to 1.
        """
        self.probs = probabilities
        if needs_normalize:
            self.normalize()
        else:
            assert self._total() == 1, "DiscreteDistribution must have probabilities that sum to 1"

    def normalize(self):
        """
        Normalizes the distribution so that probabilities sum to 1
        """
        total = self._total()
        assert total != 0, "Cannot normalize a DiscreteDistribution with zero total probability"

        for key in self.probs:
            self.probs[key] /= total

    def get_probability(self, value):
        """
        Returns the probability corresponding to a particular value.
        """
        if value not in self.probs:
            raise Exception(f"[DiscreteDistribution] Attempted to get probability of nonexistent value {value}")
        return self.probs[value]

    def sample(self, num_samples=1):
        """
        Returns `num_samples` samples from the distribution as a numpy array.
        """
        if num_samples > 1:
            samples = np.random.choice(list(self.probs.keys()), num_samples, p=list(self.probs.values()))
            return samples

        # Single sample implementation (more efficient)
        z = random.random() * self._total()
        curr = 0
        for key, val in self.probs.items():
            curr += val
            if z <= curr:
                return key
        
        # This should never happen
        raise Exception("[DiscreteDistribution] sample() failed to return a value")

    def __str__(self):
        return f"DiscreteDistribution({str(self.probs)})"

    def _total(self):
        return sum(self.probs.values())


class GaussianDistribution:
    """
    A Gaussian distribution with a mean and diagonal covariance matrix.
    """
    def __init__(self, mean, cov_or_sd):
        """
        Params
            mean (int|float|torch.tensor)      -  Mean of the distribution
            cov_or_sd (int|float|torch.tensor) -  Std dev (if number) or covariance matrix (if tensor)
        """
        self.mean = mean
        self.cov = cov_or_sd

        if isinstance(cov_or_sd, int) or isinstance(cov_or_sd, float):
            self.rv = Normal(torch.tensor([float(mean)]), torch.tensor([float(cov_or_sd)]))
        elif len(cov_or_sd.shape) == 0:
            self.rv = Normal(mean.float(), cov_or_sd.float())
        else:
            self.rv = MultivariateNormal(mean, cov_or_sd)

    def get_probability(self, value):
        return np.e ** self.rv.log_prob(value)

    def get_log_prob(self, value):
        return self.rv.log_prob(value)

    def sample(self, num_samples=1):
        """
        Params
        - num_samples (int): The number of samples to take
        
        Returns
        - A batch of samples from the distribution as a torch.Tensor
        """
        return self.rv.sample([num_samples])

    @staticmethod
    def fit_to_data(data):
        """
        Params
            data (tensor) - A shape [N, D] batch of data

        Returns
            A GaussianDistribution object whose mean and covariance are fit using MLE on the
             provided data
        """
        mean = torch.sum(data, axis=0).float() / len(data)
        cov = 1. / len(data) * ((data - mean).T @ (data - mean))
        
        if data.shape[1] == 1:
            # 1D case: want mean and standard deviation (NOT variance) as scalars
            mean = mean.item()
            sd = cov.squeeze().item() ** 0.5
        return GaussianDistribution(mean, sd)
    
    def __str__(self):
        return f"GaussianDistribution(mean: {self.mean}, cov: {self.cov})"