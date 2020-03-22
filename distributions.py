from typing import Dict
import random
from scipy.stats import multivariate_normal

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
        total = self._total()
        for key in self.probs:
            self.probs[key] /= total

    def get_probability(self, value):
        """
        Returns the probability corresponding to a particular value.
        """
        if value not in self.probs:
            raise Exception(f"[DiscreteDistribution] Attempted to get probability of nonexistent value {value}")
        return self.probs[value]

    def sample(self):
        """
        Returns a single sample from the distribution.
        """
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
    A (possibly multivariate) Gaussian distribution, represented by a mean and covariance matrix.
    """
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.rv = multivariate_normal(mean=mean, cov=cov)

    def get_probability(self, value):
        return self.rv.pdf(value)

    def sample(self, shape=None):
        return self.rv.rvs(shape)
    
    def __str__(self):
        return f"GaussianDistribution(mean: {self.mean}, cov: {self.cov})"