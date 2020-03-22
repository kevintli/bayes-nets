from typing import Dict
import random
from scipy.stats import multivariate_normal

class DiscreteDistribution:
    def __init__(self, probabilities, needs_normalize=False):
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
        if value not in self.probs:
            raise Exception(f"[DiscreteDistribution] Attempted to get probability of nonexistent value {value}")
        return self.probs[value]

    def sample(self):
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