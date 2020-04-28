import torch
import torch.nn as nn

from distributions import Distribution
from fitting import LinearGaussianConditionalFn, learn_gaussian_conditional_fn

class CPD(Distribution):
    """
    Base class for a conditional probability distribution.

    We should be able to query for probabilties and sample from the distribution
        conditioned on a set of evidence variables.
    """
    def __init__(self):
        Distribution.__init__(self)

    def get_probability(self, x, evidence):
        """
        Returns P(X=x|e_1, e_2, ..., e_n)

        Params
        - x                  any object (e.g. an integer, string, or tuple of values)
        - evidence (tuple)   values for each evidence variable
        """
        raise NotImplementedError

    def sample(self, evidence, num_samples=1):
        """
        Returns an array of `num_samples` samples from the distribution P(X|evidence).

        Params
        - evidence (tuple)      values for each evidence variable
        - num_samples (int)     number of samples to return
        """
        raise NotImplementedError


class TabularCPD(CPD):
    """
    Represents a tabular CPD, where we manually specify the conditional distribution
        for each possible combination of evidence variables.

    Example:
        t = TabularCPD({
            (0, 0): DiscreteDistribution({0: 0.5, 1: 0.5}),  # Represents P(X | e_1=0, e_2=0)
            (0, 1): DiscreteDistribution({0: 0.2, 1: 0.8}),  # Represents P(X | e_1=0, e_2=1)
            (1, 0): DiscreteDistribution({0: 0.9, 1: 0.1}),  # Represents P(X | e_1=1, e_2=0)
            (1, 1): DiscreteDistribution({0: 0.2, 1: 0.8})   # Represents P(X | e_1=1, e_2=1)
        })

        t.get_probability(0, (0, 1))  # => 0.2
        t.sample((1, 1))              # Samples from P(X | e_1=1, e_2=1)

    Only the evidence variable values need to be enumerated; the resulting conditional for a particular
        set of evidence values can be set to any arbitrary distribution. For example, we can specify P(X|A, B) 
        where A, B take on discrete values listed in the table, but the resulting X|A, B distributions are Gaussian.
    """

    def __init__(self, probabilities):
        """
        probabilities (dict): each key-value pair maps a tuple of numerical quantities for e_1, ..., e_n 
                                to a Distribution object representing P(X | e_1, ..., e_n)

        """
        CPD.__init__(self)
        self._assert_valid_conditionals(probabilities)
        self.conditional_probs = probabilities
        self.num_evidence_vars = self._get_dim(list(probabilities.keys())[0])

    def get_probability(self, x, evidence):
        """
        Returns P(x|evidence), assuming it exists in the table.
        """
        self._assert_valid_evidence(evidence)
        return self.conditional_probs[evidence].get_probability(x)

    def sample(self, evidence, num_samples=1):
        self._assert_valid_evidence(evidence)
        return self.conditional_probs[evidence].sample(num_samples)

    @staticmethod
    def fit_to_data(data, x_vals=None):
        """
        Returns a TabularCPD object whose probabilities are set according to the frequencies of
        each value in the provided dataset.

        Params
        - data   (np.array):        A numpy array with data points of any type (must be hashable)
        - x_vals (list, optional):  All possible values this distribution can take on. If specified,
                                     any other values in the dataset will be ignored.
        """
        pass

    def _assert_valid_conditionals(self, probs):
        assert probs, "[TabularCPD] Cannot create an empty distribution"
        assert [self._get_dim(key) for key in probs].count(self._get_dim(list(probs.keys())[0])) == len(probs), \
            "[TabularCPD] Must condition on the same number of evidence variables for each case"

    def _get_dim(self, val):
        return 1 if not isinstance(val, tuple) else len(val)

    def _assert_valid_evidence(self, evidence):
        if evidence not in self.conditional_probs:
            raise Exception(f"[TabularCPD] values for evidence variables not found in table: {evidence}")


class GaussianCPD(CPD):
    """
    Represents a general Gaussian CPD, where we specify some parameterized function that maps values for evidence variables
        to the mean and covariance of a Gaussian.
    """
    def __init__(self, cond_fn, linear_coeffs=None, sd=None):
        """
        Params
            cond_fn - a function that takes in values for evidence variables and returns a GaussianDistribution object

        Example:
        g = GaussianCPD(self, lambda z: GaussianDistribution(z, 1))  # P(X|z) is a unit Gaussian centered at z

        g.get_probability(5.3, 5)  # => 0.38138781546052414
        g.sample(3)                # Samples from P(X | z=3)
        """
        CPD.__init__(self)
        self.cond_fn = cond_fn
        self.linear_coeffs = linear_coeffs
        self.sd = sd

    def get_probability(self, x, evidence):
        return self.cond_fn(evidence).get_probability(x)

    def get_log_prob(self, x, evidence):
        return self.cond_fn(evidence).get_log_prob(x)

    def sample(self, evidence, num_samples=1):
        return self.cond_fn(evidence).sample(num_samples)

class LinearGaussianCPD(GaussianCPD):
    def __init__(self, linear_coeffs, cov):
        """
        Parameters
        ----------
        linear_coeffs : list[tuple[torch.tensor, torch.tensor]]
            A list of coefficients, one for each evidence variable.
            Each coefficient is a 2-element tuple with a weight and bias.

        cov : torch.tensor
            The covariance, which is fixed regardless of evidence.
        """
        GaussianCPD.__init__(self, self.cond_fn)
        self.weights = [coeff[0] for coeff in linear_coeffs]
        self.biases = [coeff[1] for coeff in linear_coeffs]
        self.cov = cov

    def cond_fn(self, evidence):
        evidence = evidence.float()

        def mult(weight, evidnece):
            if isinstance(weight, (int, float)):
                return weight * evidence
            if isinstance(weight, torch.Tensor) and weight.shape == 0:
                return weight * evidence
            else:
                return weight @ evidence

        mean = sum([mult(self.weights[i], evidence) + self.biases[i] for i, evidence in enumerate(evidence)])
        cov = self.cov
        return GaussianDistribution(mean, cov)

    @staticmethod
    def empty(evidence_dims, data_dim):
        """
        Returns an uninitialized linear Gaussian CPD whose weights can be learned (e.g. using MLE or VI).
        This allows you to set up the structure of a Bayes Net without having to specify values yet.

        Parameters
        ----------
        evidence_dims : list[int]
            A list of dimensions for each of the evidence variables
        
        data_dim : int
            The dimensionality of the distribution
        """
        return EmptyGaussianCPD(
            # Initialize a cond_fn with random linear weights and biases
            LinearGaussianConditionalFn(evidence_dims, data_dim)
        )

class EmptyGaussianCPD(GaussianCPD):
    def __init__(self, cond_fn):
        assert isinstance(cond_fn, nn.Module), "cond_fn for EmptyGaussianCPD must be an instance of nn.Module"
        GaussianCPD.__init__(self, cond_fn)
        self.is_learnable = True

    def learnable_params(self):
        return list(self.cond_fn.parameters())

    def freeze_values(self):
        for param in self.cond_fn.parameters():
            param.requires_grad_(False)
        self.is_learnable = False

    def fit_to_data(self, evidence, data):
        learn_gaussian_conditional_fn(self.cond_fn, evidence, data)
        self.freeze_values()