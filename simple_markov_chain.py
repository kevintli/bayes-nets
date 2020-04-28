import numpy as np
import torch

from bayes_net import BayesNet
from conditionals import GaussianCPD, LinearGaussianCPD
from distributions import GaussianDistribution
from fitting import fit_MLE

class SimpleMarkovChain(BayesNet):
    """
    Represents a simple Markov Chain
    X_1 -> X_2 -> ... -> X_n

    The CPDs can be set using ONE of the following methods:

    (1) generate_cpds(): randomly generates parameters for linear CPDs
    (2) fit_cpds_to_data(): fits linear CPDs according to data sampled from the true joint
    """
    def __init__(self, num_nodes):
        super(SimpleMarkovChain, self).__init__(num_nodes)
    
    def generate_cpds(self, degree=1, coeff_range=[-5, 5], max_sd=2):
        vals, covs = [], []
        for i in range(1, self.num_nodes):
            # Generate random coefficients for each degree
            coeffs = []
            for deg in range(degree + 1):
                coeff = np.random.uniform(*coeff_range)
                coeffs.append(coeff)
            vals.append(tuple(coeffs))

            # Generate random SD (TODO: handle multivariate case - covariance matrix)
            sd = np.random.uniform(max_sd)
            covs.append(sd)

        return self.specify_polynomial_cpds((0, 1), vals, covs)

    def _create_polynomial_cond_fn(self, coeffs, sd):
        def cond_fn(evidence):
            mean = sum([coeff * (evidence[0] ** deg) for deg, coeff in enumerate(reversed(coeffs))])
            return GaussianDistribution(mean, sd)
        return cond_fn

    def specify_polynomial_cpds(self, prior, vals, covs):
        """
        Params
            - prior (tuple):      The mean and SD of X_1

            - vals (list[tuple]): A list of polynomial coefficients for each CPD.
                                    For example, for a chain of length 3 with quadratic Gaussian CPDs,
                                    len(vals) = 2 and each tuple in vals has 3 items.

            - covs (list[tensor]): A list of covariance matrices for each CPD.
        """
        assert len(vals) == self.num_nodes - 1
        assert len(covs) == self.num_nodes - 1

        parameters = []

        self.set_prior("X_1", GaussianDistribution(prior[0], prior[1], linear_coeffs=(0, prior[0]), sd=prior[1]))

        for i, (coeffs, cov) in enumerate(zip(vals, covs), 1):
            parameters.append({"coeffs": coeffs, "sd": cov})
            cond_fn = self._create_polynomial_cond_fn(coeffs, cov)
            self.set_parents(f"X_{i+1}", [f"X_{i}"], GaussianCPD(cond_fn, linear_coeffs=coeffs, sd=cov))

        self.build()
        self.parameters = parameters
        return parameters

    def initialize_empty_cpds(self):
        self.set_prior("X_1", GaussianDistribution.empty())
        for i in range(1, self.num_nodes):
            self.set_parents(f"X_{i+1}", [f"X_{i}"], LinearGaussianCPD.empty([1], 1))
        self.build()

    def fit_cpds_to_data(self, data, log_fn=None):
        """
        Params
            data (list[tensor]) - A list containing values for each variable, sampled from the joint distribution
            log_fn (function)   - A function that takes three arguments: the node number, epoch number, and epoch data
        """
        self.initialize_empty_cpds()
        fit_MLE(data, self)


def test_markov_chain(length=5, num_samples=10000):
    mc = SimpleMarkovChain(length)
    true_params = mc.generate_cpds()

    data = mc.sample_batch(num_samples)
    fitted_mc = SimpleMarkovChain(length)
    (x1_m, x1_sd), fitted_params = fitted_mc.fit_cpds_to_data(data)

    def print_comparison(true_params, fitted_params):
        print(f"a  | True: {true_vals['coeffs'][0]}, Fitted: {fitted_vals['a']}")
        print(f"b  | True: {true_vals['coeffs'][1]}, Fitted: {fitted_vals['b']}")
        print(f"SD | True: {true_vals['sd']}, Fitted: {fitted_vals['sd']}")

    print("\nParams for X_1")
    print(f"mean | True: {0}, Fitted: {x1_m}")
    print(f"SD   | True: {1}, Fitted: {x1_sd}")

    for i, [true_vals, fitted_vals] in enumerate(zip(true_params, fitted_params), 2):
        print(f"\nParams for X_{i}")
        print_comparison(true_vals, fitted_vals)
    
    print("\n")


if __name__ == "__main__":
    test_markov_chain(length=3, num_samples=10000)
    
