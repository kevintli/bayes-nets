import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from bayes_net import BayesNet
from conditionals import GaussianCPD, LinearGaussianCPD
from distributions import GaussianDistribution
from fitting import LinearGaussianConditionalFn

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        Params
        - evidence_vars:  A list of evidence variable data, where the ith item is a shape (N, E_i) batch of evidence
                           data for the ith evidence variable
        - data:           A shape (N, D) batch of data sampled from the conditional distribution
        """
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

class DiscriminativeMarkovChain(BayesNet):
    """
    Represents a simple Markov Chain
    X_n -> X_1, ..., X_{n-1}
    """

    def __init__(self, num_nodes):
        super(DiscriminativeMarkovChain, self).__init__(num_nodes)

    def initialize_empty_cpds(self):
        self.set_prior(f"X_{self.num_nodes}", GaussianDistribution(0, 1))
        for i in range(self.num_nodes - 1):
            self.set_parents(f"X_{i+1}", [f"X_{self.num_nodes}"], LinearGaussianCPD.empty([1], 1))
        self.build()

    def fit_cpds_to_data(self, data, log_fn=None):
        """
        Params
            data (list[tensor]) - A list containing values for each variable, sampled from the joint distribution
            log_fn (function)   - A function that takes three arguments: the node number, epoch number, and epoch data
        """
        parameters = []

        end_idx = self.num_nodes
        xend_prior = GaussianDistribution.fit_to_data(data[f"X_{end_idx}"])
        self.set_prior(f"X_{end_idx}", xend_prior)

        for i in range(1, self.num_nodes):
            print(f"\nFitting X_{i}\n==========")
            evidence = [data[f"X_{end_idx}"]]
            vals = data[f"X_{i}"]

            # Fit a linear Gaussian CPD to the data, and save the learned parameters for testing/debugging
            new_log_fn = None if not log_fn else (lambda num: (lambda *args: log_fn(num, *args)))(i)
            cpd, cond_fn_approx = GaussianCPD.fit_linear_to_data(evidence, vals, new_log_fn)
            parameters.append({
                "a": cond_fn_approx.weights[0].weight.squeeze().item(),
                "b": cond_fn_approx.weights[0].bias.item(),
                "sd": cond_fn_approx.cov.item()
            })

            # Set the actual CPD according to the fitted values
            self.set_parents(f"X_{i}", [f"X_{end_idx + 1}"], cpd)

        self.build()
        return (xend_prior.mean, xend_prior.cov), parameters

if __name__ == "__main__":
    test_markov_chain(length=3, num_samples=10000)

