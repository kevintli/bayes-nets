import numpy as np
import torch

from bayes_net import BayesNet
from conditionals import GaussianCPD
from distributions import GaussianDistribution

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
    
    def generate_cpds(self, a_range=[-10, 10], b_range=[-5, 5], max_sd=2):
        parameters = []

        self.set_prior("X_1", GaussianDistribution(0, 1))

        for i in range(1, self.num_nodes):
            # Randomly assign values for a, b, sd, where:
            # P(X_i+1 | X_i) ~ N(aX_i + b, sd)
            a = np.random.uniform(*a_range)
            b = np.random.uniform(*b_range)
            sd = np.random.uniform(max_sd)

            # Save the generated parameters (for testing/debugging purposes only)
            parameters.append({"a": a, "b": b, "sd": sd})

            # Create a conditional function with the fixed values of a, b, sd generated
            cond_fn = (lambda a, b, sd: (lambda e: GaussianDistribution(a * e[0] + b, sd)))(a, b, sd)

            # Set the actual CPD
            self.set_parents(f"X_{i+1}", [f"X_{i}"], GaussianCPD(cond_fn))

        self.build()
        return parameters

    def fit_cpds_to_data(self, data):
        """
        Params
            data (list[tensor]) - A list containing values for each variable, sampled from the joint distribution
        """
        data = torch.tensor(data)
        parameters = []

        x1_prior = GaussianDistribution.fit_to_data(self._get_data(data, 0))
        self.set_prior("X_1", x1_prior)

        for i in range(1, self.num_nodes):
            print(f"\nFitting X_{i+1}\n==========")
            evidence = [self._get_data(data, i-1)]
            vals = self._get_data(data, i)

            # Fit a linear Gaussian CPD to the data, and save the learned parameters for testing/debugging
            cpd, cond_fn_approx = GaussianCPD.fit_linear_to_data(evidence, vals)
            parameters.append({
                "a": cond_fn_approx.weights[0].weight.squeeze().item(), 
                "b": cond_fn_approx.weights[0].bias.item(), 
                "sd": cond_fn_approx.cov.item()
            })

            # Set the actual CPD according to the fitted values
            self.set_parents(f"X_{i+1}", [f"X_{i}"], cpd)
        
        self.build()
        return (x1_prior.mean, x1_prior.cov), parameters

    def _get_data(self, data, num):
        return data[:,num].unsqueeze(1)

def test_markov_chain(length=5, num_samples=10000):
    mc = SimpleMarkovChain(length)
    true_params = mc.generate_cpds()

    data = mc.sample_batch(num_samples)
    fitted_mc = SimpleMarkovChain(length)
    (x1_m, x1_sd), fitted_params = fitted_mc.fit_cpds_to_data(data)

    def print_comparison(true_params, fitted_params):
        print(f"a  | True: {true_vals['a']}, Fitted: {fitted_vals['a']}")
        print(f"b  | True: {true_vals['b']}, Fitted: {fitted_vals['b']}")
        print(f"SD | True: {true_vals['sd']}, Fitted: {fitted_vals['sd']}")

    print("\nParams for X_1")
    print(f"mean | True: {0}, Fitted: {x1_m}")
    print(f"SD   | True: {1}, Fitted: {x1_sd}")

    for i, [true_vals, fitted_vals] in enumerate(zip(true_params, fitted_params), 2):
        print(f"\nParams for X_{i}")
        print_comparison(true_vals, fitted_vals)
    
    print("\n")


if __name__ == "__main__":
    test_markov_chain(length=5, num_samples=10000)
    
