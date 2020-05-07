from bayes_net import BayesNet
from conditionals import GaussianCPD, LinearGaussianCPD
from distributions import GaussianDistribution

class DiscriminativeMarkovChain(BayesNet):
    """
    Represents a simple Markov Chain
    X_n -> X_1, ..., X_{n-1}
    """

    def __init__(self, num_nodes):
        super(DiscriminativeMarkovChain, self).__init__(num_nodes)

    def initialize_empty_cpds(self, cpd_class=LinearGaussianCPD):
        self.set_prior(f"X_{self.num_nodes}", GaussianDistribution(0, 1))
        for i in range(self.num_nodes - 1):
            self.set_parents(f"X_{i+1}", [f"X_{self.num_nodes}"], cpd_class.empty([1], 1))
        self.build()

if __name__ == "__main__":
    test_markov_chain(length=3, num_samples=10000)

