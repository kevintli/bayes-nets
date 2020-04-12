from bayes_net import BayesNet
from conditionals import GaussianCPD
from distributions import GaussianDistribution
import numpy as np
import torch

def test_linear_fit():
    # A -> B

    bn = BayesNet(["A", "B"])
    bn.set_prior("A", GaussianDistribution(0, 1))

    # Univariate Gaussian example
    # bn.set_parents("B", ["A"], GaussianCPD(lambda a: GaussianDistribution(7.5 * a[0], 1.6)))

    # Multivariate Gaussian example
    bn.set_parents("B", ["A"], 
        GaussianCPD(lambda a: GaussianDistribution(
                                torch.tensor([7.65 * a[0], 3.5 * a[0]]), 
                                torch.tensor([[5, 3.1], [3.1, 7]])
                              )
        )
    )
    bn.build()

    # List of samples; each sample is a list of tensors (one for each variable in the bayes net)
    samples = bn.sample_batch(10000)

    # The list of evidence variables (length 1 since there's only 1 conditioning variable). 
    # Put together an Nx1 tensor of all the A values
    evidence = [torch.cat([s[0] for s in samples])]

    # Put together an Nx1 tensor of all the B values
    data = torch.cat([s[1] for s in samples])

    b_hat, _ = GaussianCPD.fit_linear_to_data(evidence, data)

    return bn.get_node("B").cpd, b_hat

b, b_hat = test_linear_fit()