from bayes_net import BayesNet
from conditionals import GaussianCPD
from distributions import GaussianDistribution
import numpy as np

def test_linear_fit():
    # P(A), P(B|A)
    bn = BayesNet(["A", "B"])
    bn.set_prior("A", GaussianDistribution(np.array([4, 2]), np.eye(2)))
    bn.set_parents("B", ["A"], GaussianCPD(lambda a: GaussianDistribution(2 * a[0], np.eye(2))))
    bn.build()

    samples = bn.sample_batch(1000) # 1000 samples of the form (A, B)
    evidence = (np.array([s[0] for s in samples]),)
    data = np.array([s[1] for s in samples])
    b_hat = GaussianCPD.fit_linear_to_data(evidence, data)

    return bn.get_node("B").cpd, b_hat

b, b_hat = test_linear_fit()