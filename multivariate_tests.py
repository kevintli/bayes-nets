import torch

from conditionals import LinearGaussianCPD
from fitting import fit_MLE
from simple_markov_chain import SimpleMarkovChain

def test_polynomial_gaussian():
    a2 = torch.eye(2)
    a1 = torch.tensor([[1, 0], [2, -3]]).float()
    a0 = torch.tensor([1, 1]).float()

    b2 = torch.diag(torch.tensor([3, 2])).float()
    b1 = torch.tensor([[0.5, 4], [-0.8, 6]]).float()
    b0 = torch.tensor([-3, 5]).float()

    cov = torch.diag(torch.tensor([2, 3]).float())

    qg = LinearGaussianCPD([(a2, a1, a0), (b2, b1, b0)], cov)
    lg = LinearGaussianCPD([(a1, a0), (b1, b0)], cov)

    evidence = [torch.tensor([[1, 2], [0, 0]]).float(), torch.tensor([[0, 1], [0, 0]]).float()]
    print("QG:", qg.cond_fn(evidence)) # Should have means [[3.2, 12], [-2, 6]]
    print("LG:", lg.cond_fn(evidence)) # Should have means [[2.2, 6], [-2, 6]]

    lg_scalar = LinearGaussianCPD([(2, 0), (-1, 3)], 0.5)
    print("LG scalar:", lg_scalar.cond_fn([1, 2])) # Should have mean 3

if __name__ == "__main__":
    test_polynomial_gaussian()