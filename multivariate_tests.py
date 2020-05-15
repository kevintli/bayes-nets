import torch

from conditionals import LinearGaussianCPD
from fitting import fit_MLE
from simple_markov_chain import SimpleMarkovChain
from inference import compute_joint_linear, get_coeffs_from_generative

a2 = torch.eye(2)
a1 = torch.tensor([[1, 2], [0, -3]]).float()
a0 = torch.tensor([1, 1]).float()

b2 = torch.diag(torch.tensor([3, 2])).float()
b1 = torch.tensor([[0.5, -0.8], [4, 6]]).float()
b0 = torch.tensor([-3, 5]).float()

cov1 = torch.diag(torch.tensor([2, 3]).float())
cov2 = torch.tensor([[3, 0], [0, 7]]).float()

def test_exact_inference():
    # Scalar case (should be 0.33, 0, 0.67)
    mc = SimpleMarkovChain(3)
    mc.specify_polynomial_cpds((0, 1), [(1, 0), (1, 0)], [1, 1])
    weight, bias, cov = get_coeffs_from_generative(mc, "X_1", "X_3")
    print(weight, bias, cov)

    # Multivariate case
    mc = SimpleMarkovChain(2)
    mc.specify_polynomial_cpds(
        (torch.tensor([0, 2]).float(), torch.diag(torch.tensor([1, 0.4]))), 
        [(torch.diag(torch.tensor([2., -1.])), torch.tensor([-3., 3.]))], 
        [torch.diag(torch.tensor([0.5, 1.2]))]
    )
    weight, bias, cov = get_coeffs_from_generative(mc, "X_1", "X_2", 2)
    print("Weight:", weight)
    print("Bias:", bias)
    print("Cov:", cov)

def test_fitting():
    mc = SimpleMarkovChain(3)
    mc.specify_polynomial_cpds((torch.zeros(2), torch.eye(2)), [(a1, a0), (b1, b0)], [cov1, cov2])
    data = mc.sample_labeled(1000)
    print(data)

    fitted_mc = SimpleMarkovChain(3)
    fitted_mc.fit_cpds_to_data(data, dim=2)
    
    for i in range(2, fitted_mc.num_nodes+1):
        fn = fitted_mc.get_node(f"X_{i}").cpd.cond_fn
        print("Weight:", fn.weights[0].weight)
        print("Bias:", fn.weights[0].bias)
        print("Cov:", fn.cov_matrix())

    fitted_data = fitted_mc.sample_labeled(1000)
    print(fitted_data)


def test_polynomial_gaussian():
    qg = LinearGaussianCPD([(a2, a1, a0), (b2, b1, b0)], cov1)
    lg = LinearGaussianCPD([(a1, a0), (b1, b0)], cov1)

    evidence = [torch.tensor([[1, 2], [0, 0]]).float(), torch.tensor([[0, 1], [0, 0]]).float()]
    print("QG:", qg.cond_fn(evidence)) # Should have means [[3.2, 12], [-2, 6]]
    print("LG:", lg.cond_fn(evidence)) # Should have means [[2.2, 6], [-2, 6]]

    lg_scalar = LinearGaussianCPD([(2, 0), (-1, 3)], 0.5)
    print("LG scalar:", lg_scalar.cond_fn([1, 2])) # Should have mean 3

if __name__ == "__main__":
    # test_polynomial_gaussian()
    test_exact_inference()
    # test_fitting()