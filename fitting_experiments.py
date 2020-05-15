import matplotlib.pyplot as plt
import numpy as np
import torch

from discriminative_markov_chain import DiscriminativeMarkovChain
from distributions import GaussianDistribution
from fitting import fit_VI, fit_MLE
from prob_utils import compute_joint_linear
from simple_markov_chain import SimpleMarkovChain

def print_comparison(true_params, fitted_params):
    print(f"a  | True: {true_params['coeffs'][0]}, Fitted: {fitted_params['a']}")
    print(f"b  | True: {true_params['coeffs'][1]}, Fitted: {fitted_params['b']}")
    print(f"SD | True: {true_params['sd']}, Fitted: {fitted_params['sd']}")

def test_limited_samples(length=2, sample_amounts=[100, 1000, 5000, 8000, 10000, 50000]):
    mc = SimpleMarkovChain(length)
    true_params = mc.generate_cpds()

    results = {amt: [] for amt in sample_amounts}
    def log_fn(node, epoch, data):
        a_true, b_true = true_params[node-2]['coeffs']
        sd_true = true_params[node-2]['sd']
        a_err = (data['a'] - a_true) ** 2
        b_err = (data['b'] - b_true) ** 2
        sd_err = (data['cov'] - sd_true) ** 2
        results[num_samples] += [(a_err.item(), b_err.item(), sd_err.item())]

    for num_samples in sample_amounts:
        data = mc.sample(num_samples)
        fitted_mc = SimpleMarkovChain(length)
        (x1_m, x1_sd), fitted_params = fitted_mc.fit_cpds_to_data(data, log_fn)

    for i, name in enumerate(['Weight', 'Bias', 'SD']):
        plt.figure()
        plt.title(f"{name} errors")
        plt.xlabel("Epochs")
        plt.ylabel("Squared error")

        for num_samples, errors in results.items():
            plt.plot([error[i] for error in errors], label=num_samples)

        plt.legend(title="Number of samples")
        plt.show()


def test_true_linear(length=4, num_samples=10000):
    mc = SimpleMarkovChain(length)
    true_params = mc.generate_cpds()

    node_errs = {}

    def log_fn(node, epoch, data):
        """
        Track errors for each epoch on every node
        """
        a_true, b_true = true_params[node-2]['coeffs']
        sd_true = true_params[node-2]['sd']
        a_err = (data['a'] - a_true) ** 2
        b_err = (data['b'] - b_true) ** 2
        sd_err = (data['cov'] - sd_true) ** 2
        node_errs[node] = node_errs.get(node, []) + [(a_err.item(), b_err.item(), sd_err.item())]

    data = mc.sample(num_samples)

    fitted_mc = SimpleMarkovChain(length)
    (x1_m, x1_sd), fitted_params = fitted_mc.fit_cpds_to_data(data, log_fn)


    print("\nParams for X_1")
    print(f"mean | True: {0}, Fitted: {x1_m}")
    print(f"SD   | True: {1}, Fitted: {x1_sd}")

    for i, [true_vals, fitted_vals] in enumerate(zip(true_params, fitted_params), 2):
        print(f"\nParams for X_{i}")
        print_comparison(true_vals, fitted_vals)
    
    print("\n")

    for node, errors in node_errs.items():
        plt.figure()
        for i, name in enumerate(['Weight', 'Bias', 'SD']):
            plt.plot([error[i] for error in errors], label=name)
        plt.legend()
        plt.title(f"P(X_{node} | X_{int(node) - 1}) parameters")
        plt.xlabel("Epoch")
        plt.ylabel("Squared error")
        plt.show()


def test_true_linear_discriminative(data):
    node_errs = {}
    true_params = mc.parameters

    def log_fn(node, epoch, data):
        """
        Track errors for each epoch on every node
        """
        a_true, b_true = true_params[node - 2]['coeffs']
        sd_true = true_params[node - 2]['sd']
        a_err = (data['a'] - a_true) ** 2
        b_err = (data['b'] - b_true) ** 2
        sd_err = (data['cov'] - sd_true) ** 2
        node_errs[node] = node_errs.get(node, []) + [(a_err.item(), b_err.item(), sd_err.item())]

    p_hat = DiscriminativeMarkovChain(mc.num_nodes)
    p_hat.initialize_empty_cpds()
    fit_MLE(data, p_hat)

    return p_hat

def test_true_linear_variational(data):
    print("Testing VI")

    true_params = mc.parameters

    a_errs, b_errs, sd_errs = {}, {}, {}

    fitted_mc = SimpleMarkovChain(mc.num_nodes)
    fitted_mc.fit_cpds_to_data(data)

    # plt.plot(list(a_errs.values())[0], label="Weight")
    # plt.plot(list(b_errs.values())[0], label="Bias")
    # plt.plot(list(sd_errs.values())[0], label="SD")
    # plt.legend()
    # plt.savefig("fitted_mc_errors.png")
    # plt.close()

    q = DiscriminativeMarkovChain(mc.num_nodes)
    q.initialize_empty_cpds()
    fit_VI(data, fitted_mc, q)
    # ========

    return q


if __name__ == "__main__":
    print("Fitting experiments...")

    # Basic X_1 -> X_2
    # Goal is to infer P(X_2 | X_1)
    mc = SimpleMarkovChain(2)

    ## More extreme example: should learn to do X1 = X2 - 1 for values near X2=2
    ## X_1 ~ N(1, 0.0001)
    ## X_2 | X_1 ~ N(X_1 + 1, 0.0001)
    # mc.specify_polynomial_cpds((1, 0.0001), [(1, 1)], [0.0001])

    mc.specify_polynomial_cpds((0, 1), [(1, 0)], [1])

    data = mc.sample_labeled(10000)

    # Exact inference to get a baseline
    true_mean, true_cov = compute_joint_linear(mc, "X_1", "X_2")

    # Amortized VI to approximate X_1 | X_2
    vi_model = test_true_linear_variational(data)
    x_1 = vi_model.get_node("X_1").cpd.cond_fn
    vi_params = x_1.weights[0].weight.item(), x_1.weights[0].bias.item(), x_1.cov_matrix()

    # # Discriminative: learn X_1 | X_2 directly with MLE
    # mle_model = test_true_linear_discriminative(data)
    # x_1 = mle_model.get_node("X_1").cpd.cond_fn
    # disc_params = x_1.weights[0].weight.item(), x_1.weights[0].bias.item(), x_1.cov_matrix()

    print(f"Exact inference: X1|X2 = N({true_mean} * X1 + 0, {true_cov})")
    # print(f"Discriminative: X1|X2 = N({disc_params[0]} * X1 + {disc_params[1]}, {disc_params[2]})")
    print(f"VI: X1|X2 = N({vi_params[0]} * X1 + {vi_params[1]}, {vi_params[2]})")


    # ==========================
    # TESTING VI RECONSTRUCTION
    # ==========================

    # # Try a range of evidence values from -1 to 1
    # x_2_evidence = torch.tensor(np.linspace(-1, 1, 11))[:,None]

    # # Run values through the encoder, q(X_1 | X_2)
    # x_1_vi = vi_model([x_2_evidence])

    # # Run through the decoder, p(X_2 | X_1)
    # x_2_decoded = mc.get_node("X_2").cpd.cond_fn([x_1_vi.mean])

    # print("Original values:", x_2_evidence)
    # print(f"Encoder values: {x_1_vi.mean} with SD: {x_1_vi.cov}")
    # print(f"Decoder values: {x_2_decoded.mean} with SD: {x_2_decoded.cov}")
