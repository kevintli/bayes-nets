from simple_markov_chain import SimpleMarkovChain
from discriminative_markov_chain import DiscriminativeMarkovChain
import matplotlib.pyplot as plt
from distributions import GaussianDistribution
import numpy as np
import torch

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
        data = mc.sample_batch(num_samples)
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

    data = mc.sample_batch(num_samples)

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


def test_true_linear_discriminative(mc, num_samples=10000):
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

    data = mc.sample_batch(num_samples)

    fitted_mc = DiscriminativeMarkovChain(mc.num_nodes)
    (x1_m, x1_sd), fitted_params = fitted_mc.fit_cpds_to_data(data, log_fn)
    return fitted_params
    # TODO compare with the true posterior distribution

def test_true_linear_variational(mc, num_samples=10000):
    print("Testing VI")

    true_params = mc.parameters

    a_errs, b_errs, sd_errs = {}, {}, {}

    def log_fn(node, epoch, data):
        """
        Track errors for each epoch on every node
        """
        a_true, b_true = true_params[node - 2]['coeffs']
        sd_true = true_params[node - 2]['sd']
        a_err = (data['a'] - a_true) ** 2
        b_err = (data['b'] - b_true) ** 2
        sd_err = (data['cov'] - sd_true) ** 2
        a_errs[node] = a_errs.get(node, []) + [a_err.item()]
        b_errs[node] = b_errs.get(node, []) + [b_err.item()]
        sd_errs[node] = sd_errs.get(node, []) + [sd_err.item()]

    # Sample from p (the true joint distribution)
    data = mc.sample_batch(num_samples)

    # Initialize a discriminative model and fit it using VI
    variational_mc = DiscriminativeMarkovChain(mc.num_nodes)
    variational_mc.generate_cpds()

    # Uncomment to use p instead of p_hat for VI
    # ========
    # models = variational_mc.fit_ELBO(data, mc, plot_name="vi_loss_true")
    # ========
    
    # Uncomment to use p_hat instead of p for VI
    # ========
    fitted_mc = SimpleMarkovChain(mc.num_nodes)
    (x1_m, x1_sd), fp = fitted_mc.fit_cpds_to_data(data, log_fn)

    plt.plot(list(a_errs.values())[0], label="Weight")
    plt.plot(list(b_errs.values())[0], label="Bias")
    plt.plot(list(sd_errs.values())[0], label="SD")
    plt.legend()
    plt.savefig("fitted_mc_errors.png")
    plt.close()

    models = variational_mc.fit_ELBO(data, mc, plot_name="vi_loss_fitted")
    # ========

    return models[0]

    # TODO compare with the true posterior distribution


def compute_joint_linear(mc, num_samples=10000):
    # Only works for scalar linear gaussian
    evidence = {"X_2": 1}
    query_node = "X_1"
    evidence_node = "X_2"

    all_coeffs = []
    means = []
    covs = []
    prev_mean = 0
    prev_cov = 1
    all_cov = []
    for i, var in enumerate(mc.ordering):
        node = mc.get_node(var)
        coeffs = node.cpd.linear_coeffs
        cov_node = node.cpd.sd**2
        all_coeffs.append(coeffs) #Putting all the coefficients together
        all_cov.append(cov_node)

        # Mean computations
        curr_mean = prev_mean * coeffs[0] + coeffs[1]
        means.append(curr_mean)
        prev_mean = curr_mean

        # cov computations
        curr_cov = coeffs[0]*prev_cov*coeffs[0] + cov_node
        covs.append(curr_cov)
        prev_cov = curr_cov

    # Getting the things we need for cross covariances
    query_idx = mc.ordering.index(query_node)
    evidence_idx = mc.ordering.index(evidence_node)

    # getting the front multiplier on the later variable in terms of the earlier one.
    front_multiplier = 1
    start = min(query_idx, evidence_idx)
    end = max(query_idx, evidence_idx) + 1
    for i in range(start + 1, end):
        front_multiplier *= all_coeffs[i][0]

    # Marginalizing out everything else and computing the cross covariances
    query_mean = means[query_idx]
    query_cov = covs[query_idx]

    evidence_mean = means[evidence_idx]
    evidence_cov = covs[evidence_idx]

    cross_cov = front_multiplier * query_cov

    # Conditioning: Conditioning on the evidence to get the query distribution
    # Currently works both ways because it is a scalar
    cond_mean = query_mean + cross_cov*(1./evidence_cov)*(evidence[evidence_node] - evidence_mean)
    cond_cov = query_cov - cross_cov * (1. / evidence_cov) * cross_cov

    # print(all_coeffs)
    # print(all_cov)
    c = [coeff[0] for coeff in all_coeffs[1:]]

    # print(f"query_cov: {query_cov}, cross_cov: {cross_cov}, evidence_cov: {evidence_cov}")
    # print(f"Numerator: {cross_cov ** 2}, should be: {(c[0] * c[1] * c[2]) ** 2 * all_cov[0] ** 2}")
    # print(f"Denominator: {evidence_cov}, should be: {(c[0] * c[1] * c[2]) ** 2 * all_cov[0] + all_cov[3] + c[2]**2 * all_cov[2] + c[2]**2 * c[1]**2 * all_cov[1]}")
    # print(cond_cov)

    return GaussianDistribution(cond_mean, np.sqrt(cond_cov))

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

    # Exact inference to get a baseline
    true_params = compute_joint_linear(mc)

    # Amortized VI to approximate X_1 | X_2
    vi_model = test_true_linear_variational(mc)
    vi_params = vi_model.weights[0].weight.item(), vi_model.weights[0].bias.item(), vi_model.cov_matrix()

    # Discriminative: learn X_1 | X_2 directly with MLE
    disc_params = test_true_linear_discriminative(mc)

    print(f"Exact inference: X1|X2 = N({true_params.mean} * X1 + 0, {true_params.cov ** 2})")
    print(f"Discriminative: X1|X2 = N({disc_params[0]['a']} * X1 + {disc_params[0]['b']}, {disc_params[0]['sd'] ** 2}")
    print(f"VI: X1|X2 = N({vi_params[0]} * X1 + {vi_params[1]}, {vi_params[2] ** 2})")


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
