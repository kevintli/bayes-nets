from simple_markov_chain import SimpleMarkovChain
import matplotlib.pyplot as plt
from distributions import GaussianDistribution

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

def compute_joint_linear(mc):
    # Only works for scalar linear gaussian
    evidence = {"X_1": 1}
    query_node = "X_5"
    evidence_node = "X_1"

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

    return GaussianDistribution(cond_mean, np.sqrt(cond_cov))

if __name__ == "__main__":
    # test_true_linear()
    mc = SimpleMarkovChain(5)
    true_params = mc.generate_cpds()
    compute_joint_linear(mc)