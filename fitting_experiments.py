from simple_markov_chain import SimpleMarkovChain
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    test_limited_samples()