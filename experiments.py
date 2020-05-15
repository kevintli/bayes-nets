import torch
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayes_net import BayesNet
from conditionals import GaussianCPD, LinearGaussianCPD
from discriminative_markov_chain import DiscriminativeMarkovChain
from distributions import GaussianDistribution
from fitting import fit_MLE, fit_VI, reverse_KL_linear, variational_loss
from inference import *
from simple_markov_chain import SimpleMarkovChain

def get_inference_results(mc, num_samples=10000):
    data = mc.sample_labeled(num_samples)

    print("Data shapes:", [d.shape for d in data.values()])

    # Generative
    fitted_mc = SimpleMarkovChain(mc.num_nodes)
    fitted_mc.initialize_empty_cpds()
    log_fn, plot_results = make_linear_log_fn(mc)
    fit_MLE(data, fitted_mc, log_fn=log_fn)
    plot_results(f"{num_samples}-linear-fit")

    # Jumpy generative (every other node)
    jumpy = SimpleMarkovChain(mc.num_nodes, step=2)
    jumpy.initialize_empty_cpds()
    fit_MLE(data, jumpy, log_fn=log_fn)

    full_jump = SimpleMarkovChain(mc.num_nodes, step=mc.num_nodes-1)
    full_jump.initialize_empty_cpds()
    fit_MLE(data, full_jump, log_fn=log_fn)

    # Discriminative
    p_hat = DiscriminativeMarkovChain(mc.num_nodes)
    p_hat.initialize_empty_cpds()
    fit_MLE(data, p_hat, batch_size=min(32, num_samples // 5))

    # VI: using reverse KL from true posterior and the ELBO as two different losses
    q_ideal = DiscriminativeMarkovChain(mc.num_nodes)
    q_ideal.initialize_empty_cpds()
    fit_VI(data, mc, q_ideal, num_epochs=400, loss_fn=reverse_KL_linear, plot_name="reverse_kl_loss")

    q = DiscriminativeMarkovChain(mc.num_nodes)
    q.initialize_empty_cpds()
    fit_VI(data, mc, q, ideal_variational_mc=q_ideal, batch_size=min(32, num_samples // 5))

    # VI loss on the entire dataset of evidence
    ideal_full_loss = variational_loss(mc, q_ideal, f"X_{mc.num_nodes}", data[f"X_{mc.num_nodes}"])
    q_full_loss = variational_loss(mc, q, f"X_{mc.num_nodes}", data[f"X_{mc.num_nodes}"])
    print(f"q loss: {q_full_loss}, ideal loss: {ideal_full_loss}")

    return fitted_mc, jumpy, full_jump, p_hat, q

def make_linear_log_fn(true_mc):
    a_errs, b_errs, cov_errs = {}, {}, {}

    def log_fn(node_name, cond_fn):
        a = cond_fn.weights[0].weight.squeeze()
        b = cond_fn.weights[0].bias.squeeze()
        cov = cond_fn.cov_matrix().item()

        true_a, true_b = true_mc.parameters[node_name]["coeffs"]
        true_cov = true_mc.parameters[node_name]["cov"]

        a_errs[node_name] = a_errs.get(node_name, []) + [abs(a - true_a)]
        b_errs[node_name] = b_errs.get(node_name, []) + [abs(b - true_b)]
        cov_errs[node_name] = cov_errs.get(node_name, []) + [abs(cov - true_cov)]

    def plot_results(prefix="linear-fit"):
        for node_name in a_errs:
            plt.figure()
            plt.title(f"Errors in fitting {node_name}")
            plt.plot(a_errs[node_name], label="Weight")
            plt.plot(b_errs[node_name], label="Bias")
            plt.plot(cov_errs[node_name], label="Cov")
            plt.legend()
            plt.savefig(f"data/{prefix}-{node_name}")
    
    return log_fn, plot_results
        
def test_finite_samples(sample_amounts=[1000]):
    # Arbitrary length 5 chain
    # length = 5
    # true_mc = SimpleMarkovChain(5)
    # true_mc.specify_polynomial_cpds((0, 1), [(1.2, 0.3), (0.4, 2), (2.3, -1.2), (1.5, 0.6)], [0.5, 1.5, 0.8, 2.3])

    # Easy length 5 chain
    # length = 5
    # true_mc = SimpleMarkovChain(length)
    # true_mc.specify_polynomial_cpds((0, 1), [(1, 0), (1, 0), (1, 0), (1, 0)], [1, 1, 1, 1])

    # Arbitrary length 2 chain
    # length = 2
    # true_mc = SimpleMarkovChain(length)
    # true_mc.specify_polynomial_cpds((0, 1), [(-1.5, 0.6)], [0.7])

    # Length 3 chain
    length = 3
    true_mc = SimpleMarkovChain(length)
    true_mc.specify_polynomial_cpds((0, 1), [(1, 0), (1, 0)], [1, 1])

    # Show distribution of evidence
    # evidence = true_mc.sample_labeled(1000)[f"X_{length}"]
    # plt.hist(evidence.squeeze(-1), bins=50)
    # plt.show()

    true_weight, true_bias, true_cov = get_coeffs_from_generative(true_mc, "X_1", f"X_{length}")
    print(f"True posterior: X_1|X_{length} ~ N({true_weight} * X_{length} + {true_bias}, {true_cov})")

    gen_results = []
    jumpy_results = []
    full_jump_results = []
    disc_results = []
    vi_results = []

    for num_samples in sample_amounts:
        print(f"Trying sample size: {num_samples}")

        # Get p_hat using discriminative, q using VI
        p_hat, jumpy, full_jump, disc_posterior, vi_posterior = get_inference_results(true_mc, num_samples)

        gen_kl = expected_kl_gaussian(true_mc, p_hat, "X_1", f"X_{length}", posterior_func=infer_from_generative)
        # jumpy_kl = expected_kl_gaussian(true_mc, jumpy, "X_1", f"X_{length}", posterior_func=infer_from_generative)
        # full_jump_kl = expected_kl_gaussian(true_mc, full_jump, "X_1", f"X_{length}", posterior_func=infer_from_generative)
        disc_kl = expected_kl_gaussian(true_mc, disc_posterior, "X_1", f"X_{length}")
        vi_kl = expected_kl_gaussian(true_mc, vi_posterior, "X_1", f"X_{length}")

        gen_results.append(gen_kl)
        # # jumpy_results.append(jumpy_kl)
        # # full_jump_results.append(full_jump_kl)
        disc_results.append(disc_kl)
        vi_results.append(vi_kl)

        # ==========================================
        # Printing detailed results for each node
        # ==========================================

        for i in range(1, length):
            true_weight, true_bias, true_cov = get_coeffs_from_generative(true_mc, f"X_{i}", f"X_{length}")
            print("\n" + "==========" * 10)
            print(f"True posterior: X_{i}|X_{length} ~ N({true_weight} * X_{length} + {true_bias}, {true_cov})")
            print("==========" * 10)

            gen_weight, gen_bias, gen_cov = get_coeffs_from_generative(p_hat, f"X_{i}", f"X_{length}")
            # # jumpy_weight, jumpy_bias, jumpy_cov = get_coeffs_from_generative(jumpy, "X_1", f"X_{length}")
            # # full_jump_weight, full_jump_bias, full_jump_cov = get_coeffs_from_generative(full_jump, "X_1", f"X_{length}")
            dif = disc_posterior.get_node(f"X_{i}").cpd.cond_fn
            vif = vi_posterior.get_node(f"X_{i}").cpd.cond_fn

            print(f'Gen posterior: N({gen_weight} * X_{length} + {gen_bias}, {gen_cov})')
            # # print(f'Jumpy posterior: N({jumpy_weight} * X_{length} + {jumpy_bias}, {jumpy_cov})')
            # # print(f'Full jump posterior: N({full_jump_weight} * X_{length} + {full_jump_bias}, {full_jump_cov})')
            print(f"Disc posterior: N({dif.weights[0].weight.item()} * X_{length} + {dif.weights[0].bias.item()}, {dif.cov_matrix().item()})")
            print(f"VI posterior: N({vif.weights[0].weight.item()} * X_{length} + {vif.weights[0].bias.item()}, {vif.cov_matrix().item()})")
                

    # df = pd.DataFrame(np.c_[gen_results, jumpy_results, full_jump_results, disc_results, vi_results], columns=["Generative", "Jumpy (step=2)", "Full jump", "Discriminative", "VI"], index=sample_amounts)
    df = pd.DataFrame(np.c_[gen_results, disc_results, vi_results], columns=["Generative", "Discriminative", "VI"], index=sample_amounts)
    df.plot.bar(rot=0)
    plt.title(f"Inference of P(X1 | X{length})")
    plt.xlabel("Number of samples")
    plt.ylabel("Expected KL Divergence")
    plt.show()

def test_model_mismatch():
    mc = SimpleMarkovChain(5)
    mc.specify_polynomial_cpds((0, 1), [(0.7, 1.2, 0.3), (0.6, 0.4, 2), (1.3, 2.3, -1.2), (1.8, 1.5, 0.6)], [0.5, 1.5, 0.8, 2.3])

    disc_posterior, vi_posterior = get_inference_results(mc)

    data = mc.sample_labeled(10000)
    nn_posterior = DiscriminativeMarkovChain(mc.num_nodes)
    nn_posterior.initialize_empty_cpds(GaussianCPD)
    fit_MLE(data, nn_posterior, batch_size=64)

    def get_true_posterior(_, query_node, __, evidences):
        true_posterior = nn_posterior.get_node(query_node).cpd.cond_fn([evidences])
        return true_posterior.mean, true_posterior.cov

    disc_kl = expected_kl_gaussian(mc, disc_posterior, "X_1", "X_5", true_func=get_true_posterior)
    vi_kl = expected_kl_gaussian(mc, vi_posterior, "X_1", "X_5", true_func=get_true_posterior)

    print(f"Disc KL: {disc_kl}")
    print(f"VI KL: {vi_kl}")

if __name__ == "__main__":
    test_finite_samples()
    # test_model_mismatch()



    