import torch
from torch.distributions.normal import Normal

from distributions import GaussianDistribution

def get_coeffs_from_generative(mc, query, evidence):
    true_bias = infer_from_generative(mc, query, evidence, 0).mean
    true_mean = infer_from_generative(mc, query, evidence, 1).mean - true_bias
    true_cov = infer_from_generative(mc, query, evidence, 0).cov

    return true_mean, true_bias, true_cov

def compute_joint_linear(mc, query_node, evidence_node, evidence=1):
    # Only works for scalar linear gaussian
    evidence = {evidence_node: evidence}

    all_coeffs = []
    means = []
    covs = []
    prev_mean = 0
    prev_cov = 1
    all_cov = []
    for i, var in enumerate(mc.ordering):
        node = mc.get_node(var)

        # TODO: clean this up
        if isinstance(node.cpd, GaussianDistribution):
            coeffs = (0, node.cpd.mean)
            cov_node = node.cpd.cov
        elif hasattr(node.cpd.cond_fn, "weights"):
            coeffs = (node.cpd.cond_fn.weights[0].weight, node.cpd.cond_fn.weights[0].bias)
            cov_node = node.cpd.cov
        else:
            coeffs = node.cpd.coeffs_list[0]
            cov_node = node.cpd.cov

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

    # print(f"Exact inference on {query_node}|{evidence_node}={evidence}")
    # print(f"All coeffs: {[all_coeffs[i][0] for i in range(start+1, end)]}")
    # print(f"query_idx: {query_idx}, ev_idx: {evidence_idx}")
    # print(f"means: {means}")
    # print(f"covs: {covs}")
    # print(f"query_mean: {query_mean}")
    # print(f"query_cov: {query_cov}")
    # print(f"front_multiplier: {front_multiplier}")
    # print(f"cross_cov: {cross_cov}")
    # print(f"evidence_mean: {evidence_mean}")
    # print(f"evidence_cov: {evidence_cov}")

    # Conditioning: Conditioning on the evidence to get the query distribution
    # Currently works both ways because it is a scalar
    if query_idx < evidence_idx:
        cond_mean = query_mean + cross_cov*(1./evidence_cov)*(evidence[evidence_node] - evidence_mean)
        cond_cov = query_cov - cross_cov * (1. / evidence_cov) * cross_cov
    else:
        cond_mean = query_mean
        cond_cov = query_cov

    # print(all_coeffs)
    # print(all_cov)
    c = [coeff[0] for coeff in all_coeffs[1:]]

    # print(f"query_cov: {query_cov}, cross_cov: {cross_cov}, evidence_cov: {evidence_cov}")
    # print(f"Numerator: {cross_cov ** 2}, should be: {(c[0] * c[1] * c[2]) ** 2 * all_cov[0] ** 2}")
    # print(f"Denominator: {evidence_cov}, should be: {(c[0] * c[1] * c[2]) ** 2 * all_cov[0] + all_cov[3] + c[2]**2 * all_cov[2] + c[2]**2 * c[1]**2 * all_cov[1]}")
    # print(cond_cov)
    # print(f"cond_mean: {cond_mean}, cond_cov: {cond_cov}")

    return cond_mean, cond_cov

def infer_from_posterior(mc, query_node, evidence_node, evidences):
    return mc.get_node(query_node).cpd.cond_fn([evidences])

def infer_from_generative(mc, query_node, evidence_node, evidences):
    mean, cov = compute_joint_linear(mc, query_node, evidence_node, evidences)
    return GaussianDistribution(mean, cov)

def kl_div_gaussian(true_mean, true_cov, mean, cov):
    return torch.distributions.kl.kl_divergence(Normal(true_mean, true_cov ** 0.5), Normal(mean, cov ** 0.5))

def expected_kl_gaussian(true_mc, inferred_mc, query_node, evidence_node, 
                        true_func=compute_joint_linear, posterior_func=infer_from_posterior, num_samples=5000):

    evidences = true_mc.sample_labeled(num_samples)[evidence_node]

    true_mean, true_cov = true_func(true_mc, query_node, evidence_node, evidences)
    posteriors = posterior_func(inferred_mc, query_node, evidence_node, evidences)

    kl = kl_div_gaussian(true_mean, true_cov, posteriors.mean, posteriors.cov)
    return torch.mean(kl)
