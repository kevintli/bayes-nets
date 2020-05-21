import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import one_hot

from distributions import GaussianDistribution

def get_coeffs_from_generative(mc, query_node, evidence_node, dim=1):
    """
    Computes the weight, bias, and covariance of the linear Gaussian posterior
     P(query_node | evidence_node) using exact inference.

    Parameters
    ----------
    mc : BayesNet
        The generative model — CPDs must be linear Gaussian!

    query_node : str
        Name of the query node

    evidence_node : str
        Name of the evidence node

    dim : int
        Dimensionality of each node
    """
    zero = 0 if dim == 1 else torch.zeros(dim)

    true_bias = infer_from_generative(mc, query_node, evidence_node, zero).mean
    true_cov = infer_from_generative(mc, query_node, evidence_node, zero).cov

    if dim == 1:
        true_weight = infer_from_generative(mc, query_node, evidence_node, 1).mean - true_bias
    else:
        cols = []
        for i in range(dim):
            x = one_hot(torch.tensor(i), dim)
            col_i = infer_from_generative(mc, query_node, evidence_node, x).mean - true_bias
            cols.append(col_i)
        true_weight = torch.stack(cols).T

    return true_weight, true_bias, true_cov

def infer_from_generative(mc, query_node, evidence_node, evidences):
    """
    Returns a GaussianDistribution object representing the posterior 
     P(query_node|evidence_node = evidences) for fixed evidence values.
    """
    mean, cov = compute_joint_linear(mc, query_node, evidence_node, evidences)
    return GaussianDistribution(mean, cov)

def compute_joint_linear(mc, query_node, evidence_node, evidence=1):
    """
    Returns the mean and covariance of P(query_node | evidence_node=evidence).
    Assumes that `mc` is a BayesNet whose CPDs are *linear Gaussian*.
    """
    dim = isinstance(evidence, torch.Tensor) and evidence.shape and evidence.shape[-1]
    multi = dim > 1

    evidence = {evidence_node: evidence}

    zero = 0 if not multi else torch.zeros(dim)
    identity = 1 if not multi else torch.eye(dim)

    all_coeffs = []
    means = []
    covs = []
    prev_mean = zero
    prev_cov = identity
    all_cov = []
    for i, var in enumerate(mc.ordering):
        node = mc.get_node(var)

        # TODO: clean this up
        if isinstance(node.cpd, GaussianDistribution):
            coeffs = (zero, node.cpd.mean)
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
        if multi:
            curr_mean = prev_mean @ coeffs[0].T + coeffs[1]
        else:
            curr_mean = prev_mean * coeffs[0] + coeffs[1]
        means.append(curr_mean)
        prev_mean = curr_mean

        # cov computations
        if multi:
            curr_cov = coeffs[0] @ prev_cov @ coeffs[0].T + cov_node
        else:
            curr_cov = coeffs[0]*prev_cov*coeffs[0] + cov_node
        covs.append(curr_cov)
        prev_cov = curr_cov

    # Getting the things we need for cross covariances
    query_idx = mc.ordering.index(query_node)
    evidence_idx = mc.ordering.index(evidence_node)

    # getting the front multiplier on the later variable in terms of the earlier one.
    front_multiplier = identity
    start = min(query_idx, evidence_idx)
    end = max(query_idx, evidence_idx) + 1
    for i in range(start + 1, end):
        coeff = all_coeffs[i][0]
        if multi:
            front_multiplier = coeff @ front_multiplier
        else:
            front_multiplier *= coeff

    # Marginalizing out everything else and computing the cross covariances
    query_mean = means[query_idx]
    query_cov = covs[query_idx]

    evidence_mean = means[evidence_idx]
    evidence_cov = covs[evidence_idx]

    if multi:
        cross_cov = query_cov @ front_multiplier.T
    else:
        cross_cov = front_multiplier * query_cov

    # Conditioning: Conditioning on the evidence to get the query distribution
    # Currently works both ways because it is a scalar
    if query_idx < evidence_idx:
        if isinstance(cross_cov, torch.Tensor):
            cond_mean = query_mean + (evidence[evidence_node] - evidence_mean) @ (cross_cov @ torch.inverse(evidence_cov)).T
            cond_cov = query_cov - cross_cov @ torch.inverse(evidence_cov) @ cross_cov.T
        else:
            cond_mean = query_mean + cross_cov*(1./evidence_cov)*(evidence[evidence_node] - evidence_mean)
            cond_cov = query_cov - cross_cov * (1. / evidence_cov) * cross_cov
    else:
        # TODO: this isn't correct, not sure what the actual equations should be though?
        cond_mean = query_mean
        cond_cov = query_cov

    return cond_mean, cond_cov

def infer_from_posterior(mc, query_node, evidence_node, evidences):
    return mc.get_node(query_node).cpd.cond_fn([evidences])

def kl_div_gaussian(true_mean, true_cov, mean, cov):
    if len(cov.shape) and cov.shape[-1] > 1:
        return torch.distributions.kl.kl_divergence(MultivariateNormal(true_mean, true_cov), MultivariateNormal(mean, cov))
    else:
        return torch.distributions.kl.kl_divergence(Normal(true_mean, true_cov ** 0.5), Normal(mean, cov ** 0.5))

def expected_kl_gaussian(true_mc, inferred_mc, query_node, evidence_node, 
                        true_func=compute_joint_linear, posterior_func=infer_from_posterior, num_samples=5000):
    """
    Given the true Gaussian posterior p(X|e) represented by `true_mc`, and the inferred posterior q(X|e) represented by `inferred_mc`,
     returns the expected KL divergence between the two:

     E_e~p(e)[KL(p(X|e) || q(X|e))]

    The `true_func` and `posterior_func` are functions that take in a BayesNet, query node, evidence node, and evidence,
     then return the true and learned posterior distributions respectively. These can be changed depending on the model;
     for example, if your learned posterior is a generative model instead of a discriminative one, you should set posterior_func=compute_joint_linear.
    """
    evidences = true_mc.sample_labeled(num_samples)[evidence_node]

    true_mean, true_cov = true_func(true_mc, query_node, evidence_node, evidences)
    posteriors = posterior_func(inferred_mc, query_node, evidence_node, evidences)

    kl = kl_div_gaussian(true_mean, true_cov, posteriors.mean, posteriors.cov)
    return torch.mean(kl)
