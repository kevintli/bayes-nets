def expected_kl_divergence(p, q):
    """
    Returns the approximate expected KL divergence between two conditional probability distributions, i.e.
    E_z[ D_kl(p(x|z) || q(x|z)) ], using sampling

    Params
        - p (CPD): represents p(x|z)
        - q (CPD): represents q(x|z)
    """
    