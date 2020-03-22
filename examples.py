from bayes_net import BayesNet
from distributions import DiscreteDistribution, GaussianDistribution
from conditionals import TabularCPD, GaussianCPD
from collections import Counter
import numpy as np
import json

def create_basic_example():
    # Create a simple Bayes net with factors
    # P(X_1), P(X_2|X_1)

    # Theoretical joint prob should be:
    # - (0, 0): 0.25 * 0.47 = 0.1175
    # - (0, 1): 0.25 * 0.53 = 0.1325
    # - (1, 0): 0.75 * 0.18 = 0.135
    # - (1, 1): 0.75 * 0.82 = 0.615

    bn = BayesNet(2)

    bn.set_prior("X_1", DiscreteDistribution({
        0: 0.25,
        1: 0.75
    }))

    bn.set_parents("X_2", ["X_1"], TabularCPD({
        0: DiscreteDistribution({
            0: 0.47,
            1: 0.53
        }),
        1: DiscreteDistribution({
            0: 0.18,
            1: 0.82
        })
    }))

    bn.build()
    return bn

def create_medium_example():
    # Create a Bayes net with the factors:
    # P(A), P(B|A), P(C|A, B), P(D|C)

    bn = BayesNet(["A", "B", "C", "D"])

    bn.set_prior("A", DiscreteDistribution({
        0: 0.3,
        1: 0.7
    }))

    bn.set_parents("B", ["A"], TabularCPD({
        0: DiscreteDistribution({
            0: 0.6,
            1: 0.4
        }),
        1: DiscreteDistribution({
            0: 0.8,
            1: 0.2,
        })
    }))

    bn.set_parents("C", ["A", "B"], TabularCPD({
        (0, 0): DiscreteDistribution({
            0: 0.3,
            1: 0.7
        }),
        (0, 1): DiscreteDistribution({
            0: 0.7,
            1: 0.3
        }),
        (1, 0): DiscreteDistribution({
            0: 0.3,
            1: 0.7
        }),
        (1, 1): DiscreteDistribution({
            0: 0.45,
            1: 0.55
        })
    }))

    bn.set_parents("D", ["C"], TabularCPD({
        # Adding an unnecessary parent, i.e. P(D|C) = P(D)
        0: DiscreteDistribution({
            0: 0.85,
            1: 0.15
        }),
        1: DiscreteDistribution({
            0: 0.85,
            1: 0.15
        })
    }))

    bn.build()
    return bn

def create_gaussian_example():
    # Bayes Net with factors P(A) and P(B|A).
    # P(A) is N(0, 1), and P(B|A) is [N(A, 1)  N(2A, 4)]

    bn = BayesNet(["A", "B"])
    bn.set_prior("A", GaussianDistribution(5, 1))
    bn.set_parents("B", ["A"], GaussianCPD(lambda a: GaussianDistribution((a, 2*a), np.array([[1, 0], [0, 4]]))))

    bn.build()
    return bn


def sampling_example():
    """
    Samples from different Bayes nets to verify that the frequencies are what we expect
    """

    # Sampling without evidence
    print("WITHOUT EVIDENCE\n==========")

    bn1 = create_basic_example()
    bn2 = create_medium_example()
    setups = [bn1, bn2]

    for i, bn in enumerate(setups):
        print(f"\nSetup {i+1}")
        print_sample_freqs(bn, 100000)

    # Sampling with evidence
    print("\nWITH EVIDENCE\n==========")
    print_sample_freqs(bn2, 100000, evidence={"A": 0, "B": 0, "D": 0})

    # Gaussian conditionals
    print("\nGAUSSIAN CONDITIONAL PROBABILITIES\n==========")
    print_sample_mean_cov(create_gaussian_example(), 1000)

def print_sample_freqs(bn, num_samples, evidence=None):
    samples = []
    for _ in range(num_samples):
        samples.append(bn.sample(evidence) if evidence else bn.sample())
    
    freqs = DiscreteDistribution(Counter(samples), needs_normalize=True).probs
    for val, freq in freqs.items():
        print(f"{val}: {freq}")

def print_sample_mean_cov(bn, num_samples):
    total = np.array(bn.sample())
    for _ in range(num_samples - 1):
        sample = np.array(bn.sample())
        total += sample
    print(f"Mean: {total / num_samples}")