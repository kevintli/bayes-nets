class CPD:
    """
    Base class for a conditional probability distribution.

    We should be able to query for probabilties and sample from the distribution
        conditioned on a set of evidence variables.
    """
    def get_probability(self, x, evidence):
        """
        Returns P(X=x|e_1, e_2, ..., e_n)

        x          -  any object (e.g. an integer, string, or tuple of values)
        evidence   -  tuple of values for each evidence variable
        """
        raise NotImplementedError

    def sample(self, evidence):
        """
        Returns a value X sampled from the distribution P(X|*evidence)
        """
        raise NotImplementedError


class TabularCPD(CPD):
    """
    Represents a tabular CPD, where we manually specify a probability distribution
        for each possible combination of evidence variables.

    Example:
        t = TabularCPD({
            (0, 0): DiscreteDistribution({0: 0.5, 1: 0.5}),  # Represents P(X | e_1=0, e_2=0)
            (0, 1): DiscreteDistribution({0: 0.2, 1: 0.8}),  # Represents P(X | e_1=0, e_2=1)
            (1, 0): DiscreteDistribution({0: 0.9, 1: 0.1}),  # Represents P(X | e_1=1, e_2=0)
            (1, 1): DiscreteDistribution({0: 0.2, 1: 0.8})   # Represents P(X | e_1=1, e_2=1)
        })

        t.get_probability(0, (0, 1))  # => 0.2
        t.sample((1, 1))              # Samples from P(X | e_1=1, e_2=1)

    Only the evidence variable values need to be enumerated; the resulting conditional for a particular
        set of evidence values can be set to any arbitrary distribution. For example, we can specify P(X|A, B) 
        where A, B take on discrete values listed in the table, but the resulting X|A, B distributions are Gaussian.
    """

    def __init__(self, probabilities):
        """
        probabilities (dict): each key-value pair maps a tuple of numerical quantities for e_1, ..., e_n 
                                to a Distribution object representing P(X | e_1, ..., e_n)

        """
        CPD.__init__(self)
        self._assert_valid_conditionals(probabilities)
        self.conditional_probs = probabilities
        self.num_evidence_vars = self._get_dim(list(probabilities.keys())[0])

    def get_probability(self, x, evidence):
        self._assert_valid_evidence(evidence)
        return self.conditional_probs[evidence].get_probability(x)

    def sample(self, evidence):
        self._assert_valid_evidence(evidence)
        return self.conditional_probs[evidence].sample()

    def _assert_valid_conditionals(self, probs):
        assert probs, "[TabularCPD] Cannot create an empty distribution"
        assert [self._get_dim(key) for key in probs].count(self._get_dim(list(probs.keys())[0])) == len(probs), \
            "[TabularCPD] Must condition on the same number of evidence variables for each case"

    def _get_dim(self, val):
        return 1 if not isinstance(val, tuple) else len(val)

    def _assert_valid_evidence(self, evidence):
        if evidence not in self.conditional_probs:
            raise Exception(f"[TabularCPD] values for evidence variables not found in table: {evidence}")


class GaussianCPD(CPD):
    """
    Represents a Gaussian CPD, where we specify some parameterized function that maps values for evidence variables
        to the mean and covariance of a Gaussian.
    """
    def __init__(self, mean_cov_fn):
        """
        mean_cov_fn: a function that takes in values for evidence variables and returns a GaussianDistribution object

        Example:
        g = GaussianCPD(self, lambda z: GaussianDistribution(z, 1))  # P(X|z) is a unit Gaussian centered at z

        g.get_probability(5.3, 5)  # => 0.38138781546052414
        g.sample(3)                # Samples from P(X | z=3)
        """

        CPD.__init__(self)
        self.mean_cov_fn = mean_cov_fn

    def get_probability(self, x, evidence):
        return self.mean_cov_fn(evidence).get_probability(x)

    def sample(self, evidence, shape=None):
        return self.mean_cov_fn(evidence).sample(shape)
