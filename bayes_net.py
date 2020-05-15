from dag import DirectedAcyclicGraph
import numpy as np
import torch

class BNNode:
    """
    Simple data class for a node in a Bayes net, which stores its own CPD and a list of all parents.
    """
    def __init__(self, name, cpd=None, parents=None):
        self.name = name
        self.cpd = cpd
        self.parents = parents or []

class BayesNet(DirectedAcyclicGraph):
    """
    A Bayes Net is a DAG of BNNode objects, each of which has a corresponding CPD according to the 
        dependencies encoded by the edges.

    Provides methods for sampling and inference.

    Example: Setting up a Bayes net with factors P(A), P(B|A), B(C|A, B)

    A -> B -> C
    |         ^
    |         |  
     ---------

    bn = BayesNet(["A", "B", "C"])
    bn.set_prior("A", GaussianDistribution(...))
    bn.set_parents("B", ["A"], TabularDistribution(...))
    bn.set_parents("C", ["A", "B"], TabularDistribution(...))
    bn.build()

    bn.sample({"A": 5}) => (5, 2, 3)
    bn.sample()         => (2, 1, 4)
    ...
    """

    def __init__(self, nodes):
        if isinstance(nodes, int):
            nodes = [f"X_{i+1}" for i in range(nodes)] # Default node names are X_1, X_2, ..., X_n
        DirectedAcyclicGraph.__init__(self, nodes)

        # Maps node names to BNNode objects, which have conditional probabilities and other useful data
        self._bnnode_mapping = {name: BNNode(name) for name in nodes}
        self.num_nodes = len(nodes)

        # Before the BayesNet is built (using build()), we cannot run sampling/inference/parameter estimation yet.
        self.build_done = False

    def set_parents(self, node_name, parent_names, cpd):
        """
        Use this to specify a dependency of a node on a set of parent nodes.
        This will add incoming edges from each node in <parent_names> to the node <node_name>, and
            save the given CPD to the node.

        When creating the CPD, the order of evidence variables should exactly match the ordering of nodes
            in <parent_names>.
        
        Params
        - node_name (str):     Name of a node in the Bayes net whose parents we want to set
        - parent_names (list): List of names of desired parent nodes
        - cpd (CPD):           A CPD object representing the distribution of the node conditioned on its parents
        """
        for parent in parent_names:
            self.add_edge(parent, node_name)
        self._bnnode_mapping[node_name].cpd = cpd
        self._bnnode_mapping[node_name].parents = parent_names

    def set_prior(self, node_name, cpd):
        """
        Sets the probability distribution of a node directly without conditioning on any other nodes.
        """
        self.set_parents(node_name, [], cpd)

    def get_node(self, name):
        return self._bnnode_mapping[name]

    def all_nodes(self):
        return [self.get_node(name) for name in self.ordering]

    def build(self):
        """
        Finalizes the BayesNet by checking that everything is properly specified, then assigning
        a fixed topological ordering to the nodes.

        Call this ONCE after creating all desired connections between nodes.

        After the BayesNet is built, you can run sampling, parameter estimation, and inference on it.
        """
        if self.build_done:
            print("[BayesNet] Warning: Attempted to build() an already-finalized Bayes net")
            return

        # TODO: validate by checking that there are no cycles

        self.ordering = self.topological_order()
        self.build_done = True

    def sample(self, num_samples=1, evidence_dict=None):
        """
        Samples values from the entire joint distribution of the Bayes net using ancestral sampling.

        Optionally, a dictionary of evidence variables and their values can be specified,
            and this would sample conditioned on the known evidence variables.

        Params
        - num_samples (int): Number of samples to take
        - evidence_dict (dict): Map of node names (representing evidence variables) to their known values

        Returns: a list of sampled values according to the topological ordering of the graph
        """

        self._assert_build_done()

        evidence_dict = dict(evidence_dict or {}) # Make a copy
        results = []
        shape = [] if num_samples == 1 else [num_samples]
        for node_name in self.ordering:
            if node_name in evidence_dict:
                results.append(evidence_dict[node_name])
            else:
                node = self.get_node(node_name)
                evidence = [evidence_dict[parent_name] for parent_name in node.parents]
                sample = node.cpd.sample(evidence) if evidence else node.cpd.sample(shape=shape)
                results.append(sample)
                evidence_dict[node_name] = sample
        return results

    def sample_labeled(self, num_samples=1, evidence_dict=None):
        sample = self.sample(num_samples, evidence_dict)
        return {name: sample[i] for i, name in enumerate(self.ordering)}

    def _log_prob_for_node(self, node_name, data):
        node = self.get_node(node_name)
        evidence = [data[parent] for parent in node.parents]
        if evidence:
            return node.cpd.get_log_prob(data[node_name], evidence)
        else:
            return node.cpd.get_log_prob(data[node_name])

    def get_log_prob(self, data, exclude=[]):
        """
        Returns the log joint probability of the the given data.

        Parameters
        ----------
        data : dict[str, tensor]
            A named dataset where the keys are the node names, and the values are
            a list of sampled values for that node

        exclude : list[str]
            A list of names of nodes to exclude from the log prob calculation.
        """
        return sum([self._log_prob_for_node(node_name, data) for node_name in self.ordering if not node_name in exclude])
    
    def _assert_build_done(self):
        assert self.build_done, "[BayesNet] Must call build() before running sampling, parameter estimation, or inference!"
