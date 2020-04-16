import numpy as np
import torch
import torch.optim as optim
from bayes_net import BayesNet
from conditionals import GaussianCPD
from distributions import GaussianDistribution
from fitting import LinearGaussianConditionalFn

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        Params
        - evidence_vars:  A list of evidence variable data, where the ith item is a shape (N, E_i) batch of evidence
                           data for the ith evidence variable
        - data:           A shape (N, D) batch of data sampled from the conditional distribution
        """
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

class DiscriminativeMarkovChain(BayesNet):
    """
    Represents a simple Markov Chain
    X_1 -> X_2 -> ... -> X_n

    The CPDs can be set using ONE of the following methods:

    (1) generate_cpds(): randomly generates parameters for linear CPDs
    (2) fit_cpds_to_data(): fits linear CPDs according to data sampled from the true joint
    """

    def __init__(self, num_nodes):
        super(DiscriminativeMarkovChain, self).__init__(num_nodes)

    def generate_cpds(self, degree=1, coeff_range=[-5, 5], max_sd=2):
        vals, covs = [], []
        for i in range(1, self.num_nodes):
            # Generate random coefficients for each degree
            coeffs = []
            for deg in range(degree + 1):
                coeff = np.random.uniform(*coeff_range)
                coeffs.append(coeff)
            vals.append(tuple(coeffs))

            # Generate random SD (TODO: handle multivariate case - covariance matrix)
            sd = np.random.uniform(max_sd)
            covs.append(sd)

        return self.specify_polynomial_cpds((0, 1), vals, covs)

    def _create_polynomial_cond_fn(self, coeffs, sd):
        def cond_fn(evidence):
            mean = sum([coeff * (evidence[0] ** deg) for deg, coeff in enumerate(reversed(coeffs))])
            return GaussianDistribution(mean, sd)

        return cond_fn

    def specify_polynomial_cpds(self, prior, vals, covs):
        """
        Params
            - prior (tuple):      The mean and SD of X_1

            - vals (list[tuple]): A list of polynomial coefficients for each CPD.
                                    For example, for a chain of length 3 with quadratic Gaussian CPDs,
                                    len(vals) = 2 and each tuple in vals has 3 items.

            - covs (list[tensor]): A list of covariance matrices for each CPD.
        """
        assert len(vals) == self.num_nodes - 1
        assert len(covs) == self.num_nodes - 1

        parameters = []
        end_idx = self.num_nodes
        self.set_prior(f"X_{end_idx}", GaussianDistribution(prior[0], prior[1], linear_coeffs=(0, prior[0]), sd=prior[1]))

        for i, (coeffs, cov) in enumerate(zip(vals, covs), 1):
            parameters.append({"coeffs": coeffs, "sd": cov})
            cond_fn = self._create_polynomial_cond_fn(coeffs, cov)
            self.set_parents(f"X_{i}", [f"X_{end_idx}"], GaussianCPD(cond_fn, linear_coeffs=coeffs, sd=cov))

        self.build()
        return parameters

    def fit_cpds_to_data(self, data, log_fn=None):
        """
        Params
            data (list[tensor]) - A list containing values for each variable, sampled from the joint distribution
            log_fn (function)   - A function that takes three arguments: the node number, epoch number, and epoch data
        """
        data = torch.tensor(data)
        parameters = []

        end_idx = self.num_nodes - 1
        xend_prior = GaussianDistribution.fit_to_data(self._get_data(data, end_idx))
        self.set_prior(f"X_{end_idx + 1}", xend_prior)

        for i in range(1, self.num_nodes):
            print(f"\nFitting X_{i}\n==========")
            evidence = [self._get_data(data, end_idx)]
            vals = self._get_data(data, i - 1)

            # Fit a linear Gaussian CPD to the data, and save the learned parameters for testing/debugging
            new_log_fn = None if not log_fn else (lambda num: (lambda *args: log_fn(num, *args)))(i)
            cpd, cond_fn_approx = GaussianCPD.fit_linear_to_data(evidence, vals, new_log_fn)
            parameters.append({
                "a": cond_fn_approx.weights[0].weight.squeeze().item(),
                "b": cond_fn_approx.weights[0].bias.item(),
                "sd": cond_fn_approx.cov.item()
            })

            # Set the actual CPD according to the fitted values
            self.set_parents(f"X_{i}", [f"X_{end_idx + 1}"], cpd)

        self.build()
        return (xend_prior.mean, xend_prior.cov), parameters

    def fit_ELBO(self, data, fitted_mc, curr_evidence=1):
        """
        Params
            data (list[tensor]) - A list containing values for each variable, sampled from the joint distribution
            log_fn (function)   - A function that takes three arguments: the node number, epoch number, and epoch data
        """


        end_idx = self.num_nodes - 1
        evidence_dims = [1]
        data_dim = 1

        # Define a bunch of models for the linear gaussian controllers to be fit
        models = []
        for i in range(self.num_nodes - 1):
            model = LinearGaussianConditionalFn(evidence_dims, data_dim)
            models.append(model)

        # Variational family q(X1|X5)q(X2|X5)q(X3|X5)q(X4|X5)

        # The function that computes the ELBO
        def variational_loss(datapoints, models, num_nodes, fitted_mc):
            # Fitted mc is p_hat
            # models is q

            # Extracting the last node evidence
            evidence = datapoints

            # Get the samples from the encoder
            variational_samples = []
            for i in range(num_nodes - 1):
                # A single sample from the variational family per X1, X2, X3, X4...
                variational_samples.append(models[i]([evidence]).sample()[0])
            variational_samples.append(evidence)

            # Get logprobs from the fitted model
            log_probs = []
            for i in range(num_nodes):
                node = fitted_mc.get_node(f"X_{i + 1}")
                evidence_nodes = node.parents
                evidence_idx = [int(e.split('_')[1]) for e in evidence_nodes]
                evidences = variational_samples[evidence_idx[0]] if len(evidence_idx) > 0 else []
                x = variational_samples[i]
                
                with torch.no_grad():
                # Special case of distribution vs CPD
                # E_q[log p(X1)p(X2|X1)p(X3|X2)p(X4|X3)p(X5|X4))] - H(q)
                    if evidences == []:
                        log_probs.append(torch.mean(node.cpd.get_log_probability(x)))
                    else:
                        log_probs.append(torch.mean(node.cpd.get_log_probability(x, [evidences])))


            # Get the additional entropy terms for q -> E_q(-log q)
            # get log probs
            q_entropies = []
            for i in range(num_nodes - 1):
                q_entropies.append(torch.mean(models[i]([evidence]).get_log_probability(variational_samples[i])))

            # Combine the loss overall
            overall_loss = 0
            # NLL
            for lp in log_probs:
                overall_loss -= lp
            # Negative entropy
            for qe in q_entropies:
                overall_loss += qe
            return overall_loss

        # Setting up pytorch iteration
        dataset_size = 10000
        batch_size = 32
        num_epochs = 50
        evidence_data = curr_evidence*np.ones((dataset_size,))
        # Iterable that gives data from training set in batches with shuffling
        trainloader = torch.utils.data.DataLoader(Dataset(evidence_data), batch_size=batch_size,
                                                  shuffle=True)
        # Parameters to optimize with
        all_params = []
        for i in range(self.num_nodes - 1):
            all_params += list(models[i].parameters())

        optimizer = optim.Adam(all_params)

        # Pytorch training loop
        for epoch in range(num_epochs):
            total_loss = 0

            for i, d in enumerate(trainloader):
                optimizer.zero_grad()
                loss = variational_loss(d[:, None], models, self.num_nodes, fitted_mc)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            # Print statistics
            print(f"\nEpoch {epoch}; Avg Loss: {total_loss / len(trainloader)}")  # Avg loss per batch
            epoch += 1

        return models

    def _get_data(self, data, num):
        return data[:, num].unsqueeze(1)


if __name__ == "__main__":
    test_markov_chain(length=3, num_samples=10000)

