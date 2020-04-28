import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from distributions import GaussianDistribution

class ConditionalDataset(torch.utils.data.Dataset):
    def __init__(self, evidence_vars, data):
        """
        Params
        - evidence_vars:  A list of evidence variable data, where the ith item is a shape (N, E_i) batch of evidence
                           data for the ith evidence variable
        - data:           A shape (N, D) batch of data sampled from the conditional distribution
        """
        self.evidence_vars = evidence_vars
        self.data = data

    def __getitem__(self, i):
        return tuple(e[i] for e in self.evidence_vars), self.data[i]

    def __len__(self):
        return len(self.data)

class LinearGaussianConditionalFn(nn.Module):
    """
    Represents a Gaussian CPD where the mean is a linear function of the evidence, 
     and the covariance matrix does not depend on the conditioning variables.
    """
    def __init__(self, evidence_dims, data_dim):
        """
        Params
            evidence_dims (list[int]) - A list of dimensions for each of the evidence variables
            data_dim (int)            - The dimensionality of the data points
        """
        super(LinearGaussianConditionalFn, self).__init__()
        
        # Initialize one set of linear weights for each evidence variable
        self.weights = nn.ModuleList()
        for dim in evidence_dims:
            self.weights.append(nn.Linear(dim, data_dim))
        self.data_dim = data_dim

        # Fixed covariance matrix (learnable), regardless of evidence
        self.cov = torch.nn.Parameter(torch.eye(data_dim) if data_dim > 1 else torch.tensor(1.))

    def cov_matrix(self):
        # Scalar case: don't do anything
        if len(self.cov.shape) == 0:
            return self.cov

        # To ensure that covariance matrix is PD, don't use self.cov directly.
        #  Instead, let L = lower diagonal matrix with contents of self.cov,
        #  and use LL^T + eps*I as the covariance matrix.
        L = torch.tril(self.cov)
        cov = L @ L.T + 1e-8 * torch.eye(L.shape[0])
        return torch.clamp(cov, min=0)
    
    def forward(self, evidence):
        """
        Outputs a GaussianDistribution object representing P(X|evidence)

        Params
            evidence (list[torch.tensor]) - A list containing values for each evidence variable
        """
        mean = sum([self.weights[i](evidence.float()) for i, evidence in enumerate(evidence)])
        cov = self.cov_matrix()

        return GaussianDistribution(mean, cov)

    def loss(self, data, distr):
        """
        Returns the negative log likelihood of the data (for doing MLE via gradient descent).

        Params
            data  (tensor)               - A batch of data points
            distr (GaussianDistribution) - The current learned distribution
        """
        return -torch.sum(distr.get_log_prob(data))


class NeuralNetGaussianConditionalFn(nn.Module):
    """
    TOOD: implement neural net function approximator for arbitrary Gaussian CPDs
    """
    pass


def learn_gaussian_conditional_fn(cond_fn_approx, evidence, data, num_epochs=25, batch_size=16, verbose=True, log_fn=None):
    """
    Given evidence and data, uses MLE to learn the optimal parameters for cond_fn, 
     which maps evidence values to a GaussianDistribution object (with a mean and covariance). 

    Params
        cond_fn_approx (nn.Module)  -   A function approximator with a learnable set of parameters
                                         (e.g. LinearGaussianConditionalFn)

        evidence (list[tensor])     -   A list of evidence variable data, where the ith item is a shape [N, E_i] 
                                         batch of evidence data for the ith evidence variable

        data (list[tensor])         -   A shape [N, D] batch of data sampled from the conditional distribution

    Returns
        cond_fn - the learned function mapping evidence to GaussianDistributions
    """

    # Iterable that gives data from training set in batches with shuffling
    trainloader = torch.utils.data.DataLoader(ConditionalDataset(evidence, data), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(cond_fn_approx.parameters())

    for epoch in range(num_epochs):
        total_loss = 0

        for i, (evidence, data) in enumerate(trainloader):
            optimizer.zero_grad()

            output_distr = cond_fn_approx(evidence)
            loss = cond_fn_approx.loss(data, output_distr)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Log current parameters
        a = cond_fn_approx.weights[0].weight.squeeze()
        b = cond_fn_approx.weights[0].bias.squeeze()
        with torch.no_grad():
            cov = cond_fn_approx.cov_matrix()
        if verbose:
            print(f"a: {a}\nb: {b}\nCov: {cov}")
        if log_fn:
            log_fn(epoch, {"a": a, "b": b, "cov": cov})

        # Print statistics
        print(f"\nEpoch {epoch}; Avg Loss: {total_loss / len(trainloader)}") # Avg loss per batch
        epoch += 1
        total_loss = 0
        
    # No gradients during actual evaluation
    for param in cond_fn_approx.parameters():
        param.requires_grad_(False)

    def cond_fn(evidence):
        distr = cond_fn_approx(evidence)
        return distr

    return cond_fn, cond_fn_approx

def fit_VI(data, mc, variational_mc, plot_name="vi_loss"):
    """
    Parameters
    ----------
    data : dict[str, tensor]
        A named dataset where the keys are the node names, and the values are
        a list of sampled values for that node

    mc : BayesNet
        Represents p, the true joint distribution

    """
    end_idx = mc.num_nodes

    def variational_loss(evidence):
        # Get a labeled sample from q(x)
        sample = variational_mc.sample_labeled(evidence_dict={f"X_{end_idx}": evidence})

        # log q(x) - everything except for the evidence
        q_entropies = variational_mc.get_log_prob(sample, exclude=[f"X_{end_idx}"])

        # log p(x)
        log_probs = mc.get_log_prob(sample)

        # D_KL(q||p) = E_q[log q(x) - log p(x)]
        return torch.mean(q_entropies) - torch.mean(log_probs)

    # Setting up pytorch iteration
    dataset_size = 10000
    batch_size = 32
    num_epochs = 25

    # Iterable that gives data from training set in batches with shuffling
    evidence_data = data[f"X_{end_idx}"]
    trainloader = torch.utils.data.DataLoader(evidence_data, batch_size=batch_size,
                                                shuffle=True)
    # Parameters to optimize with
    params = []
    for node in variational_mc.all_nodes():
        if node.cpd.is_empty:
            params += list(node.cpd.learnable_params())
    optimizer = optim.Adam(params)

    print("Begin training loop")
    train_losses = []
    # Pytorch training loop
    for epoch in range(num_epochs):
        total_loss = 0

        for i, d in enumerate(trainloader):
            optimizer.zero_grad()
            loss = variational_loss(d[:, None])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Print statistics
        print(f"\nEpoch {epoch}; Avg Loss: {total_loss / len(trainloader)}")  # Avg loss per batch
        # print(models[0].weights[0].weight.item(), models[0].weights[0].bias.item())
        # print(models[0].cov_matrix())
        train_losses += [total_loss / len(trainloader)]
        plt.plot(train_losses)
        plt.savefig(f"{plot_name}.png")
        epoch += 1

    for node in variational_mc.all_nodes():
        if node.cpd.is_empty:
            node.cpd.freeze_values()

    return variational_mc