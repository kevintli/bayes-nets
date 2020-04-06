import torch
import torch.nn as nn
import torch.optim as optim

from distributions import GaussianDistribution

class ConditionalDataset(torch.utils.data.Dataset):
    def __init__(self, evidence_vars, data):
        """
        Params
        - evidence_vars:  A tuple of evidence variable data, where the ith item is a shape (N, E_i) batch of evidence
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
    
    def forward(self, evidence):
        """
        Outputs a GaussianDistribution object representing P(X|evidence)

        Params
            evidence (list[torch.tensor]) - A list containing values for each evidence variable
        """
        mean = sum([self.weights[i](evidence.float()) for i, evidence in enumerate(evidence)]).squeeze()
        return GaussianDistribution(mean, self.cov)

    def loss(self, data, distr):
        """
        Returns the negative log likelihood of the data (for doing MLE via gradient descent).

        Params
            data  (tensor)               - A batch of data points
            distr (GaussianDistribution) - The current learned distribution
        """
        return -torch.sum(distr.get_log_prob(data).diag())

class NeuralNetGaussianConditionalFn(nn.Module):
    """
    TOOD: implement neural net function approximator for arbitrary Gaussian CPDs
    """
    pass


def learn_gaussian_conditional_fn(cond_fn_approx, evidence, data, num_epochs=30, batch_size=16):
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
        
        print(f"\nEpoch {epoch}; Avg Loss: {total_loss / len(trainloader)}") # Avg loss per batch
        print(cond_fn_approx.weights[0].weight)
        print(cond_fn_approx.cov)
        epoch += 1
        total_loss = 0

    def cond_fn(evidence):
        with torch.no_grad():
            distr = cond_fn_approx(evidence)
        return distr

    return cond_fn, cond_fn_approx