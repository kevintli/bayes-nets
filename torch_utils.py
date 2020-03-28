import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import functools

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

class LinearGaussianCPD(nn.Module):
    def __init__(self, evidence_dims, data_dim):
        super(LinearGaussianCPD, self).__init__()

        self.weights = nn.ModuleList()
        for dim in evidence_dims:
            self.weights.append(nn.Linear(dim, data_dim))
        self.data_dim = data_dim
    
    def forward(self, evidence_vars):
        """
        Output the conditional mean of Gaussian distribution given evidence
        """
        result = torch.zeros(evidence_vars[0].shape[0], self.data_dim) # (N, D) tensor of output means
        for i, evidence in enumerate(evidence_vars):
            result += self.weights[i](evidence.float())
        return result

    def loss(self, data, mean, cov_inv):
        """
        Returns (x - mu)^T Sigma^-1 (x - mu), 
        the loss for doing MLE on the mean of a conditional Gaussian distribution
        """
        return torch.trace((data.float() - mean) @ ((data.float() - mean) @ torch.Tensor(cov_inv).float()).T)
        

def _dim(arr):
    if len(arr.shape) > 1:
        return arr.shape[1]
    return 1

def learn_linear_gaussian_params(evidence_vars, data):
    """
    Returns the parameters for a linear conditional Gaussian, where the mean is a linear function 
     of the evidence, and the covariance matrix does not depend on the conditioning variables.

    Params
    - evidence: A tuple of evidence variable data, where the ith item is a shape (N, E_i) batch of evidence
                 data for the ith evidence variable
    - data:     A shape (N, D) batch of data sampled from the conditional distribution

    Returns
    - an nn.Module object that maps evidence to a mean and covariance
    """
    cov = np.cov(data.T)
    if cov.shape:
        cov_inv = np.linalg.inv(cov)
    else: # 1D case
        cov_inv = np.array([[1 / cov]])

    cpd = LinearGaussianCPD([_dim(e) for e in evidence_vars], _dim(data))

    # Iterable that gives data from training set in batches with shuffling
    trainloader = torch.utils.data.DataLoader(ConditionalDataset(evidence_vars, data), batch_size=4, shuffle=True)
    optimizer = optim.SGD(cpd.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(3):
        # Accumulated over up to 2000 mini-batches; used to compute running average loss
        running_loss = 0.0  

        for i, (evidence, data) in enumerate(trainloader):
            optimizer.zero_grad()

            outputs = cpd(evidence)
            loss = cpd.loss(data, outputs, cov_inv)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000)) # Average loss per mini-batch
                running_loss = 0.0

    def mean_cov_fn(evidence):
        with torch.no_grad():
            mean = cpd(torch.Tensor([evidence]))
        return GaussianDistribution(mean.squeeze(), cov)

    return mean_cov_fn