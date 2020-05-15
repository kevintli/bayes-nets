import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from distributions import GaussianDistribution
from inference import compute_joint_linear, kl_div_gaussian

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

class GaussianConditionalFn(nn.Module):
    def __init__(self, data_dim):
        super(GaussianConditionalFn, self).__init__()

        # Fixed covariance matrix (learnable), regardless of evidence
        self.cov = torch.nn.Parameter(torch.eye(data_dim) if data_dim > 1 else torch.tensor(1.))
        self.data_dim = data_dim

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

    def loss(self, data, distr):
        """
        Returns the negative log likelihood of the data (for doing MLE via gradient descent).

        Params
            data  (tensor)               - A batch of data points
            distr (GaussianDistribution) - The current learned distribution
        """
        return -torch.sum(distr.get_log_prob(data))


class LinearGaussianConditionalFn(GaussianConditionalFn):
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
        super(LinearGaussianConditionalFn, self).__init__(data_dim)
        
        # Initialize one set of linear weights for each evidence variable
        self.weights = nn.ModuleList()
        for dim in evidence_dims:
            self.weights.append(nn.Linear(dim, data_dim))
    
    def forward(self, evidence):
        """
        Outputs a GaussianDistribution object representing P(X|evidence)

        Params
            evidence (list[torch.tensor]) - A list containing values for each evidence variable
        """
        mean = sum([self.weights[i](evidence.float()) for i, evidence in enumerate(evidence)])
        cov = self.cov_matrix()

        return GaussianDistribution(mean, cov)


class NeuralNetGaussianConditionalFn(GaussianConditionalFn):
    """
    TOOD: implement neural net function approximator for arbitrary Gaussian CPDs
    """
    def __init__(self, evidence_dims, data_dim, hidden_layers=2, layer_size=32, nonlinearity=F.relu):
        super(NeuralNetGaussianConditionalFn, self).__init__(data_dim)

        self.layers = nn.ModuleList()
        input_dim = sum(evidence_dims)
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(input_dim, layer_size))
            input_dim = layer_size
        self.fc_out = nn.Linear(input_dim, data_dim)
        self.nonlinearity = nonlinearity

    def forward(self, evidence):
        """
        Outputs a GaussianDistribution object representing P(X|evidence)

        Parameters
        ==========
        evidence : list[torch.tensor]
            A list containing values for each evidence variable
        """
        curr = torch.cat(evidence, axis=1)
        for layer in self.layers:
            curr = self.nonlinearity(layer(curr))
        mean = self.fc_out(curr)
        cov = self.cov_matrix()
        return GaussianDistribution(mean, cov)


def create_stopping_criterion(recent_epochs=20, eps=3e-3):
    recent_losses = []

    def should_stop(loss):
        recent_losses.append(loss)
        if len(recent_losses) > recent_epochs:
            recent_losses.pop(0)
            
            diffs = 0
            for i in range(1, recent_epochs):
                diffs += abs(recent_losses[i] - recent_losses[i-1])
            
            if diffs / (recent_epochs - 1) < eps:
                return True
        
        return False

    return should_stop


def learn_gaussian_conditional_fn(cond_fn_approx, evidence, data, batch_size=128, eps=5e-3, recent_epochs=20, max_epochs=100, verbose=True, log_fn=None):
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

    recent_losses = []

    max_epochs = int(5e5 // len(data))
    stop_criterion = create_stopping_criterion()

    for epoch in range(max_epochs):
        total_loss = 0

        for i, (evidence, data) in enumerate(trainloader):
            optimizer.zero_grad()

            output_distr = cond_fn_approx(evidence)
            loss = cond_fn_approx.loss(data, output_distr)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Log current parameters
        if log_fn:
            log_fn(cond_fn_approx)

        # Print statistics
        avg_loss = total_loss / len(trainloader)
        print(f"\nEpoch {epoch}; Avg Loss: {avg_loss}") # Avg loss per batch
        epoch += 1

        # Check stopping criterion
        if stop_criterion(avg_loss):
            print("Stopping early.")
            break
        
    # No gradients during actual evaluation
    for param in cond_fn_approx.parameters():
        param.requires_grad_(False)

    return cond_fn_approx


def make_log_fn_with_node_name(node_name, log_fn):
    if not log_fn:
        return None

    def log_fn_wrapper(*args):
        log_fn(node_name, *args)
    return log_fn_wrapper


def fit_MLE(data, p_hat, batch_size=32, log_fn=None):
    """
    Parameters
    ----------
    data : dict[str, tensor]
        A named dataset where the keys are the node names, and the values are
        a list of sampled values for that node

    p_hat : BayesNet
        Represents the distribution to fit via MLE.
        Only learnable factors (is_learnable = True) will be fit, so it is possible to pass in a p_hat
            with some already specified factors and only fit the rest.
    """
    for node in p_hat.all_nodes():
        print(f"Learning {node.name}")
        if node.cpd.is_learnable:
            evidence = [data[parent] for parent in node.parents]
            node.cpd.fit_to_data(evidence, data[node.name], batch_size=batch_size, log_fn=make_log_fn_with_node_name(node.name, log_fn)) if evidence \
                else node.cpd.fit_to_data(data[node.name], batch_size=batch_size)
    
    return p_hat

def reverse_KL_linear(p, q, evidence_node, evidence, query_node="X_1"):
    """
    Returns the "reverse" KL divergence (between q and the true posterior).

    For debugging purposes, can be used as the loss function (instead of ELBO loss) 
     for learning q when the exact posterior p is known.

    Parameters
    ----------
    p : BayesNet
        Represents the joint distribution p(x, z). MUST BE a linear Gaussian model
         since we need to do exact inference on p(z|x).

    q : BayesNet
        Represents the learned variational distribution q(z|x)

    evidence_node : str
        The name of the evidence node (x)

    evidence : Tensor
        A (batch_size, evidence_dim) tensor of evidence values
    """

    # Sample z ~ q(z|x)
    sample = q.sample_labeled(evidence_dict={evidence_node: evidence})

    # Compute log q(z|x)
    q_entropies = q.get_log_prob(sample, exclude=[evidence_node])

    # Compute log p(z|x)
    end_idx = int(evidence_node.split("_")[1])
    log_probs = []
    for i in range(1, end_idx):
        true_mean, true_cov = compute_joint_linear(p, f"X_{i}", evidence_node, evidence)
        # print(sample[f'X_{i}'])
        log_probs.append(GaussianDistribution(true_mean, true_cov).get_log_prob(sample[f"X_{i}"]))
        # print("result shape:", result.shape)

    # Sum the log probs across nodes for each sample
    log_probs = torch.sum(torch.cat(log_probs, axis=1), axis=1)

    # D_KL(q||p) = E_q[log q(z|x) - log p(z|x)]
    return torch.mean(q_entropies) - torch.mean(log_probs)
    

def variational_loss(p, q, evidence_node, evidence):
    """
    The loss function for variational inference (negative ELBO).

    J(q) = D_KL(q(z|x) || p(x, z)) = D_KL(q(z|x) || p(z|x)) - log p(x)

    Parameters
    ----------
    p : BayesNet
        Represents the joint distribution p(x, z)

    q : BayesNet
        Represents the learned variational distribution q(z|x)

    evidence_node : str
        The name of the evidence node (x)

    evidence : Tensor
        A (batch_size, evidence_dim) tensor of evidence values
    """

    # Get a labeled sample from q(z|x)
    sample = q.sample_labeled(evidence_dict={evidence_node: evidence})
    # log q(z|x) - everything except for the evidence
    q_entropies = q.get_log_prob(sample, exclude=[evidence_node])
    # log p(x, z)
    log_probs = p.get_log_prob(sample)
    # D_KL(q||p) = E_q[log q(z|x) - log p(x, z)]
    return torch.mean(q_entropies) - torch.mean(log_probs)


def fit_VI(data, mc, variational_mc, loss_fn=variational_loss, ideal_variational_mc=None, num_epochs=200, batch_size=128, plot_name="vi_loss"):
    """
    Parameters
    ----------
    data : dict[str, tensor]
        A named dataset where the keys are the node names, and the values are
        a list of sampled values for that node

    mc : BayesNet
        Represents p, the true joint distribution

    variational_mc : BayesNet
        Represents q, the distribution to learn via VI

    For DEBUGGING PURPOSES ONLY:
    ---
    loss_fn : function (optional)
        The loss to use for VI. Can switch this to something else like reverse_KL_linear to test performance

    ideal_variational_mc : BayesNet
        Represents the true posterior. When plotting the loss over epochs, we will compare the loss on the learned q vs. this ideal q.
    """
    end_idx = mc.num_nodes

    # Iterable that gives data from training set in batches with shuffling
    evidence_data = data[f"X_{end_idx}"]
    trainloader = torch.utils.data.DataLoader(evidence_data, batch_size=batch_size,
                                                shuffle=True)
    # Parameters to optimize with
    params = []
    for node in variational_mc.all_nodes():
        if node.cpd.is_learnable:
            params += list(node.cpd.learnable_params())
    optimizer = optim.Adam(params)

    print("Begin training loop")
    train_losses = []
    ideal_losses = []
    stop_criterion = create_stopping_criterion()

    # Pytorch training loop
    for epoch in range(num_epochs):
        total_loss = 0
        total_ideal_loss = 0

        for i, d in enumerate(trainloader):
            optimizer.zero_grad()
            loss = loss_fn(mc, variational_mc, f"X_{end_idx}", d)
            total_loss += loss.item()

            if ideal_variational_mc:
                ideal_loss = loss_fn(mc, ideal_variational_mc, f"X_{end_idx}", d)
                total_ideal_loss += ideal_loss.item()

            loss.backward()
            optimizer.step()

        # Print statistics
        avg_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch}; Avg Loss: {avg_loss}")  # Avg loss per batch
        train_losses += [avg_loss]
        
        if ideal_variational_mc:
            ideal_losses += [total_ideal_loss / len(trainloader)]

        epoch += 1

        if stop_criterion(avg_loss):
            print("Stopping early.")
            break

    plt.figure()
    plt.plot(train_losses, color='r', label="VI loss (q)")
    if ideal_losses:
        plt.plot(ideal_losses, color='b', label="VI loss (ideal)")  
        plt.legend()
    plt.savefig(f"{plot_name}.png")

    for node in variational_mc.all_nodes():
        if node.cpd.is_learnable:
            node.cpd.freeze_values()

    return variational_mc