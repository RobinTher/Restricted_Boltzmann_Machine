import torch
import numpy as np
from functools import partial
import gc

def build_m_diag_range(m_range):
    '''
    Extract the largest P elements of Mattis magnetization matrices of size (P, P_t), P <= P_t contained in m_range
    where elements on the same row and column of each extracted element are set to zero after extraction.
    These P largest elements are considered as the "diagonal" elements of the matrices as in the paper.
    This manipulation counteracts permutation symmetry of the Mattis magnetization.

    Inputs
    ------
    m_range (torch.Tensor):
        A tensor of shape (number_samples, P, P_t) containing the Mattis magnetization.
    
    Outputs
    -------
    m_diag_range (torch.Tensor):
        A tensor of shape (number_samples, P) containing the extracted "diagonal" elements.
    '''
    number_samples, P, P_t = m_range.shape
    m_range_view = np.reshape(m_range, newshape = (number_samples, -1))
    m_diag_range = np.zeros((number_samples, P))
    sample_range = range(number_samples)
    for i in range(P):
        max_index = np.argmax(m_range_view, axis = 1)
        max_index = np.column_stack(np.unravel_index(max_index, (P, P_t)))
        
        m_diag_range[:, i] = m_range[sample_range, max_index[:, 0], max_index[:, 1]]
        # print(m_diag_range[:, i])
        m_range[sample_range, max_index[:, 0], :] = 0
        m_range[sample_range, :, max_index[:, 1]] = 0
        # print(m_range)
    
    return m_diag_range

def ix_(N_batch, P_batch):
    '''
    Create a meshgrid of indices for a batch of samples.
    '''
    return N_batch[:, None], P_batch[None, :]

def init_spins(P):
    '''
    Initialize spins for the Restricted Boltzmann Machine.
    The spins are generated as a Cartesian product of {-1, 1}^P.
    
    Inputs
    ------
    P (int):
        Number of spins.
    Outputs
    -------
    spins (torch.Tensor):
        Spins array with shape (2**P, P, 1).
    '''
    spins_base = np.full((P, 2), np.array([-1., 1.]))
    
    spins = np.reshape(np.array(np.meshgrid(*spins_base)).T, newshape = (-1,) + (P, 1))
    
    return spins

def logcosh(x):
    '''
    Helper function.
    '''
    c = torch.maximum(-x, x)
    
    return c + torch.log1p(torch.exp(-2*c)) - torch.log(torch.tensor(2))

def sech(x):
    '''
    Helper function.
    '''
    return 1/torch.cosh(x)

def rademacher(prob, generator : torch.Generator):
    '''
    Generate Rademacher random variables
    i.e. 1 with probability prob, -1 with probability 1-prob.

    Inputs
    ------
    prob (torch.Tensor):
        Probability of generating 1.
    generator (torch.Generator):
        Random number generator.
    Outputs
    -------
    torch.Tensor:
        Rademacher random variables.
    '''
    return 2*torch.bernoulli(prob, generator = generator)-1

class RBM(torch.nn.Module):
    '''
    Restricted Boltzmann Machine class.
    Inherits from torch.nn.Module.
    Initializes the weights of the RBM and provides methods for training the RBM.
    Inputs
    ------
    N (int):
        Number of visible units.
    P (int):
        Number of hidden units.
    device (torch.device):
        Device on which to perform computations.
    random_number_generator (torch.Generator):
        Random number generator to initialize weights and add stochasticity during training.
    random_batch_generator (torch.Generator):
        Random number generator to sample training batches.
    teacher (RBM, optional):
        Teacher RBM that genererated the data used to train the this RBM.
    
    Attributes not defined in the inputs
    ------------------------------------
    xi (torch.nn.Parameter):
        Weights of the RBM.
    backup_xi (torch.nn.Parameter):
        Backup of the weights of the RBM.
    tau (torch.Tensor):
        All possible hidden unit configurations.
    velocity (torch.nn.Parameter):
        Velocity for updating the weights when using momentum in ord update.

    Methods
    -------
    backup_weights:
        Backup the weights of the RBM.
    initialize_weights:
        Initialize the weights of the RBM.
    sample_hidden_given_visible:
        Sample hidden units given visible units.
    sample_visible_given_hidden:
        Sample visible units given hidden units.
    sample_visible_and_hidden:
        Sample visible and hidden units together using Gibbs sampling.
    gibbs_sample_visible:
        Sample visible units using Gibbs sampling.
    free_entropy:
        Compute the partial free entropy of the RBM.
        log(psi) in the paper (see Introduction).
    free_entropy_difference:
        Compute the difference in partial free entropy between two states.
    log_likelihood_difference:
        Compute the difference in log-likelihood between two states to use the Metropolis algorithm.
    contrastive_divergence:
        Compute the contrastive divergence loss, reconstruction error and gradient.
    ord_update:
        Update the weights of the RBM using underdamped stochastic Langevin dynamics of the contrastive divergence gradient.
    langevin_training:
        Train the RBM using underdamped stochastic Langevin dynamics.
        Run the ord_update method for a number of training epochs.
    Metropolis_training:
        Train the RBM using the Metropolis algorithm.
        Accept/reject weight spin flips based on the difference in log-likelihood between states.
    '''
    def __init__(self, N : int, P : int, device : torch.device, random_number_generator : torch.Generator, random_batch_generator : torch.Generator, teacher = None):
        super(RBM, self).__init__()
        
        self.N = N
        self.P = P
        
        self.training_device = device
        
        self.random_number_generator = random_number_generator
        
        self.random_batch_generator = random_batch_generator
        
        self.teacher = teacher
        
        self.xi = torch.nn.Parameter(torch.zeros((N, P), device = device), requires_grad = False)
        
        self.backup_xi = torch.nn.Parameter(torch.zeros((N, P), device = device), requires_grad = False)
        
        self.tau = torch.from_numpy(np.float32(init_spins(P))).to(device)
        
        self.velocity = 1/N**(1/2)*torch.nn.Parameter(torch.randn((N, P), device = device, generator = random_number_generator), requires_grad = False)
        
        # self.energy = torch.nn.Parameter(torch.tensor(0.), requires_grad = False)
        
        # self.potential_energy = torch.nn.Parameter(torch.tensor(0.), requires_grad = False)
    
    def backup_weights(self):
        '''
        Backup the weights of the RBM.
        '''
        self.backup_xi.copy_(self.xi.detach().clone())
    
    def initialize_weights(self, m_0 : float, xi_0 = None, P = 0, binarize = False, bias_corner = False, mat_cor = None):
        '''
        Initialize the weights of the RBM.

        Inputs
        ------
        m_0 (float):
            Initial magnetization, i.e. overlap with the weights xi_0.
        xi_0 (torch.Tensor, optional):
            See m_0.
        P (int, defaults to 0):
            Number of hidden units.
        binarize (bool, defaults to False):
            Whether to binarize the weights, i.e. set weights = sign(weights).
        bias_corner (bool, defaults to False):
            Whether to use m_0 to initialize the columns j > xi_0.shape[1] of the weights.
        mat_cor (torch.Tensor, optional):
            Correlations between the columns of the weights.
        Outputs
        -------
        xi (torch.Tensor):
            Initialized weights.
        '''
        P_t = self.P
        xi = torch.randn((self.N - P, P_t), device = self.training_device, generator = self.random_number_generator)
        
        xi.copy_((xi - torch.flip(xi, dims = (0,)))/torch.sqrt(torch.tensor(2)))
        
        C = (torch.transpose(xi, 0, 1) @ xi) / self.N
        
        L = torch.linalg.cholesky(C)
        xi.copy_(torch.linalg.solve_triangular(L, xi.T, upper = False).T)
        
        if mat_cor is not None:
            L = torch.linalg.cholesky(mat_cor) # .to(self.training_device)
            xi = xi @ L.T
            # (N, P) = (N, P) @ (P, P)
        
        elif xi_0 is not None:
            Q, R = torch.linalg.qr(xi_0, mode = "complete")
            
            xi = Q[:, P :] @ xi
            
            xi[:, 0 : P].copy_((1 - m_0**2)**(1/2) * xi[:, 0 : P] + m_0 * xi_0.detach())
            
            if bias_corner:
                xi[:, P : P_t].copy_((1 - m_0**2)**(1/2) * xi[:, P : P_t] + m_0 * xi_0[:, 0 : 1].detach())
        
        if binarize:
            self.xi.copy_(torch.sign(xi))
        else:
            self.xi.copy_(1/self.N**(1/2) * xi)
        
        return xi
    
    def sample_hidden_given_visible(self, tau, sigma, beta : float):
        '''
        Sample hidden units given visible units.

        Inputs
        ------
        tau (torch.Tensor):
            Hidden units.
        sigma (torch.Tensor):
            Visible units.
        beta (float):
            Inverse temperature.
        Outputs
        -------
        P_tau_given_sigma (torch.Tensor):
            Probability of hidden units given visible units.
        '''
        P_tau_given_sigma = torch.sigmoid(2 * beta * sigma @ self.xi)
        # (M, P) = (M, N) @ (N, P)
        
        tau.copy_(rademacher(P_tau_given_sigma, generator = self.random_number_generator))
        
        return P_tau_given_sigma
    
    def sample_visible_given_hidden(self, sigma, tau, beta : float):
        '''
        Sample visible units given hidden units.

        Inputs
        ------
        sigma (torch.Tensor):
            Visible units.
        tau (torch.Tensor):
            Hidden units.
        beta (float):
            Inverse temperature.
        Outputs
        -------
        P_sigma_given_tau (torch.Tensor):
            Probability of visible units given hidden units.
        '''
        P_sigma_given_tau = torch.sigmoid(2 * beta * tau @ torch.transpose(self.xi, 0, 1))
        # (M, N) = (M, P) @ (P, N)
        
        sigma.copy_(rademacher(P_sigma_given_tau, generator = self.random_number_generator))
        
        return P_sigma_given_tau
    
    def sample_visible_and_hidden(self, sigma, tau, beta : float, number_sampling_steps : int,
                                  number_monitored_sampling_steps : int, calculate_loss : bool):
        '''
        Sample visible and hidden units together using Gibbs sampling.

        Inputs
        ------
        sigma (torch.Tensor):
            Visible units.
        tau (torch.Tensor):
            Hidden units.
        beta (float):
            Inverse temperature.
        number_sampling_steps (int):
            Number of sampling steps.
        number_monitored_sampling_steps (int):
            Number of sampling steps where the partial free entropy is monitored.
        calculate_loss (bool):
            Whether to calculate the loss.
        Outputs
        -------
        P_sigma_given_tau (torch.Tensor):
            Probability of the sampled visible units.
        loss (torch.Tensor):
            Average partial free entropy difference of the RBM if calculated.
        '''
        f_0 = torch.tensor(0.)
        f = torch.tensor(0.)
        
        P_sigma_given_tau = torch.zeros_like(sigma, device = self.training_device)
        P_tau_given_sigma = torch.zeros_like(tau, device = self.training_device)
        
        if number_monitored_sampling_steps != 0:
            f_0 = torch.mean(self.free_entropy(sigma, self.xi, beta))
            print("Step [{}/{}], free entropy: {:.4f}".format(0, number_sampling_steps, f_0))
        
        elif calculate_loss:
            f_0 = torch.mean(self.free_entropy(sigma, self.xi, beta))
        
        for sampling_step in range(1, number_sampling_steps + 1):
            if number_monitored_sampling_steps != 0:
                monitor_this_step = sampling_step % (number_sampling_steps // number_monitored_sampling_steps) == 0
            else:
                monitor_this_step = False
            
            P_sigma_given_tau = self.sample_visible_given_hidden(sigma, tau, beta)
            
            P_tau_given_sigma = self.sample_hidden_given_visible(tau, sigma, beta)
            
            if monitor_this_step:
                f = torch.mean(self.free_entropy(sigma, self.xi, beta))
                print("Step [{}/{}], free entropy: {:.4f}".format(sampling_step, number_sampling_steps, f))
        
        if calculate_loss:
            f = torch.mean(self.free_entropy(sigma, self.xi, beta))
            loss = f - f_0
        else:
            loss = torch.tensor(0.)
        
        return P_sigma_given_tau, loss
    
    def gibbs_sample_visible(self, sigma, beta : float, number_sampling_steps : int,
                             number_monitored_sampling_steps : int):
        '''
        Sample visible units using Gibbs sampling.

        Inputs
        ------
        sigma (torch.Tensor):
            Visible units.
        beta (float):
            Inverse temperature.
        number_sampling_steps (int):
            Number of sampling steps.
        number_monitored_sampling_steps (int):
            Number of sampling steps where the partial free entropy is monitored.
        Outputs
        -------
        P_sigma_given_tau (torch.Tensor):
            Probability of the sampled visible units.
        '''
        tau = torch.zeros((len(sigma), self.P), device = self.training_device)
        
        P_tau_given_sigma = self.sample_hidden_given_visible(tau, sigma, beta)
        
        P_sigma_given_tau, _ = self.sample_visible_and_hidden(sigma, tau, beta, number_sampling_steps,
                                                              number_monitored_sampling_steps = number_monitored_sampling_steps,
                                                              calculate_loss = False)
        
        del tau
        
        return P_sigma_given_tau
    
    def free_entropy(self, sigma, xi, beta : float):
        '''
        Compute the partial free entropy of the RBM.
        log(psi) in the paper (see Introduction).

        Inputs
        ------
        sigma (torch.Tensor):
            Visible units.
        xi (torch.Tensor):
            RBM weights.
        beta (float):
            Inverse temperature.
        Outputs
        -------
        f (torch.Tensor):
            Partial free entropy.
        '''
        f = torch.sum(logcosh(beta * sigma @ xi), dim = 1)
        
        return f
    
    def free_entropy_difference(self, N_batch, sigma, beta : float):
        '''
        Compute the difference in partial free entropy between two states.

        Inputs
        ------
        N_batch (torch.Tensor):
            Size of visible unit batch.
        sigma (torch.Tensor):
            Visible units.
        beta (float):
            Inverse temperature.
        Outputs
        -------
        d_E (torch.Tensor):
            Difference in partial free entropy.
        '''
        # Use beta/sqrt(N) as beta
        
        sigma_overlaps = sigma @ self.xi
        # (M, P) = (M, N) @ (N, P)
        
        h_1 = -2 * beta * torch.tanh(beta * sigma_overlaps) @ torch.transpose(self.xi[N_batch], 0, 1)
        # (M, N_batch) = (M, P) @ (P, N_batch)
        
        h_2 = 2 * beta**2 * torch.sum(sech(beta * sigma_overlaps)**2, dim = 1, keepdim = True)
        # (M, 1) = sum_P (M, P)
        
        d_E = h_1 * sigma[:, N_batch] + h_2
        
        return d_E
    
    def log_likelihood_difference(self, N_batch, P_batch, sigma, beta : float):
        '''
        Compute the difference in log-likelihood between two states to use the Metropolis algorithm.

        Inputs
        ------
        N_batch (torch.Tensor):
            Size of visible unit batch.
        P_batch (torch.Tensor):
            Size of hidden unit batch.
        sigma (torch.Tensor):
            Visible units.
        beta (float):
            Inverse temperature.
        Outputs
        -------
        d_E (torch.Tensor):
            Difference in log-likelihood.
        '''
        # Use beta/sqrt(N) as beta
        
        sigma_overlaps = sigma @ self.xi[:, P_batch]
        # (M, P_batch) = (M, N) @ (N, P_batch)
        
        h_1 = -2 * beta * torch.transpose(sigma, 0, 1)[N_batch] @ torch.tanh(beta * sigma_overlaps)
        # (N_batch, P_batch) = (N_batch, M) @ (M, P_batch)
        
        h_2 = 2 * beta**2 * torch.sum(sech(beta * sigma_overlaps)**2, dim = 0, keepdim = True)
        # (1, P_batch) = sum_M (M, P_batch)
        
        d_E = h_1 * self.xi[ix_(N_batch, P_batch)] + h_2
        
        tau_overlaps = self.xi @ self.tau
        # (2**P, N, 1) = (N, P) @ (2**P, P, 1)
        
        H = torch.sum(1/2*beta**2 * tau_overlaps**2, dim = 1, keepdim = True)
        # (2**P, 1, 1)
        
        tau_overlap_batch = self.xi[N_batch] @ self.tau
        # (2**P, N_batch, 1)
        
        c = torch.max(H)
        
        d_log_Z = -2 * beta**2 * len(sigma) * torch.mean(torch.exp(H - c) / torch.mean(torch.exp(H - c))
                                                         * (self.xi[N_batch, P_batch, None] * self.tau[:, P_batch] * tau_overlap_batch - 1), dim = 0)
        
        return d_E - d_log_Z
    
    def contrastive_divergence(self, sigma, beta : float, number_sampling_steps : int,
                               monitor_sampling : bool, calculate_loss : bool):
        '''
        Compute the contrastive divergence loss, reconstruction error and gradient.

        Inputs
        ------
        sigma (torch.Tensor):
            Visible units.
        beta (float):
            Inverse temperature.
        number_sampling_steps (int):
            Number of sampling steps.
        monitor_sampling (bool):
            Whether to monitor the partial free entropy during sampling.
        calculate_loss (bool):
            Whether to calculate the partial free entropy.
        Outputs
        -------
        loss (torch.Tensor):
            Average partial free entropy difference of the RBM.
        reconstruction_error (torch.Tensor):
            Reconstruction error.
        gradient (torch.Tensor):
            Contrastive divergence gradient.
        '''
        if monitor_sampling:
            number_monitored_sampling_steps = number_sampling_steps
        else:
            number_monitored_sampling_steps = 0
        
        tau = torch.zeros((len(sigma), self.P), device = self.training_device)
        
        P_tau_given_sigma = self.sample_hidden_given_visible(tau, sigma, beta)
        
        positive_gradient = torch.mean(torch.reshape(sigma, (-1, self.N, 1)) @ torch.reshape(tau, (-1, 1, self.P)), dim = 0)
        
        _, loss = self.sample_visible_and_hidden(sigma, tau, beta, number_sampling_steps,
                                                 number_monitored_sampling_steps = number_monitored_sampling_steps,
                                                 calculate_loss = calculate_loss)
        
        negative_gradient = torch.mean(torch.reshape(sigma, (-1, self.N, 1)) @ torch.reshape(tau, (-1, 1, self.P)), dim = 0)
        
        del tau
        
        gradient = positive_gradient - negative_gradient
        
        reconstruction_error = torch.sum(gradient**2)
        
        gradient = beta*gradient
        
        return loss, reconstruction_error, gradient
    
    def ord_update(self, sigma, beta : float, alpha : float, learning_rate : float,
                   weight_decay : float, momentum : float, number_sampling_steps : int,
                   monitor_sampling : bool, calculate_loss : bool):
        '''
        Update the weights of the RBM using underdamped stochastic Langevin dynamics of the contrastive divergence gradient.
        Inputs
        ------
        sigma (torch.Tensor):
            Visible units.
        beta (float):
            Inverse temperature.
        alpha (float):
            Load of examples in the training set, i.e. num_training_examples/num_visible_units.
        learning_rate (float):
            Learning rate.
        weight_decay (float):
            L2 regularization parameter.
        momentum (float):
            Momentum parameter.
        number_sampling_steps (int):
            Number of sampling steps used to compute the contrastive divergence gradient.
        monitor_sampling (bool):
            Whether to monitor the partial free entropy during sampling.
        calculate_loss (bool):
            Whether to calculate the partial free entropy.
        Outputs
        -------
        loss (torch.Tensor):
            Average partial free entropy difference of the RBM.
        reconstruction_error (torch.Tensor):
            Reconstruction error.
        '''
        loss, reconstruction_error, gradient = self.contrastive_divergence(sigma, beta, number_sampling_steps,
                                                                           monitor_sampling = monitor_sampling,
                                                                           calculate_loss = calculate_loss)
        
        noise = torch.randn((self.N, self.P), device = self.training_device, generator = self.random_number_generator)
        
        with torch.no_grad():
            gradient = alpha * gradient - weight_decay * self.xi
            
            self.velocity.copy_(momentum * self.velocity + learning_rate * gradient + ((1 - momentum**2)/self.N)**(1/2) * noise)
            
            self.xi.copy_(self.xi + learning_rate * self.velocity)
            
        return loss, reconstruction_error
    
    def langevin_training(self, loader : torch.utils.data.DataLoader, beta : float, alpha : float, initial_learning_rate : float,
                          learning_rate_decay : float, weight_decay : float, momentum : float, number_sampling_steps : int,
                          number_training_epochs : int, number_monitored_training_epochs : int, monitor_sampling : bool,
                          number_burn_in_epochs : int, number_magnetization_samples : int):
        '''
        Train the RBM using underdamped stochastic Langevin dynamics.
        Run the ord_update method for a number of training epochs.

        Inputs
        ------
        loader (torch.utils.data.DataLoader):
            Data loader for the training set.
        beta (float):
            Inverse temperature.
        alpha (float):
            Load of examples in the training set, i.e. num_training_examples/num_visible_units.
        initial_learning_rate (float):
            Initial learning rate.
        learning_rate_decay (float):
            Learning rate decay.
        weight_decay (float):
            L2 regularization parameter.
        momentum (float):
            Momentum parameter.
        number_sampling_steps (int):
            Number of sampling steps used to compute the contrastive divergence gradient.
        number_training_epochs (int):
            Number of training epochs.
        number_monitored_training_epochs (int):
            Number of training epochs where the partial free entropy, reconstruction error and magnetization are monitored.
        monitor_sampling (bool):
            Whether to monitor the partial free entropy during sampling.
        number_burn_in_epochs (int):
            Number of burn-in epochs before recording the magnetization between this RBM and the teacher.
        number_magnetization_samples (int):
            Number of magnetization samples to record.
        Outputs
        -------
        m_list (list):
            List of magnetization samples.
        '''
        m_list = []
        
        if self.teacher is None:
            number_magnetization_samples = 0
        
        for training_epoch in range(1, number_training_epochs + 1):
            if number_monitored_training_epochs != 0:
                monitor_training_this_epoch = training_epoch % (number_training_epochs // number_monitored_training_epochs) == 0
                monitor_training_this_epoch = monitor_training_this_epoch or (training_epoch == 0)
            else:
                monitor_training_this_epoch = False
            
            if number_magnetization_samples != 0:
                record_magnetization_this_epoch = training_epoch % ((number_training_epochs - number_burn_in_epochs) // number_magnetization_samples) == 0
            else:
                record_magnetization_this_epoch = False
            
            learning_rate = initial_learning_rate * (1 + learning_rate_decay * training_epoch)**(-1)
            
            average_loss = torch.tensor(0.)
            average_reconstruction_error = torch.tensor(0.)
            
            for batch, sigma_batch in enumerate(loader):
                if number_monitored_training_epochs != 0:
                    monitor_sampling_this_epoch = monitor_training_this_epoch & monitor_sampling
                else:
                    monitor_sampling_this_epoch = False
                
                sigma_batch = sigma_batch.view(-1, self.N).to(self.training_device)
                
                loss, reconstruction_error = self.ord_update(sigma_batch, beta, alpha, learning_rate, weight_decay, momentum, number_sampling_steps,
                                                             monitor_sampling = monitor_sampling_this_epoch, calculate_loss = monitor_training_this_epoch)
                
                if monitor_training_this_epoch:
                    average_loss += (loss.detach().item() - average_loss) / (batch + 1)
                    average_reconstruction_error += (reconstruction_error.detach().item() - average_reconstruction_error) / (batch + 1)
            
            if record_magnetization_this_epoch:
                m = torch.transpose(self.teacher.xi, 0, 1) @ self.xi
                m_list.append(torch.abs(m).detach().tolist())
            
            if monitor_training_this_epoch:
                if self.teacher is not None:
                    m = torch.transpose(self.teacher.xi, 0, 1) @ self.xi
                    
                    print("Epoch [{}/{}], loss: {:.4f}, reconstruction error: {:.4f}, Magnetization sample: {:.4f}"
                          .format(training_epoch, number_training_epochs, average_loss, average_reconstruction_error, torch.max(torch.abs(m)).detach()))
                else:
                    print("Epoch [{}/{}], loss: {:.4f}, reconstruction error: {:.4f}"
                          .format(training_epoch, number_training_epochs, average_loss, average_reconstruction_error))
            
        return m_list
    
    def metropolis_training(self, sigma, field, beta : float, number_training_epochs : int, N_batch_size : int,
                            number_monitored_training_epochs : int, number_burn_in_epochs : int,
                            number_magnetization_samples : int, anneal : bool):
        '''
        Train the RBM using the Metropolis algorithm.
        Accept/reject weight spin flips based on the difference in log-likelihood between states.
        Inputs
        ------
        sigma (torch.Tensor):
            Visible units.
        field (torch.Tensor):
            External field biasing the RBM towards certain weights.
        beta (float):
            Inverse temperature.
        number_training_epochs (int):
            Number of training epochs.
        N_batch_size (int):
            Size of visible unit batch.
        number_monitored_training_epochs (int):
            Number of training epochs where the partial free entropy, reconstruction error and magnetization are monitored.
        number_burn_in_epochs (int):
            Number of burn-in epochs before recording the magnetization between this RBM and the teacher.
        number_magnetization_samples (int):
            Number of magnetization samples to record.
        anneal (bool):
            Whether to anneal the inverse temperature beta.
        Outputs
        -------
        m_list (list):
            List of magnetization samples.
        '''
        average_d_log_L = torch.tensor(0.)
        average_acceptance_rate = torch.tensor(0.)
        batch = 0
        
        m_list = []
        
        if self.teacher is None:
            number_magnetization_samples = 0
        
        for training_epoch in range(1, number_training_epochs + 1):
            if number_monitored_training_epochs != 0:
                monitor_training_this_epoch = training_epoch % (number_training_epochs // number_monitored_training_epochs) == 0
                monitor_training_this_epoch = monitor_training_this_epoch or (training_epoch == 0)
            else:
                monitor_training_this_epoch = False
            
            if number_magnetization_samples != 0:
                record_magnetization_this_epoch = training_epoch % ((number_training_epochs - number_burn_in_epochs) // number_magnetization_samples) == 0
            else:
                record_magnetization_this_epoch = False
            
            if anneal:
                beta_cur = training_epoch/number_training_epochs * beta # + (1 - training_epoch/number_training_epochs) * beta/2
            else:
                beta_cur = beta
            
            N_batch = torch.randint(0, self.N, size = (N_batch_size,), generator = self.random_batch_generator).to(self.training_device)
            
            P_batch = torch.randint(0, self.P, size = (1,), generator = self.random_batch_generator).to(self.training_device)
            
            d_log_L = self.log_likelihood_difference(N_batch, P_batch, sigma, beta_cur)
            
            # Bias to break the symmetry
            d_log_L = d_log_L - 2 * len(sigma) * field[ix_(N_batch, P_batch)] * self.xi[ix_(N_batch, P_batch)]
            
            random_numbers = -torch.log(torch.rand((len(N_batch), len(P_batch)), generator = self.random_batch_generator).to(self.training_device))
            
            accept_spins = (random_numbers > -d_log_L).to(torch.float32)
            
            self.xi[ix_(N_batch, P_batch)] *= 1 - 2*accept_spins
            
            d_log_L = torch.mean(d_log_L).detach().item()
            acceptance_rate = torch.mean(accept_spins).detach().item()
            
            average_d_log_L += (d_log_L - average_d_log_L) / (batch + 1)
            average_acceptance_rate += (acceptance_rate - average_acceptance_rate) / (batch + 1)
            batch += 1
            
            if record_magnetization_this_epoch:
                m = torch.transpose(self.teacher.xi, 0, 1) @ self.xi / self.N
                m_list.append(torch.abs(m).detach().tolist())
            
            if monitor_training_this_epoch:
                if self.teacher is not None:
                    m = torch.transpose(self.teacher.xi, 0, 1) @ self.xi / self.N
                
                    print("Epoch [{}/{}], log-likelihood difference: {:.4f}, acceptance rate: {:.4f}, magnetization sample: {:.4f}"
                          .format(training_epoch, number_training_epochs, average_d_log_L, average_acceptance_rate, torch.max(torch.abs(m)).detach()))
                else:
                    print("Epoch [{}/{}], log-likelihood difference: {:.4f}, acceptance rate: {:.4f}"
                          .format(training_epoch, number_training_epochs, average_d_log_L, average_acceptance_rate))
                
                average_d_log_L = torch.tensor(0.)
                average_acceptance_rate = torch.tensor(0.)
                batch = 0
            
        return m_list