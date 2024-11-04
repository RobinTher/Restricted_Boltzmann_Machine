import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.integrate import simpson

# import matplotlib.pyplot as plt
# from matplotlib import colors, ticker
# import cmasher as cmr
# import re
# %matplotlib inline

# import time

machine_eps = np.finfo("float32").eps

def integral(y, x):
    '''
    Definite integral of y with respect to x.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html.
    Used as a helper function.
    '''
    return simpson(y, x, axis = 0)

def real_mean(w):
    
    return jnp.mean(jnp.real(w), axis = 1, keepdims = True)

def random_mat_cor(mat_cor_cor, P, n_beta, n_c, key):
    '''
    Generate n_beta x n_c random P x P correlation matrices
    from the projected Wishart distribution defined in the paper.
    Refer to the paper for a full reference.

    Inputs
    ------
    mat_cor_cor (jnp.ndarray):
        Correlation matrices of the noise in the correlation matrices.
    P (int):
        Size of the correlation matrix.
    n_beta (int):
        Number of matrices along beta axis.
    n_c (int):
        Number of matrices along c axis.
    key (jax.random.PRNGKey):
        Random key for generating samples.
    
    Outputs
    -------
    mat_cor (jnp.ndarray):
        Correlation matrices.
    '''
    # Generate random symmetric positive definite matrices from the Wishart distribution with covariance matrix mat_cor_cor
    A = jax.random.multivariate_normal(key, mean = jnp.zeros(P), cov = mat_cor_cor, shape = (n_beta, n_c, P))
    B = A @ jnp.transpose(A, axes = (0, 1, 3, 2))
    
    # Remove spurious negative values
    B = jnp.abs(B)
    
    # Normalize the diagonal elements to 1, i.e. find the corresponding correlation matrices
    D = jnp.diagonal(B, axis1 = -2, axis2 = -1)[..., jnp.newaxis]
    mat_cor = jnp.transpose(D, axes = (0, 1, 3, 2))**(-1/2) * B * D**(-1/2)
    
    return mat_cor

def mat_equi_cor(c, P):
    '''
    Generate an equicorrelation matrix (uniform correlation matrix in the paper).
    An equicorrelation matrix has diagonal entries equal to 1 and off-diagonal entries equal to a given number c.
    
    Inputs
    ------
    c (float):
        The correlation coefficient between the off-diagonal entries of the matrix.
    P (int):
        The size of the matrix.
    
    Outputs
    -------
    mat_cor (numpy.ndarray):
        The equicorrelation matrix with shape (P, P).
    '''
    mat_cor = (1 - c) * jnp.eye(P, P) + c * jnp.ones((P, P))
    
    return mat_cor

def random_multivariate_binary(mat_cor, P, n_binary_samples, key):
    '''
    Generate binary random samples with mean 0 and a fixed covariance matrix mat_cor.
    This sampling method is commonly known as the arcsine law.
    Refer to the paper for a full reference.
       
    Inputs
    ------
    mat_cor (float array):
        Covariance matrix for random number generation.
    P (int):
        Size of the matrix.
    n_binary_samples (int):
        Number of random samples to generate.
    key (jax.random.PRNGKey):
        Random key for generating samples.
    
    Outputs
    -------
    spins_x (float array):
        Random samples generated.
    '''
    spins_x = random_white_normal(P, n_binary_samples, key)
    C = np.sin(np.pi/2 * mat_cor)
    L = jnp.linalg.cholesky(C)
    spins_x = jnp.sign(L @ spins_x.T).T
    
    # spins_x = jnp.sign(jax.random.multivariate_normal(key, mean = np.zeros(P), cov = np.sin(np.pi/2 * mat_cor), shape = (n_binary_samples,)))
    
    return spins_x

def init_spins(P, n_pre_appended_axes):
    '''
    Initialize spins for the Restricted Boltzmann Machine.
    The spins are generated as a Cartesian product of {-1, 1}^P.
    They are then reshaped by prepending P axes to facilitate matrix operations.
    
    Inputs
    ------
    P (int):
        Number of spins.
    n_pre_appended_axes (int):
        Number of axes to be prepended to the spins.
    Outputs
    -------
    spins_T (jnp.ndarray):
        Transposed spins array with shape (2**P, 1, ..., 1, P).
    spins (jnp.ndarray):
        Spins array with shape (2**P, 1, ..., P, 1).
    '''
    
    spins_base = jnp.full((P, 2), jnp.array([-1, 1]))
    
    spins = jnp.reshape(jnp.array(jnp.meshgrid(*spins_base)).T, newshape = (-1,) + n_pre_appended_axes * (1,) + (P, 1))
    
    spins_T = jnp.transpose(spins, axes = tuple(range(spins.ndim-2)) + (spins.ndim-1,) + (spins.ndim-2,))
    
    return spins_T, spins

def random_white_normal(n_normal_dims, n_normal_samples, key):
    '''
    Generate samples from a whitened normal distribution.
    Samples are whitened using Cholesky decomposition.

    Inputs
    ------
    n_normal_dims (int):
        Number of dimensions of the normal distribution.
    n_normal_samples (int):
        Number of samples to generate.
    key (jax.random.PRNGKey):
        Random key for generating samples.
    
    Outputs
    -------
    z (jnp.ndarray):
        Generated samples from the whitened normal distribution.
    '''

    z = jax.random.normal(key, shape = (n_normal_samples, n_normal_dims))
    
    # z = z - jnp.mean(z, axis = 0, keepdims = True)
    
    z = (z - jnp.flip(z, axis = 0))/jnp.sqrt(2)
    
    # z = z.at[n_normal_samples//2 :].set(jnp.flip(z[n_normal_samples//2 :], axis = 0))
    
    C = jnp.array(jnp.cov(z, rowvar = False), ndmin = 2)
    
    # eigvals, eigvecs = jnp.linalg.eigh(C)
    # eigvecs, eigvals, _ = jnp.linalg.svd(C, hermitian = True)
    
    # z = ((z @ eigvecs) * 1/jnp.sqrt(eigvals)) @ eigvecs.T
    
    L = jnp.linalg.cholesky(C)
    # z = jax.scipy.linalg.solve_triangular(L, z.T, lower = True).T
    z = jnp.linalg.solve(L, z.T).T
    
    return z

def hamiltonian_M(spins_T, spins, mat_cor, beta):
    '''
    Evaluate the effective Hamiltonian M (or M_s) for the given input parameters.
    '''
    return 1/2*beta**2 * (spins_T @ mat_cor @ spins)
    
def probability(H):
    '''
    Evaluate the Gibbs distribution of input input energy levels.
    '''
    C = jnp.max(H, axis = 0) # jnp.real(jnp.max(H, axis = 0))
    M = jnp.exp(H - C)
    
    prob = M / jnp.sum(M, axis = 0)
    
    return prob

def log_Z(H):
    '''
    Evaluate the log partition function of input energy levels.
    '''
    C = jnp.max(H, axis = 0) # jnp.real(jnp.max(H, axis = 0))
    
    f = C + jnp.log(jnp.sum(jnp.exp(H - C), axis = 0))
    
    return f

def max_abs(x, axis = None, keepdims = False):
    '''
    Max of the absolute value of x.
    '''
    return jnp.maximum(jnp.max(x, axis = axis, keepdims = keepdims), -jnp.min(x, axis = axis, keepdims = keepdims))

def do_nothing(message):
    return

def print_message(message):
    jax.debug.print("{message}", message = message)
    return
    
def keep_looping(t, tol, args):
    """
    Check if the loop should continue based on the given conditions.

    Inputs
    ------
    t (int):
        Max number of iterations.
    tol (float): Tolerance value. The loop stops when the distance between m_cur and m_prev is less than tol.
    args (tuple):
        Tuple containing the current and previous values of the order parameters,
        among which t_cur and m_cur and m_hat.
        t_cur (int):
            Current iteration number.
        m_cur (jnp.ndarray):
            Current value of the Mattis magnetization m.
        m_prev (jnp.ndarray):
            Previous value of the Mattis magnetization m.
    
    Outputs
    -------
    bool:
        True if the loop should continue, False otherwise.
    """ 
    
    t_cur, m_cur, s_cur, q_cur, m_prev, s_prev, q_prev, m_hat, s_hat, q_hat, key_update = args
    return (t_cur < t) & (jnp.sum((m_cur - m_prev)**2)**(1/2) >= tol)

class Iterator():
    '''
    Class for solving the saddle-point equations (Eqs. 3) via numerical iteration when the prior on the student patterns is the Rademacher distribution.
    The superscript (and subscript) * frequently used in the paper is replaced by _s in the code.
    The tilde frequently used in the paper is replaced by _t in the code.
    
    Inputs
    ------
    mat_cor (float array):
        P x P covariance matrix of the teacher patterns.
    P (int):
        Size of the hidden layer of the teacher.
    P_t (int):
        Size of the hidden layer of the student.
    n_normal_samples (int):
        Number of normally distributed random samples used to approximate each integral over a Gaussian variable z_{mu nu}.
    n_binary_samples (int):
        Number of binary samples used to approximate the probability distribution of the teacher patterns.
    key (jax.random.PRNGKey):
        Key for random number generation.
    tol (float):
        Precision of the iterator. Iteration stops when the distance between the current and previous values of the Mattis magnetization is less than tol.
    
    Attributes not defined in the inputs
    ------------------------------------
    spins_s_T (jnp.ndarray):
        Transposed spins array with shape (2**P, 1, 1, P).
    spins_s (jnp.ndarray):
        Spins array with shape (2**P, 1, P, 1).
    spins_T (jn.ndarray):
        Transposed spins array with shape (2**P_t, 1, 1, 1, P_t).
    spins (jnp.ndarray):
        Spins array with shape (2**P_t, 1, 1, P_t, 1).
    key_update (jax.random.PRNGKey):
        Random key used for permutation of self.z during the iteration.
    z (jnp.ndarray):
        Random samples from a whitened normal distribution with shape (n_normal_samples, P_t, P_t).
    p_x (jnp.ndarray):
        Probability distribution of the teacher patterns.
    
    Methods
    -------
    hamiltonian_L:
        Evaluate the effective Hamiltonian L for the given input parameters.
    free_entropy:
        Evaluate the free entropy for the given input parameters.
    update:
        Single iteration of the saddle-point equations.
        Update the conjugate order parameters first, then update the ordinary order parameters.
    iterate:
        Run update multiple times to solve the saddle-point equations.
    '''
    def __init__(self, mat_cor, P, P_t, n_normal_samples, n_binary_samples, key, tol):
        self.mat_cor = mat_cor
        self.P = P
        self.P_t = P_t
        
        self.spins_s_T, self.spins_s = init_spins(P, n_pre_appended_axes = 1)
        
        self.spins_T, self.spins = init_spins(P_t, n_pre_appended_axes = 2)
        
        key_normal, key_binary, self.key_update = jax.random.split(key, num = 3)
        
        z = random_white_normal(P_t**2, n_normal_samples, key_normal)
        self.z = jnp.reshape(z, newshape = (-1, P_t, P_t))
        
        spins_x = random_multivariate_binary(mat_cor, P, n_binary_samples, key_binary).reshape(1, -1, P, 1)
        
        self.p_x = jnp.mean(jnp.all(self.spins_s == spins_x, axis = 2, keepdims = True), axis = 1, keepdims = True)
        
        self.tol = tol
    
    @partial(jax.jit, static_argnums = 0)
    def hamiltonian_L(self, m, s, q, lambda_1, lambda_2, key_update):
        '''
        Calculate the effective Hamiltonian L for the given input parameters.

        Inputs
        ------
        m (jnp.ndarray):
            Mattis magnetization.
        s (jnp.ndarray):
            Overlap between patterns from the same sample.
        q (jnp.ndarray):
            Overlap between patterns from different samples.
        lambda_1 (float):
            Equal to either beta_s in L_O or 1 in L_C. See paper.
        lambda_2 (float):
            Equal to either beta in L_O or 1 in L_C. See paper.
        key_update (jnp.ndarray):
            Random key used for permutation of self.z.
        
        Outputs
        -------
        H_L (jnp.ndarray):
            The calculated Hamiltonian.
        '''
        
        perm = jax.random.permutation(key_update, self.P_t)
        z = self.z[:, perm[:, jnp.newaxis], perm]
        
        # z = self.z
        
        A_q_squared = 2*q - jnp.diag(jnp.sum(q, axis = 1))
        A_q = jnp.sqrt(jnp.abs(A_q_squared))
        A_q = jnp.where(A_q_squared >= 0, (1 + 0j) * A_q, (0 + 1j) * A_q)
        
        A_q_times_z = A_q * z
        
        H_L = lambda_2 * (jnp.sum(self.spins_T @ A_q_times_z, axis = -1, keepdims = True) + jnp.sum(A_q_times_z @ self.spins, axis = -2, keepdims = True))/2
        
        # q_off_diag = q.at[jnp.diag_indices_from(q)].set(0)
        
        # H_L = H_L + 1/2*lambda_2**2 * (self.spins_T @ (s - q_off_diag) @ self.spins)
        
        H_L = H_L + 1/2*lambda_2**2 * (self.spins_T @ (s - q) @ self.spins)
        
        H_L = H_L + lambda_1*lambda_2 * (self.spins_s_T @ m @ self.spins)
        
        return H_L
    
    @partial(jax.jit, static_argnums = 0)
    def free_entropy(self, m, s, q, m_hat, s_hat, q_hat, beta_s, beta, alpha, p_M_s, key_update):
        """
        Evaluate the free entropy for the given input parameters.

        Inputs
        ------
        m (jnp.ndarray):
            Mattis magnetization.
        s (jnp.ndarray):
            Overlap between patterns from the same sample.
        q (jnp.ndarray):
            Overlap between patterns from different samples.
        m_hat (jnp.ndarray):
            Conjugate order parameter of m.
        s_hat (jnp.ndarray):
            Conjugate order parameter of s.
        q_hat (jnp.ndarray):
            Conjugate order parameter of q.
        beta_s (float):
            Inverse temperature of the teacher.
        beta (float):
            Inverse temperature of the student.
        alpha (float):
            Normalized number of examples sent by the teacher to the student.
        p_M_s (jnp.ndarray):
            Probability distribution of the teacher hidden units.
        key_update (jnp.ndarray):
            Random key used for permutation of self.z.
        
        Outputs
        -------
        float:
            The calculated free entropy.
        """

        H_L_conjugate = self.hamiltonian_L(m_hat, s_hat, q_hat, 1, 1, key_update)
        F_L_conjugate = jnp.squeeze(jnp.sum(self.p_x * jnp.mean(log_Z(H_L_conjugate), axis = 1, keepdims = True), axis = 0))
        
        H_L_ordinary = self.hamiltonian_L(m, s, q, beta_s, beta, key_update)
        F_L_ordinary = jnp.squeeze(jnp.sum(p_M_s * jnp.mean(log_Z(H_L_ordinary), axis = 1, keepdims = True), axis = 0))
        
        H_M = hamiltonian_M(self.spins_T, self.spins, s, beta)
        F_M = jnp.squeeze(log_Z(H_M))
        
        F = -jnp.sum(m * m_hat) - 1/2 * jnp.sum(s * s_hat) + 1/2 * jnp.sum(q * q_hat)
        F = F + F_L_conjugate + alpha * F_L_ordinary - alpha * F_M
        
        return jnp.real(F)
        
    @partial(jax.jit, static_argnums = 0)
    def update(self, t_step, tau_step, beta_s, beta, alpha, p_M_s, args):
        '''
        Single iteration of the saddle-point equations.
        Update the conjugate order parameters first, then update the ordinary order parameters.
        
        Inputs
        ------
        beta_s (float):
            Inverse temperature of the teacher.
        beta (float):
            Inverse temperature of the student.
        alpha (float):
            Normalized number of examples sent by the teacher to the student.
        p_M_s (jnp.ndarray):
            Probability distribution of the teacher hidden units.
        args (tuple):
            Tuple containing the current and previous values of the order parameters.
        
        Outputs
        --------
        tuple:
            Updated values of the order parameters.
        '''
        t_cur, m_cur, s_cur, q_cur, m_prev, s_prev, q_prev, m_hat, s_hat, q_hat, key_update = args
        
        m_prev = m_cur
        s_prev = s_cur
        q_prev = q_cur
        
        key_update, key_ordinary, key_conjugate = jax.random.split(key_update, num = 3)
        
        H_L = self.hamiltonian_L(m_cur, s_cur, q_cur, beta_s, beta, key_ordinary)
        p_L = probability(H_L) # jnp.real(probability(H_L))
        
        # message = p_L[:, :, : 3]
        # jax.lax.cond(jnp.any(jnp.imag(message) != 0), print_message, do_nothing, message)
        # jax.debug.print("{p_L}", p_L = p_L[:, :, : 3])
        
        # jax.debug.print("{p_L}", p_L = jnp.max(jnp.sum(jnp.abs(p_L), axis = 0)))
        
        H_M = hamiltonian_M(self.spins_T, self.spins, s_cur, beta)
        p_M = probability(H_M)
        
        # m_grad = beta_s*beta*alpha * jnp.real(jnp.squeeze(jnp.sum(p_M_s * jnp.mean(self.spins_s * jnp.sum(p_L * self.spins_T, axis = 0), axis = 1, keepdims = True), axis = 0)))
        # s_grad = beta**2*alpha * jnp.real(jnp.squeeze(jnp.sum(p_M_s * jnp.mean(jnp.sum(p_L * self.spins_T * self.spins, axis = 0), axis = 1, keepdims = True), axis = 0) - jnp.sum(p_M * self.spins_T * self.spins, axis = 0)))
        # q_grad = beta**2*alpha * jnp.real(jnp.squeeze(jnp.sum(p_M_s * jnp.mean(jnp.sum(p_L * self.spins_T, axis = 0) * jnp.sum(p_L * self.spins, axis = 0), axis = 1, keepdims = True), axis = 0)))
        
        m_grad = beta_s*beta*alpha * jnp.squeeze(jnp.sum(p_M_s * real_mean(self.spins_s * jnp.sum(p_L * self.spins_T, axis = 0)), axis = 0))
        s_grad = beta**2*alpha * jnp.squeeze(jnp.sum(p_M_s * real_mean(jnp.sum(p_L * self.spins_T * self.spins, axis = 0)), axis = 0) - jnp.sum(p_M * self.spins_T * self.spins, axis = 0))
        q_grad = beta**2*alpha * jnp.squeeze(jnp.sum(p_M_s * real_mean(jnp.sum(p_L * self.spins_T, axis = 0) * jnp.sum(p_L * self.spins, axis = 0)), axis = 0))
        
        # q_grad = jnp.where((-beta**2*alpha <= q_grad) & (q_grad <= beta**2*alpha), q_grad, beta**2*alpha * jnp.sign(q_grad))
        
        m_hat = (1 - t_step) * m_hat + t_step * m_grad
        s_hat = (1 - t_step) * s_hat + t_step * s_grad
        q_hat = (1 - t_step) * q_hat + t_step * q_grad
        
        
        H_L = self.hamiltonian_L(m_hat, s_hat, q_hat, 1, 1, key_conjugate)
        p_L = probability(H_L) # jnp.real(probability(H_L))
        
        # m_grad = jnp.real(jnp.squeeze(jnp.sum(self.p_x * jnp.mean(self.spins_s * jnp.sum(p_L * self.spins_T, axis = 0), axis = 1, keepdims = True), axis = 0)))
        # s_grad = jnp.real(jnp.squeeze(jnp.sum(self.p_x * jnp.mean(jnp.sum(p_L * self.spins_T * self.spins, axis = 0), axis = 1, keepdims = True), axis = 0)))
        # q_grad = jnp.real(jnp.squeeze(jnp.sum(self.p_x * jnp.mean(jnp.sum(p_L * self.spins_T, axis = 0) * jnp.sum(p_L * self.spins, axis = 0), axis = 1, keepdims = True), axis = 0)))
        
        m_grad = jnp.squeeze(jnp.sum(self.p_x * real_mean(self.spins_s * jnp.sum(p_L * self.spins_T, axis = 0)), axis = 0))
        s_grad = jnp.squeeze(jnp.sum(self.p_x * real_mean(jnp.sum(p_L * self.spins_T * self.spins, axis = 0)), axis = 0))
        q_grad = jnp.squeeze(jnp.sum(self.p_x * real_mean(jnp.sum(p_L * self.spins_T, axis = 0) * jnp.sum(p_L * self.spins, axis = 0)), axis = 0))
        
        # q_grad = jnp.where((-1 <= q_grad) & (q_grad <= 1), q_grad, jnp.sign(q_grad))
        
        m_cur = (1 - tau_step) * m_cur + tau_step * m_grad
        s_cur = (1 - tau_step) * s_cur + tau_step * s_grad
        q_cur = (1 - tau_step) * q_cur + tau_step * q_grad
        
        
        t_cur = t_cur + 1
        
        return t_cur, m_cur, s_cur, q_cur, m_prev, s_prev, q_prev, m_hat, s_hat, q_hat, key_update
    
    @partial(jax.jit, static_argnums = (0, 1))
    def iterate(self, t, t_step, tau_step, beta_s, beta, alpha, m_init, s_init, q_init):
        '''
        Run update multiple times to solve the saddle-point equations.

        Inputs
        ------
        t (int)
            Max number of iterations.
        t_step (float):
            Step size of each iteration.
        beta_s (float):
            Inverse temperature of the teacher.
        beta (float):
            Inverse temperature of the student.
        alpha (float):
            Normalized number of examples sent by the teacher to the student.
        m_init (jnp.ndarray):
            Initial value of the Mattis magnetization m.
        s_init (jnp.ndarray):
            Initial value of the overlap s between patterns from the same sample.
        q_init (jnp.ndarray):
            Initial value of the overlap q between patterns from different samples.
        
        Outputs
        -------
        m_final (jnp.ndarray):
            Final value of m.
        s_final (jnp.ndarray):
            Final value of s.
        q_final (jnp.ndarray):
            Final value of q.
        p_M_s (jnp.ndarray):
            Probability distribution of the teacher hidden units.
        F (float):
            Free entropy of the final order parameters.
        '''
        H_M_s = hamiltonian_M(self.spins_s_T, self.spins_s, self.mat_cor, beta_s)
        p_M_s = probability(H_M_s)
        
        # Symmetrize q before the first iteration as explained in the text.
        q_init = (q_init + q_init.T)/2
        
        t_cur, m_cur, s_cur, q_cur, m_prev, s_prev, q_prev, m_hat, s_hat, q_hat, key_update = self.update(t_step, tau_step, beta_s, beta, alpha, p_M_s, (0, m_init, s_init, q_init, m_init, s_init, q_init, m_init, s_init, q_init, self.key_update))
        
        t_cur, m_final, s_final, q_final, m_prev, s_prev, q_prev, m_hat, s_hat, q_hat, key_update = jax.lax.while_loop(partial(keep_looping, t, self.tol), partial(self.update, t_step, tau_step, beta_s, beta, alpha, p_M_s), (t_cur, m_cur, s_cur, q_cur, m_prev, s_prev, q_prev, m_hat, s_hat, q_hat, key_update))
        
        # jax.debug.print("{t_cur}", t_cur = t_cur)
        
        key_update, key_ordinary, key_conjugate = jax.random.split(key_update, num = 3)
        
        H_L = self.hamiltonian_L(m_hat, s_hat, q_hat, 1, 1, key_conjugate)
        p_L = probability(H_L)
        
        # m_res = m_final - jnp.real(jnp.squeeze(jnp.sum(self.p_x * jnp.mean(self.spins_s * jnp.sum(p_L * self.spins_T, axis = 0), axis = 1, keepdims = True), axis = 0)))
        # s_res = s_final - jnp.real(jnp.squeeze(jnp.sum(self.p_x * jnp.mean(jnp.sum(p_L * self.spins_T * self.spins, axis = 0), axis = 1, keepdims = True), axis = 0)))
        # q_res = q_final - jnp.real(jnp.squeeze(jnp.sum(self.p_x * jnp.mean(jnp.sum(p_L * self.spins_T, axis = 0) * jnp.sum(p_L * self.spins, axis = 0), axis = 1, keepdims = True), axis = 0)))
        
        m_res = m_final - jnp.squeeze(jnp.sum(self.p_x * real_mean(self.spins_s * jnp.sum(p_L * self.spins_T, axis = 0)), axis = 0))
        s_res = s_final - jnp.squeeze(jnp.sum(self.p_x * real_mean(jnp.sum(p_L * self.spins_T * self.spins, axis = 0)), axis = 0))
        q_res = q_final - jnp.squeeze(jnp.sum(self.p_x * real_mean(jnp.sum(p_L * self.spins_T, axis = 0) * jnp.sum(p_L * self.spins, axis = 0)), axis = 0))

        jax.debug.print("Max residuals of m, s and q, respectively:")
        jax.debug.print("{m_res_max}", m_res_max = max_abs(m_res))
        jax.debug.print("{s_res_max}", s_res_max = max_abs(s_res))
        jax.debug.print("{q_res_max}", q_res_max = max_abs(q_res))
        
        H_L = self.hamiltonian_L(m_prev, s_prev, q_prev, beta_s, beta, key_ordinary)
        p_L = probability(H_L) # jnp.real(probability(H_L))
        
        H_M = hamiltonian_M(self.spins_T, self.spins, s_prev, beta)
        p_M = probability(H_M)
        
        # m_res = m_hat - beta_s*beta*alpha * jnp.real(jnp.squeeze(jnp.sum(p_M_s * jnp.mean(self.spins_s * jnp.sum(p_L * self.spins_T, axis = 0), axis = 1, keepdims = True), axis = 0)))
        # s_res = s_hat - beta**2*alpha * jnp.real(jnp.squeeze(jnp.sum(p_M_s * jnp.mean(jnp.sum(p_L * self.spins_T * self.spins, axis = 0), axis = 1, keepdims = True), axis = 0) - jnp.sum(p_M * self.spins_T * self.spins, axis = 0)))
        # q_res = q_hat - beta**2*alpha * jnp.real(jnp.squeeze(jnp.sum(p_M_s * jnp.mean(jnp.sum(p_L * self.spins_T, axis = 0) * jnp.sum(p_L * self.spins, axis = 0), axis = 1, keepdims = True), axis = 0)))
        
        m_res = m_hat - beta_s*beta*alpha * jnp.squeeze(jnp.sum(p_M_s * real_mean(self.spins_s * jnp.sum(p_L * self.spins_T, axis = 0)), axis = 0))
        s_res = s_hat - beta**2*alpha * jnp.squeeze(jnp.sum(p_M_s * real_mean(jnp.sum(p_L * self.spins_T * self.spins, axis = 0)), axis = 0) - jnp.sum(p_M * self.spins_T * self.spins, axis = 0))
        q_res = q_hat - beta**2*alpha * jnp.squeeze(jnp.sum(p_M_s * real_mean(jnp.sum(p_L * self.spins_T, axis = 0) * jnp.sum(p_L * self.spins, axis = 0)), axis = 0))
        
        jax.debug.print("Max residuals of m_hat, s_hat and q_hat, respectively:")
        jax.debug.print("{m_res_max}", m_res_max = max_abs(m_res))
        jax.debug.print("{s_res_max}", s_res_max = max_abs(s_res))
        jax.debug.print("{q_res_max}", q_res_max = max_abs(q_res))
        
        F = self.free_entropy(m_final, s_final, q_final, m_hat, s_hat, q_hat, beta_s, beta, alpha, p_M_s, key_update)
        
        return m_final, s_final, q_final, p_M_s, F

class NormalIterator():
    '''
    Class for solving the saddle-point equations (Eqs. 3) via numerical iteration when the prior on the student patterns is the standard normal distribution.
    The superscript (and subscript) * frequently used in the paper is replaced by _s in the code.
    The tilde frequently used in the paper is replaced by _t in the code.
    
    Inputs
    ------
    mat_cor (float array):
        P x P covariance matrix of the teacher patterns.
    P (int):
        Size of the hidden layer of the teacher.
    P_t (int):
        Size of the hidden layer of the student.
    n_normal_samples (int):
        Number of normally distributed random samples used to approximate each integral over a Gaussian variable z_{mu nu}.
    n_binary_samples (int):
        Number of binary samples used to approximate the probability distribution of the teacher patterns.
    key (jax.random.PRNGKey):
        Key for random number generation.
    tol (float):
        Precision of the iterator. Iteration stops when the distance between the current and previous values of the Mattis magnetization is less than tol.
    
    Attributes not defined in the inputs
    ------------------------------------
    spins_s_T (jnp.ndarray):
        Transposed spins array with shape (2**P, 1, 1, P).
    spins_s (jnp.ndarray):
        Spins array with shape (2**P, 1, P, 1).
    spins_T (jn.ndarray):
        Transposed spins array with shape (2**P_t, 1, 1, 1, P_t).
    spins (jnp.ndarray):
        Spins array with shape (2**P_t, 1, 1, P_t, 1).
    key_update (jax.random.PRNGKey):
        Random key used for permutation of self.z during the iteration.
    z (jnp.ndarray):
        Random samples from a whitened normal distribution with shape (n_normal_samples, P_t, P_t).
    
    Methods
    -------
    hamiltonian_L:
        Evaluate the effective Hamiltonian L for the given input parameters.
    update:
        Single iteration of the saddle-point equations.
        Update the conjugate order parameters first, then update the ordinary order parameters.
    iterate:
        Run update multiple times to solve the saddle-point equations.
    '''
    def __init__(self, mat_cor, P, P_t, n_normal_samples, n_binary_samples, key, tol):
        self.mat_cor = mat_cor
        self.P = P
        self.P_t = P_t
        
        self.spins_s_T, self.spins_s = init_spins(P, n_pre_appended_axes = 1)
        
        self.spins_T, self.spins = init_spins(P_t, n_pre_appended_axes = 2)
        
        key_normal, key_binary, self.key_update = jax.random.split(key, num = 3)
        
        z = random_white_normal(P_t**2, n_normal_samples, key_normal)
        self.z = jnp.reshape(z, newshape = (-1, P_t, P_t))
        
        self.teacher_mat_cor = mat_cor
        
        self.tol = tol
    
    @partial(jax.jit, static_argnums = 0)
    def hamiltonian_L(self, m, s, q, lambda_1, lambda_2, key_update):
        '''
        Calculate the effective Hamiltonian L for the given input parameters.

        Inputs
        ------
        m (jnp.ndarray):
            Mattis magnetization.
        s (jnp.ndarray):
            Overlap between patterns from the same sample.
        q (jnp.ndarray):
            Overlap between patterns from different samples.
        lambda_1 (float):
            Equal to either beta_s in L_O or 1 in L_C. See paper.
        lambda_2 (float):
            Equal to either beta in L_O or 1 in L_C. See paper.
        key_update (jnp.ndarray):
            Random key used for permutation of self.z.
        
        Outputs
        -------
        H_L (jnp.ndarray):
            The calculated Hamiltonian.
        '''
        
        perm = jax.random.permutation(key_update, self.P_t)
        z = self.z[:, perm[:, jnp.newaxis], perm]
        
        # z = self.z
        
        A_q_squared = 2*q - jnp.diag(jnp.sum(q, axis = 1))
        A_q = jnp.sqrt(jnp.abs(A_q_squared))
        A_q = jnp.where(A_q_squared >= 0, (1 + 0j) * A_q, (0 + 1j) * A_q)
        
        A_q_times_z = A_q * z
        
        H_L = lambda_2 * (jnp.sum(self.spins_T @ A_q_times_z, axis = -1, keepdims = True) + jnp.sum(A_q_times_z @ self.spins, axis = -2, keepdims = True))/2
        
        # q_off_diag = q.at[jnp.diag_indices_from(q)].set(0)
        
        H_L = H_L + 1/2*lambda_2**2 * (self.spins_T @ (s - q) @ self.spins)
        
        H_L = H_L + lambda_1*lambda_2 * (self.spins_s_T @ m @ self.spins)
        
        return H_L
    
    @partial(jax.jit, static_argnums = 0)
    def update(self, t_step, tau_step, beta_s, beta, alpha, p_M_s, args):
        '''
        Single iteration of the saddle-point equations.
        Update the conjugate order parameters first, then update the ordinary order parameters.
        
        Inputs
        ------
        beta_s (float):
            Inverse temperature of the teacher.
        beta (float):
            Inverse temperature of the student.
        alpha (float):
            Normalized number of examples sent by the teacher to the student.
        p_M_s (jnp.ndarray):
            Probability distribution of the teacher hidden units.
        args (tuple):
            Tuple containing the current and previous values of the order parameters.
        
        Outputs
        --------
        tuple:
            Updated values of the order parameters.
        '''
        t_cur, m_cur, s_cur, q_cur, m_prev, s_prev, q_prev, m_hat, s_hat, q_hat, key_update = args
        
        m_prev = m_cur
        s_prev = s_cur
        q_prev = q_cur
        
        key_update, key_ordinary, key_conjugate = jax.random.split(key_update, num = 3)
        
        H_L = self.hamiltonian_L(m_cur, s_cur, q_cur, beta_s, beta, key_ordinary)
        p_L = jnp.real(probability(H_L))
        
        H_M = hamiltonian_M(self.spins_T, self.spins, s_cur, beta)
        p_M = probability(H_M)
        
        m_grad = beta_s*beta*alpha * jnp.squeeze(jnp.sum(p_M_s * real_mean(self.spins_s * jnp.sum(p_L * self.spins_T, axis = 0)), axis = 0))
        q_grad = beta**2*alpha * jnp.squeeze(jnp.sum(p_M_s * real_mean(jnp.sum(p_L * self.spins_T, axis = 0) * jnp.sum(p_L * self.spins, axis = 0)), axis = 0))
        s_grad = beta**2*alpha * jnp.squeeze(jnp.sum(p_M_s * real_mean(jnp.sum(p_L * self.spins_T * self.spins, axis = 0)), axis = 0) - jnp.sum(p_M * self.spins_T * self.spins, axis = 0))
        
        m_hat = (1 - t_step) * m_hat + t_step * m_grad
        q_hat = (1 - t_step) * q_hat + t_step * q_grad
        s_hat = (1 - t_step) * s_hat + t_step * s_grad
        
        
        student_mat_cor = jnp.linalg.inv(jnp.eye(self.P_t) + q_hat - s_hat)
        
        m_id = m_hat @ student_mat_cor
        
        m_grad = self.teacher_mat_cor @ m_id
        q_grad = m_id.T @ m_grad + student_mat_cor @ q_hat @ student_mat_cor
        s_grad = q_grad + student_mat_cor
        
        m_cur = (1 - tau_step) * m_cur + tau_step * m_grad
        q_cur = (1 - tau_step) * q_cur + tau_step * q_grad
        s_cur = (1 - tau_step) * s_cur + tau_step * s_grad
        
        
        t_cur = t_cur + 1
        
        return t_cur, m_cur, s_cur, q_cur, m_prev, s_prev, q_prev, m_hat, s_hat, q_hat, key_update
    
    @partial(jax.jit, static_argnums = (0, 1))
    def iterate(self, t, t_step, tau_step, beta_s, beta, alpha, m_init, s_init, q_init):
        '''
        Run update multiple times to solve the saddle-point equations.

        Inputs
        ------
        t (int)
            Max number of iterations.
        t_step (float):
            Step size of each iteration.
        beta_s (float):
            Inverse temperature of the teacher.
        beta (float):
            Inverse temperature of the student.
        alpha (float):
            Normalized number of examples sent by the teacher to the student.
        m_init (jnp.ndarray):
            Initial value of the Mattis magnetization m.
        s_init (jnp.ndarray):
            Initial value of the overlap s between patterns from the same sample.
        q_init (jnp.ndarray):
            Initial value of the overlap q between patterns from different samples.
        
        Outputs
        -------
        m_final (jnp.ndarray):
            Final value of m.
        s_final (jnp.ndarray):
            Final value of s.
        q_final (jnp.ndarray):
            Final value of q.
        p_M_s (jnp.ndarray):
            Probability distribution of the teacher hidden units.
        '''
        H_M_s = hamiltonian_M(self.spins_s_T, self.spins_s, self.mat_cor, beta_s)
        p_M_s = probability(H_M_s)
        
        # Symmetrize q before the first iteration as explained in the text.
        q_init = (q_init + q_init.T)/2
        
        # start = time.time()
        
        # jax.debug.print("End.")
        
        t_cur, m_cur, s_cur, q_cur, m_prev, s_prev, q_prev, m_hat, s_hat, q_hat, key_update = self.update(t_step, tau_step, beta_s, beta, alpha, p_M_s, (0, m_init, s_init, q_init, m_init, s_init, q_init, m_init, s_init, q_init, self.key_update))
        
        t_cur, m_final, s_final, q_final, m_prev, s_prev, q_prev, m_hat, s_hat, q_hat, key_update = jax.lax.while_loop(partial(keep_looping, t, self.tol), partial(self.update, t_step, tau_step, beta_s, beta, alpha, p_M_s), (t_cur, m_cur, s_cur, q_cur, m_prev, s_prev, q_prev, m_hat, s_hat, q_hat, key_update))
        
        # jax.debug.print("Begin:")
        
        # jax.debug.print("{q_cur}", q_cur = q_cur)
        
        # end = time.time()
        # print("Timing:")
        # print(end - start)
        
        return m_final, s_final, q_final, p_M_s

class SimpleIterator():
    '''
    Class for solving the reduced saddle-point equations (Eqs. 8) via numerical iteration.

    Inputs
    ------
    n_samples (int):
        Number samples used for Gaussian integration.
    n_stds (int):
        Number of standard deviations covered by the samples.
    
    Attributes
    ----------
    z (np.ndarray):
        Array of n_samples evenly-spaced samples between -n_stds and n_stds.
    
    Methods
    -------
    update:
        Single iteration of the saddle-point equations.
        Update the conjugate order parameters first, then update the ordinary order parameters.
    iterate:
        Run update multiple times to solve the saddle-point equations.
    '''
    def __init__(self, n_samples, n_stds = 5):
        self.z = np.linspace(-n_stds, n_stds, num = n_samples, endpoint = True)
    
    def update(self, beta, alpha, m, g):
        '''
        Single iteration of the saddle-point equations.
        Update the conjugate order parameters first, then update the ordinary order parameters.

        Inputs
        ------
        beta (float):
            Inverse temperature of the student.
        alpha (float):
            Normalized number of examples sent by the teacher to the student.
        m (float):
            Mattis magnetization.
        g (float):
            Overlap between student patterns from different samples that did not converge to a student pattern.
        
        Outputs
        -------
        m (float):
            Updated value of m.
        g (float):
            Updated value of g.
        '''
        m_hat = beta**2*alpha * integral(1/np.sqrt(2*np.pi) * np.exp(-self.z**2/2)
                                         * np.tanh(beta**2 * m + beta * np.sqrt(m) * self.z), self.z)
    
        m = integral(1/np.sqrt(2*np.pi) * np.exp(-self.z**2/2) * np.tanh(m_hat + np.sqrt(m_hat) * self.z), self.z)
        
        g_hat = beta**2*alpha * integral(1/np.sqrt(2*np.pi) * np.exp(-self.z**2/2)
                                         * np.tanh(beta * np.sqrt(g) * self.z)**2, self.z)
    
        g = integral(1/np.sqrt(2*np.pi) * np.exp(-self.z**2/2) * np.tanh(np.sqrt(g_hat) * self.z)**2, self.z)
        
        return m, g
    
    def iterate(self, t, beta, alpha, m, g):
        '''
        Run update multiple times to solve the saddle-point equations.

        Inputs
        ------
        beta (float):
            Inverse temperature of the student.
        alpha (float):
            Normalized number of examples sent by the teacher to the student.
        m (float):
            Mattis magnetization.
        g (float):
            Overlap between student patterns from different samples that did not converge to a student pattern.
        
        Outputs
        -------
        m (float):
            Final value of m.
        g (float):
            Final value of g.
        '''
        for _ in range(t):
            m, g = self.update(beta, alpha, m, g)
        
        return m, g

class SimpleNormalIterator():
    '''
    Class for solving the reduced normal saddle-point equations (Eqs. 8) via numerical iteration.

    Inputs
    ------
    n_samples (int):
        Number samples used for Gaussian integration.
    n_stds (int):
        Number of standard deviations covered by the samples.
    
    Attributes
    ----------
    z (np.ndarray):
        Array of n_samples evenly-spaced samples between -n_stds and n_stds.
    
    Methods
    -------
    update:
        Single iteration of the saddle-point equations.
        Update the conjugate order parameters first, then update the ordinary order parameters.
    iterate:
        Run update multiple times to solve the saddle-point equations.
    '''
    def __init__(self, n_samples, n_stds = 5):
        self.z = np.linspace(-n_stds, n_stds, num = n_samples, endpoint = True)
    
    def update(self, beta, alpha, m, g):
        '''
        Single iteration of the saddle-point equations.
        Update the conjugate order parameters first, then update the ordinary order parameters.

        Inputs
        ------
        beta (float):
            Inverse temperature of the student.
        alpha (float):
            Normalized number of examples sent by the teacher to the student.
        m (float):
            Mattis magnetization.
        g (float):
            Overlap between student patterns from different samples that did not converge to a student pattern.
        
        Outputs
        -------
        m (float):
            Updated value of m.
        g (float):
            Updated value of g.
        '''
        m_hat = beta**2*alpha * integral(1/np.sqrt(2*np.pi) * np.exp(-self.z**2/2)
                                         * np.tanh(beta**2 * m + beta * np.sqrt(m) * self.z), self.z)
    
        m = m_hat / (1 + m_hat)
        
        g_hat = beta**2*alpha * integral(1/np.sqrt(2*np.pi) * np.exp(-self.z**2/2)
                                         * np.tanh(beta * np.sqrt(g) * self.z)**2, self.z)
    
        g = g_hat / (1 + g_hat)**2
        
        return m, g
    
    def iterate(self, t, beta, alpha, m, g):
        '''
        Run update multiple times to solve the saddle-point equations.

        Inputs
        ------
        beta (float):
            Inverse temperature of the student.
        alpha (float):
            Normalized number of examples sent by the teacher to the student.
        m (float):
            Mattis magnetization.
        g (float):
            Overlap between student patterns from different samples that did not converge to a student pattern.
        
        Outputs
        -------
        m (float):
            Final value of m.
        g (float):
            Final value of g.
        '''
        for _ in range(t):
            m, g = self.update(beta, alpha, m, g)
        
        return m, g

def mat_cor_eigval(spins_s_T, spins_s, mat_cor_cor, beta_s_range, P, n_beta, n_c, key):
    '''
    First, use the random_mat_cor function to generate n_beta x n_c random P x P correlation matrices
    mat_cor from the projected Wishart distribution defined in the paper.

    Then, calculate the max eigenvalue of the matrix S defined in the paper as a function
    of beta_s and the off-diagonal of mat_cor_cor.

    Inputs
    ------
    spins_s_T (jnp.ndarray):
        Transposed spins array with shape (2**P, 1, 1, 1, P).
    spins_s (jnp.ndarray):
        Spins array with shape (2**P, 1, 1, P, 1).
    mat_cor_cor (jnp.ndarray):
        Correlation matrices of the noise in the correlation matrices.
    beta_s_range (jnp.ndarray):
        Array of inverse temperatures of the teacher.
    P (int):
        Size of the hidden layer of the teacher.
    n_beta (int):
        Number of teacher inverse temperatures.
    n_c (int):
        Number of different noise correlation matrices.
    key (jax.random.PRNGKey):
        Key for random number generation.
    
    Outputs
    -------
    eigval (jnp.ndarray):
        Max eigenvalue of the matrix S as a function
        of beta and the off-diagonal of mat_cor_cor.
    '''
    mat_cor = random_mat_cor(mat_cor_cor, P, n_beta, n_c, key)
    
    # mat_cor: (n_c, 1, P, P) 
    
    # spins_s_T : (2**P, 1, 1, 1, P)
    # spins_s : (2**P, 1, 1, P, 1)
    
    H_M_s = hamiltonian_M(spins_s_T, spins_s, mat_cor, beta_s_range)
    p_M_s = probability(H_M_s)
    hidden_cor = jnp.sum(p_M_s * spins_s_T * spins_s, axis = 0)
    
    eigval = jnp.max(jnp.linalg.eigh(hidden_cor @ mat_cor)[0], axis = -1)
    
    return eigval

def accumulate_inv_eigvals(spins_s_T, spins_s, mat_cor_cor, beta_s_range, P, n_beta, n_c, t_cur, args):
    '''
    Accumulate eigenvalues computed with mat_cor_eigval.
    '''
    mean_inv_eigval, key = args
    key_init, key = jax.random.split(key, num = 2)
    
    eigval = mat_cor_eigval(spins_s_T, spins_s, mat_cor_cor, beta_s_range, P, n_beta, n_c, key_init)
    inv_eigval = 1/eigval
    
    # Iterative formula for the mean
    mean_inv_eigval = mean_inv_eigval + (inv_eigval - mean_inv_eigval) / (t_cur + 1)
    
    return mean_inv_eigval, key