import numpy as np
import jax.numpy as jnp
import jax
from Saddle_point_iteration import Iterator, NormalIterator, mat_equi_cor

import matplotlib.pyplot as plt
from matplotlib import ticker

# [[0.2, 0., 0.2],
#  [0., 0.4, 0.]]

# vs

# [[0.4, 0.4, 0.2],
#  [0.4, 0.4, 0.2]]

tol = np.finfo("float32").eps

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

'''
beta = 1.

c = 0.4

m_0 = 0.2
epsilon = m_0*c

t = 10000
t_step = 0.1
seed = 4

n_alpha = 20
alpha_range = np.linspace(0.1, 1, num = n_alpha, endpoint = True)
'''

def block_PS_saddle_point_run(beta, alpha_range, c, P, m_0, epsilon, t, t_step, seed):
    '''
    Solve the binary saddle-point equations over a range of alpha
    with beta_s = beta and P_t = P + 1.
    Compute both block PS solutions in order to reproduce Figs. (10).
    Save the results in .npy files.
    Initialize the order parameters with m_0 and epsilon as shown in the paper.
    '''

    n_normal_samples = 800000
    n_binary_samples = 8000000

    beta_s = beta

    alpha = 0
    P_t = P + 1
    
    mat_cor = mat_equi_cor(c, P)
    
    key = jax.random.PRNGKey(seed)

    iterator = Iterator(mat_cor, P, P_t, n_normal_samples, n_binary_samples, key, tol)
    
    m_init = (m_0 - epsilon)*np.eye(P, P_t) + epsilon
    s_init = (1 - epsilon)*np.eye(P_t) + epsilon
    q_init = (m_0 - epsilon)*np.eye(P_t) + epsilon

    m_comp_range = np.zeros((n_alpha, 2))
    F_range = np.zeros(n_alpha)

    for j, alpha in enumerate(alpha_range):
        m, s, q, _, F = iterator.iterate(t, t_step, beta_s, beta, alpha, m_init, s_init, q_init)
        
        print(m)
        
        m_comp_range[j, 0] = np.mean(np.diagonal(m[:, : P]))
        m_comp_range[j, 1] = np.mean(m[:, P])

        F_range[j] = F

    with open("./Data/aligned_block_PS_magnetization_c=%.2f_P=%d_P_t=%d_beta=%.2f.npy" % (c, P, P_t, beta), "wb") as file:
        np.save(file, m_comp_range)
    
    with open("./Data/aligned_block_PS_free_entropy_c=%.2f_P=%d_P_t=%d_beta=%.2f.npy" % (c, P, P_t, beta), "wb") as file:
        np.save(file, F_range)

    m_init[0, P] = m_0

    m_comp_range = np.zeros((n_alpha, 2))
    F_range = np.zeros(n_alpha)
    
    for j, alpha in enumerate(alpha_range):
        m, s, q, _, F = iterator.iterate(t, t_step, beta_s, beta, alpha, m_init, s_init, q_init)
    
        print(m)
        
        m_comp_range[j, 0] = np.mean(np.diagonal(m[:, : P]))
        m_comp_range[j, 1] = np.mean(m[:, P])
        
        F_range[j] = F
    
    with open("./Data/anti_aligned_block_PS_magnetization_c=%.2f_P=%d_P_t=%d_beta=%.2f.npy" % (c, P, P_t, beta), "wb") as file:
        np.save(file, m_comp_range)

    with open("./Data/anti_aligned_block_PS_free_entropy_c=%.2f_P=%d_P_t=%d_beta=%.2f.npy" % (c, P, P_t, beta), "wb") as file:
        np.save(file, F_range)

beta = 1.
# beta = 1.2

P = 3
P_t = 3

c = 0.3

m_0 = 0.1
epsilon = m_0*c

t = 10000
t_step = 0.1
seed = 4

n_alpha = 20
alpha_range = np.linspace(0.11, 2.2, num = n_alpha, endpoint = True)

def saddle_point_run(beta, alpha_range, c, P, P_t, m_0, epsilon, t, t_step, seed):
    n_normal_samples = 800000
    n_binary_samples = 8000000

    beta_s = beta

    alpha = 0

    mat_cor = mat_equi_cor(c, P)
    
    key = jax.random.PRNGKey(seed)

    iterator = Iterator(mat_cor, P, P_t, n_normal_samples, n_binary_samples, key, tol)

    m_init = (m_0 - epsilon)*np.eye(P, P_t) + epsilon
    s_init = (1 - epsilon)*np.eye(P_t) + epsilon
    q_init = (m_0 - epsilon)*np.eye(P_t) + epsilon

    m_comp_range = np.zeros((n_alpha, 2))
    q_comp_range = np.zeros((n_alpha, 2))

    for j, alpha in enumerate(alpha_range):
        m, s, q, _, _ = iterator.iterate(t, t_step, beta_s, beta, alpha, m_init, s_init, q_init)
        
        # print("Magnetization m at j = %d and alpha = %.2f:" % (j, alpha))
        # print(m)
        # print("Overlap q at j = %d and alpha = %.2f:" % (j, alpha))
        # print(q)
        
        m_diag = jnp.diagonal(m)
        q_diag = jnp.diagonal(q)
        m_comp_range[j, 0] = np.mean(m_diag)
        q_comp_range[j, 0] = np.mean(q_diag)
        
        m_off_diag = m.at[jnp.diag_indices_from(m)].set(np.nan)
        q_off_diag = q.at[jnp.diag_indices_from(q)].set(np.nan)
        m_comp_range[j, 1] = np.nanmean(m_off_diag)
        q_comp_range[j, 1] = np.nanmean(q_off_diag)

    with open("./Data/magnetization_c=%.2f_P=%d_P_t=%d_beta=%.2f.npy" % (c, P, P_t, beta), "wb") as file:
        np.save(file, m_comp_range)

    with open("./Data/overlap_c=%.2f_P=%d_P_t=%d_beta=%.2f.npy" % (c, P, P_t, beta), "wb") as file:
        np.save(file, q_comp_range)

beta = 2.8

P = 4

c = 0.05

m_0 = 0.2
epsilon = m_0*c

t = 10000
t_step = 0.1

n_alpha = 20
alpha_range = np.linspace(0.1, 1, num = n_alpha, endpoint = True)

def normal_saddle_point_run(beta, alpha_range, c, P, m_0, epsilon, t, t_step, seed):
    '''
    Solve the normal saddle-point equations over a range of alpha
    with beta_s = beta and P_t = P + 1 in order to reproduce Fig. (11).
    Save the results in .npy files.
    Initialize the order parameters with m_0 and epsilon as shown in the paper.
    '''
    
    n_normal_samples = 800000
    n_binary_samples = 8000000

    beta_s = beta

    alpha = 0
    P_t = P + 1
    
    mat_cor = mat_equi_cor(c, P)
    
    key = jax.random.PRNGKey(seed)

    iterator = NormalIterator(mat_cor, P, P_t, n_normal_samples, n_binary_samples, key, tol)

    m_init = (m_0 - epsilon)*np.eye(P, P_t) + epsilon
    s_init = (1 - epsilon)*np.eye(P_t) + epsilon
    q_init = (m_0 - epsilon)*np.eye(P_t) + epsilon

    m_comp_range = np.zeros((n_alpha, 2))
    q_comp_range = np.zeros((n_alpha, 2))

    for j, alpha in enumerate(alpha_range):
        m, s, q, _ = iterator.iterate(t, t_step, beta_s, beta, alpha, m_init, s_init, q_init)
        
        m_diag = jnp.diagonal(m)
        q_diag = jnp.diagonal(q)
        m_comp_range[j, 0] = np.mean(m_diag)
        q_comp_range[j, 0] = np.mean(q_diag)
        
        m_off_diag = m.at[jnp.diag_indices_from(m)].set(np.nan)
        q_off_diag = q.at[jnp.diag_indices_from(q)].set(np.nan)
        m_comp_range[j, 1] = np.nanmean(m_off_diag)
        q_comp_range[j, 1] = np.nanmean(q_off_diag)

    with open("./Data/normal_magnetization_c=%.2f_P=%d_beta=%.2f.npy" % (c, P, beta), "wb") as file:
        np.save(file, m_comp_range)

    with open("./Data/normal_overlap_c=%.2f_P=%d_beta=%.2f.npy" % (c, P, beta), "wb") as file:
        np.save(file, q_comp_range)