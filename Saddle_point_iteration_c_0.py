import numpy as np
import jax
from Saddle_point_iteration import Iterator, SimpleIterator, NormalIterator, SimpleNormalIterator, mat_equi_cor

import matplotlib.pyplot as plt
from matplotlib import ticker

tol = np.finfo("float32").eps

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

'''
beta_s = 1.2
beta = 1.2

m_0 = 0.2
epsilon = 0

t = 100
t_step = 1/2
t_simple = 50
seed = 4

n_alpha = 20
alpha_range = np.linspace(0.1, 2, num = n_alpha, endpoint = True)
'''

def saddle_point_run(beta, alpha_range, P, m_0, epsilon, t, t_step, t_simple, seed):
    '''
    Solve the binary saddle-point equations over a range of alpha
    with beta_s = beta, P_t = P + 1 and c = 0.
    Compute both the PSB and partial PSB solutions in order to reproduce Figs. (1), (2), (3) and (4).
    Save the results in .npy files.
    Initialize the order parameters with m_0 and epsilon as shown in the paper.
    '''
    n_normal_samples = 8000000
    n_binary_samples = 80000000
    n_simple_samples = 401
    n_alpha = len(alpha_range)

    beta_s = beta

    alpha = 0
    P_t = P + 1
    c = 0
    
    mat_cor = mat_equi_cor(c, P)
    
    key = jax.random.PRNGKey(seed)

    iterator = Iterator(mat_cor, P, P_t, n_normal_samples, n_binary_samples, key, tol)
    
    simple_iterator = SimpleIterator(n_simple_samples)
    
    m_init = (m_0 - epsilon)*np.eye(P, P_t) + epsilon
    s_init = (1 - epsilon)*np.eye(P_t) + epsilon
    q_init = (m_0 - epsilon)*np.eye(P_t) + epsilon
    
    m_simp_init = m_0
    g_simp_init = m_0

    m_comp_range = np.zeros(n_alpha)
    g_comp_range = np.zeros(n_alpha)
    m_simp_range = np.zeros(n_alpha)
    g_simp_range = np.zeros(n_alpha)
    F_range = np.zeros(n_alpha)
    
    for j, alpha in enumerate(alpha_range):
        m, s, q, _, F = iterator.iterate(t, t_step, beta_s, beta, alpha, m_init, s_init, q_init)
        
        m_comp_range[j] = np.mean(np.diagonal(m))
        if P_t > P:
            g_comp_range[j] = np.mean(np.diagonal(q)[P : P_t])
        
        m, g = simple_iterator.iterate(t_simple, beta, alpha, m_simp_init, g_simp_init)
        m_simp_range[j] = np.squeeze(m)
        if P_t > P:
            g_simp_range[j] = np.squeeze(g)
        
        F_range[j] = F
    
    with open("./Data/PSB_magnetization_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, m_comp_range)
        np.save(file, m_simp_range)
    
    with open("./Data/PSB_overlap_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, g_comp_range)
        np.save(file, g_simp_range)

    with open("./Data/PSB_free_entropy_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, F_range)

    m_init[0, P] = m_0

    m_comp_range = np.zeros((n_alpha, 2))
    g_comp_range = np.zeros((n_alpha, 2))
    m_simp_range = np.zeros(n_alpha)
    g_simp_range = np.zeros(n_alpha)
    F_range = np.zeros(n_alpha)

    for j, alpha in enumerate(alpha_range):
        m, s, q, _, F = iterator.iterate(t, t_step, beta_s, beta, alpha, m_init, s_init, q_init)

        m_comp_range[j, 0] = np.mean(np.diagonal(m)[1 :])
        m_comp_range[j, 1] = (m[0, 0] + m[0, P])/2
        if P_t > P:
            g_comp_range[j, 0] = (q[0, 0] + q[P_t, P_t])/2
            g_comp_range[j, 1] = (q[0, P_t] + q[P_t, 0])/2

        m, g = simple_iterator.iterate(t_simple, beta, alpha, m_simp_init, g_simp_init)
        m_simp_range[j] = np.squeeze(m)
        if P_t > P:
            g_simp_range[j] = np.squeeze(g)

        F_range[j] = F

    with open("./Data/partial_PSB_magnetization_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, m_comp_range)
        np.save(file, m_simp_range)

    with open("./Data/partial_PSB_overlap_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, g_comp_range)
        np.save(file, g_simp_range)

    with open("./Data/partial_PSB_free_entropy_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, F_range)
    
'''
t = 1000
t_step = 0.1
'''

def normal_saddle_point_run(beta, alpha_range, P, m_0, epsilon, t, t_step, t_simple, seed):
    '''
    Solve the normal saddle-point equations over a range of alpha
    with beta_s = beta, P_t = P + 1 and c = 0.
    Compute both the PSB and partial PSB solutions in order to reproduce Figs. (4), (5), (6), (15) and (16).
    Save the results in .npy files.
    Initialize the order parameters with m_0 and epsilon as shown in the paper.
    '''
    n_normal_samples = 800000
    n_binary_samples = 8000000
    n_simple_samples = 401
    n_alpha = len(alpha_range)

    beta_s = beta

    alpha = 0
    P_t = P + 1
    c = 0
    
    mat_cor = mat_equi_cor(c, P)
    
    key = jax.random.PRNGKey(seed)

    iterator = NormalIterator(mat_cor, P, P_t, n_normal_samples, n_binary_samples, key, tol)

    simple_iterator = SimpleNormalIterator(n_simple_samples)

    m_init = (m_0 - epsilon)*np.eye(P, P_t) + epsilon
    s_init = (1 - epsilon)*np.eye(P_t) + epsilon
    q_init = (m_0 - epsilon)*np.eye(P_t) + epsilon

    m_simp_init = m_0
    g_simp_init = m_0

    m_comp_range = np.zeros(n_alpha)
    g_comp_range = np.zeros(n_alpha)
    m_simp_range = np.zeros(n_alpha)
    g_simp_range = np.zeros(n_alpha)

    for j, alpha in enumerate(alpha_range):
        m, s, q, _ = iterator.iterate(t, t_step, beta_s, beta, alpha, m_init, s_init, q_init)

        m_comp_range[j] = np.mean(np.diagonal(m))
        if P_t > P:
            g_comp_range[j] = np.mean(np.diagonal(q)[P : P_t])

        m, g = simple_iterator.iterate(t_simple, beta, alpha, m_simp_init, g_simp_init)
        m_simp_range[j] = np.squeeze(m)
        if P_t > P:
            g_simp_range[j] = np.squeeze(g)

    with open("./Data/PSB_normal_magnetization_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, m_comp_range)
        np.save(file, m_simp_range)

    with open("./Data/PSB_normal_overlap_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, g_comp_range)
        np.save(file, g_simp_range)
    
    m_init[0, P] = m_0

    m_comp_range = np.zeros((n_alpha, 2))
    g_comp_range = np.zeros((n_alpha, 2))
    m_simp_range = np.zeros(n_alpha)
    g_simp_range = np.zeros(n_alpha)

    for j, alpha in enumerate(alpha_range):
        m, s, q, _ = iterator.iterate(t, t_step, beta_s, beta, alpha, m_init, s_init, q_init)

        m_comp_range[j, 0] = np.mean(np.diagonal(m)[1 :])
        m_comp_range[j, 1] = (m[0, 0] + m[0, P])/2
        if P_t > P:
            g_comp_range[j, 0] = (q[0, 0] + q[P_t, P_t])/2 # np.mean(np.diagonal(q)[P : P_t])
            g_comp_range[j, 1] = (q[0, P_t] + q[P_t, 0])/2

        m, g = simple_iterator.iterate(t_simple, beta, alpha, m_simp_init, g_simp_init)
        m_simp_range[j] = np.squeeze(m)
        if P_t > P:
            g_simp_range[j] = np.squeeze(g)

    with open("./Data/partial_PSB_normal_magnetization_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, m_comp_range)
        np.save(file, m_simp_range)

    with open("./Data/partial_PSB_normal_overlap_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, g_comp_range)
        np.save(file, g_simp_range)

def plot_overlap(beta, alpha_range, P_range, name):
    '''
    Load the magnetization from .npy files written by the function saddle_point_run
    Compare the prediction of the full saddle-point equations
    to that of the reduced saddle-point equations in order to reproduce Figs. (1), (2), (15) and (16).
    '''
    fig, (m_main_axes, m_res_axes) = plt.subplots(nrows = 2, ncols = len(P_range), sharex = True, sharey = "row", figsize = (15, 6))
    m_fig_axis = fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = "none", which = "both", top = False, bottom = False, left = False, right = False)
    
    fig, (g_main_axes, g_res_axes) = plt.subplots(nrows = 2, ncols = len(P_range), sharex = True, sharey = "row", figsize = (15, 6))
    g_fig_axis = fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = "none", which = "both", top = False, bottom = False, left = False, right = False)
    
    fontsize = 13
    
    set_ylabel = True
    for P, m_main_axis, m_res_axis, g_main_axis, g_res_axis in zip(P_range, m_main_axes, m_res_axes, g_main_axes, g_res_axes):
        with open("./Data/%s_magnetization_P=%d_beta=%.2f.npy" % (name, P, beta), "rb") as file:
            m_comp_range = np.load(file)
            m_simp_range = np.load(file)
        
        with open("./Data/%s_overlap_P=%d_beta=%.2f.npy" % (name, P, beta), "rb") as file:
            g_comp_range = np.load(file)
            g_simp_range = np.load(file)
        
        m_main_axis.plot(alpha_range, m_simp_range, linestyle = "-", color = "C0")
        m_main_axis.plot(alpha_range, m_comp_range, linestyle = "--", linewidth = 3, color = "C1")
        m_main_axis.set_xlim(np.min(alpha_range), np.max(alpha_range))
        m_main_axis.tick_params(axis = "both", which = "major", labelsize = fontsize)
        m_main_axis.tick_params(axis = "both", which = "minor", labelsize = fontsize)
        m_main_axis.legend([r"Simplified equations", r"Full equations with" + "\n" + r"$P = %d$ hidden units" % P], fontsize = fontsize)
        if set_ylabel:
            m_main_axis.set_ylabel(r"Magnetization $m$", fontsize = fontsize)
        
        if m_comp_range.ndim == m_simp_range.ndim + 1:
            m_simp_range = m_simp_range[:, np.newaxis]
        
        m_res_axis.plot(alpha_range, np.zeros_like(alpha_range), linestyle = "-", color = "C0")
        m_res_axis.plot(alpha_range, m_comp_range - m_simp_range, linestyle = "--", linewidth = 3, color = "C1")
        m_res_axis.set_xlim(np.min(alpha_range), np.max(alpha_range))
        m_res_axis.tick_params(axis = "both", which = "major", labelsize = fontsize)
        m_res_axis.tick_params(axis = "both", which = "minor", labelsize = fontsize)
        m_res_axis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        if set_ylabel:
            m_res_axis.set_ylabel(r"Residuals", fontsize = fontsize)
        
        g_main_axis.plot(alpha_range, g_simp_range, linestyle = "-", color = "C0")
        g_main_axis.plot(alpha_range, g_comp_range, linestyle = "--", linewidth = 3, color = "C3")
        g_main_axis.set_xlim(np.min(alpha_range), np.max(alpha_range))
        g_main_axis.tick_params(axis = "both", which = "major", labelsize = fontsize)
        g_main_axis.tick_params(axis = "both", which = "minor", labelsize = fontsize)
        g_main_axis.legend([r"Simplified equations", r"Full equations with" + "\n" + r"$P = %d$ hidden units" % P], fontsize = fontsize)
        if set_ylabel:
            g_main_axis.set_ylabel(r"Overlap $g$", fontsize = fontsize)
        
        if g_comp_range.ndim == g_simp_range.ndim + 1:
            g_simp_range = g_simp_range[:, np.newaxis]
        
        g_res_axis.plot(alpha_range, np.zeros_like(alpha_range), linestyle = "-", color = "C0")
        g_res_axis.plot(alpha_range, g_comp_range - g_simp_range, linestyle = "--", linewidth = 3, color = "C3")
        g_res_axis.set_xlim(np.min(alpha_range), np.max(alpha_range))
        g_res_axis.tick_params(axis = "both", which = "major", labelsize = fontsize)
        g_res_axis.tick_params(axis = "both", which = "minor", labelsize = fontsize)
        g_res_axis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        if set_ylabel:
            g_res_axis.set_ylabel(r"Residuals", fontsize = fontsize)
        
        set_ylabel = False
    
    m_fig_axis.set_xlabel(r"Load $\alpha$", fontsize = fontsize)
    g_fig_axis.set_xlabel(r"Load $\alpha$", fontsize = fontsize)
    plt.show()

def plot_free_entropy_difference(beta, alpha_range, P):
    fontsize = 13
    
    with open("./Data/PSB_free_entropy_P=%d_beta=%.2f.npy" % (P, beta), "rb") as file:
        F_PBS_range = np.load(file)
    
    with open("./Data/partial_PSB_free_entropy_P=%d_beta=%.2f.npy" % (P, beta), "rb") as file:
        F_partial_PBS_range = np.load(file)
    
    # plt.plot(alpha_range, F_uncorrelated_range)
    # plt.plot(alpha_range, F_unstructured_range)
    plt.plot(alpha_range, F_PBS_range - F_partial_PBS_range)
    
    plt.xlabel(r"Load $\alpha$", fontsize = fontsize)
    plt.ylabel(r"Free entropy difference $f_{\mathrm{full \ PBS}} - f_{\mathrm{partial \ PBS}}$")
    
    plt.show()