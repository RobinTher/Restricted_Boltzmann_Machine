import numpy as np
import jax
from Saddle_point_iteration import Iterator, SimpleIterator, NormalIterator, SimpleNormalIterator, mat_equi_cor

import matplotlib.pyplot as plt
from matplotlib import ticker

tol = np.finfo("float32").eps

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def saddle_point_run(beta, alpha_range, P, m_0, epsilon, t, t_step, tau_step, t_simple, seed):
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
    q_comp_range = np.zeros((n_alpha, 2))
    m_simp_range = np.zeros(n_alpha)
    q_simp_range = np.zeros((n_alpha, 2))
    F_range = np.zeros(n_alpha)

    for j, alpha in enumerate(alpha_range):
        m, s, q, _, F = iterator.iterate(t, t_step, tau_step, beta_s, beta, alpha, m_init, s_init, q_init)

        m_comp_range[j] = np.mean(np.diagonal(m))
        if P_t > P:
            q_comp_range[j, 0] = np.mean(np.diagonal(q)[: P])
            q_comp_range[j, 1] = q[P, P]

        m, g = simple_iterator.iterate(t_simple, beta, alpha, m_simp_init, g_simp_init)
        m_simp_range[j] = np.squeeze(m)
        if P_t > P:
            q_simp_range[j, 0] = np.squeeze(m)
            q_simp_range[j, 1] = np.squeeze(g)
        
        F_range[j] = F
    
    with open("./Data/PSB_magnetization_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, m_comp_range)
        np.save(file, m_simp_range)
    
    with open("./Data/PSB_overlap_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, q_comp_range)
        np.save(file, q_simp_range)

    with open("./Data/PSB_free_entropy_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, F_range)

    m_init[0, P] = m_0

    m_comp_range = np.zeros((n_alpha, 2))
    q_comp_range = np.zeros((n_alpha, 3))
    m_simp_range = np.zeros(n_alpha)
    q_simp_range = np.zeros((n_alpha, 2))
    F_range = np.zeros(n_alpha)

    for j, alpha in enumerate(alpha_range):
        m, s, q, _, F = iterator.iterate(t, t_step, tau_step, beta_s, beta, alpha, m_init, s_init, q_init)

        m_comp_range[j, 0] = np.mean(np.diagonal(m)[1 :])
        m_comp_range[j, 1] = (m[0, 0] + m[0, P])/2
        if P_t > P:
            q_comp_range[j, 0] = np.mean(np.diagonal(q)[1 : P])
            q_comp_range[j, 1] = (q[0, 0] + q[P, P])/2
            q_comp_range[j, 2] = (q[0, P] + q[P, 0])/2

        m, g = simple_iterator.iterate(t_simple, beta, alpha, m_simp_init, g_simp_init)
        m_simp_range[j] = np.squeeze(m)
        if P_t > P:
            q_simp_range[j, 0] = np.squeeze(m)
            q_simp_range[j, 1] = np.squeeze(g)

        F_range[j] = F

    with open("./Data/partial_PSB_magnetization_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, m_comp_range)
        np.save(file, m_simp_range)

    with open("./Data/partial_PSB_overlap_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, q_comp_range)
        np.save(file, q_simp_range)

    with open("./Data/partial_PSB_free_entropy_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, F_range)

def normal_saddle_point_run(beta, alpha_range, P, m_0, epsilon, t, t_step, tau_step, t_simple, seed):
    '''
    Solve the normal saddle-point equations over a range of alpha
    with beta_s = beta, P_t = P + 1 and c = 0.
    Compute both the PSB and partial PSB solutions in order to reproduce Figs. (4), (5), (6), (15) and (16).
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

    iterator = NormalIterator(mat_cor, P, P_t, n_normal_samples, n_binary_samples, key, tol)

    simple_iterator = SimpleNormalIterator(n_simple_samples)

    m_init = (m_0 - epsilon)*np.eye(P, P_t) + epsilon
    s_init = (1 - epsilon)*np.eye(P_t) + epsilon
    q_init = (m_0 - epsilon)*np.eye(P_t) + epsilon

    m_simp_init = m_0
    g_simp_init = m_0

    m_comp_range = np.zeros(n_alpha)
    q_comp_range = np.zeros((n_alpha, 2))
    m_simp_range = np.zeros(n_alpha)
    q_simp_range = np.zeros((n_alpha, 2))

    for j, alpha in enumerate(alpha_range):
        m, s, q, _ = iterator.iterate(t, t_step, tau_step, beta_s, beta, alpha, m_init, s_init, q_init)

        m_comp_range[j] = np.mean(np.diagonal(m))
        if P_t > P:
            q_comp_range[j, 0] = np.mean(np.diagonal(q)[: P])
            q_comp_range[j, 1] = q[P, P]

        m, g = simple_iterator.iterate(t_simple, beta, alpha, m_simp_init, g_simp_init)
        m_simp_range[j] = np.squeeze(m)
        if P_t > P:
            q_simp_range[j, 0] = np.squeeze(m)
            q_simp_range[j, 1] = np.squeeze(g)

    with open("./Data/PSB_normal_magnetization_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, m_comp_range)
        np.save(file, m_simp_range)

    with open("./Data/PSB_normal_overlap_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, q_comp_range)
        np.save(file, q_simp_range)
    
    m_init[0, P] = m_0

    m_comp_range = np.zeros((n_alpha, 2))
    q_comp_range = np.zeros((n_alpha, 3))
    m_simp_range = np.zeros(n_alpha)
    q_simp_range = np.zeros((n_alpha, 2))

    for j, alpha in enumerate(alpha_range):
        m, s, q, _ = iterator.iterate(t, t_step, tau_step, beta_s, beta, alpha, m_init, s_init, q_init)

        m_comp_range[j, 0] = np.mean(np.diagonal(m)[1 :])
        m_comp_range[j, 1] = (m[0, 0] + m[0, P])/2
        if P_t > P:
            q_comp_range[j, 0] = np.mean(np.diagonal(q)[1 : P])
            q_comp_range[j, 1] = (q[0, 0] + q[P, P])/2
            q_comp_range[j, 2] = (q[0, P] + q[P, 0])/2

        m, g = simple_iterator.iterate(t_simple, beta, alpha, m_simp_init, g_simp_init)
        m_simp_range[j] = np.squeeze(m)
        if P_t > P:
            q_simp_range[j, 0] = np.squeeze(m)
            q_simp_range[j, 1] = np.squeeze(g)

    with open("./Data/partial_PSB_normal_magnetization_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, m_comp_range)
        np.save(file, m_simp_range)

    with open("./Data/partial_PSB_normal_overlap_P=%d_beta=%.2f.npy" % (P, beta), "wb") as file:
        np.save(file, q_comp_range)
        np.save(file, q_simp_range)

def plot_overlap(beta, alpha_range, P_range, name):
    '''
    Load the magnetization from .npy files written by the function saddle_point_run
    Compare the prediction of the full saddle-point equations
    to that of the reduced saddle-point equations in order to reproduce Figs. (1), (2), (15) and (16).
    '''
    fig, (m_axes, q_axes) = plt.subplots(nrows = 2, ncols = len(P_range), sharex = True, sharey = "row", figsize = (15, 8))
    fig_axis = fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = "none", which = "both", top = False, bottom = False, left = False, right = False)
    
    fontsize = 13
    
    set_ylabel = True
    for P, m_axis, q_axis in zip(P_range, m_axes, q_axes):
        with open("./Data/%s_magnetization_P=%d_beta=%.2f.npy" % (name, P, beta), "rb") as file:
            m_comp_range = np.load(file)
            m_simp_range = np.load(file)
        
        with open("./Data/%s_overlap_P=%d_beta=%.2f.npy" % (name, P, beta), "rb") as file:
            q_comp_range = np.load(file)
            q_simp_range = np.load(file)
        
        m_simp_line = m_axis.plot(alpha_range, m_simp_range, linestyle = "-", color = "C0")[0]
        m_comp_line = m_axis.plot(alpha_range, m_comp_range, linestyle = "--", linewidth = 3, color = "C1")[0]
        m_axis.set_xlim(np.min(alpha_range), np.max(alpha_range))
        m_axis.tick_params(axis = "both", which = "major", labelsize = fontsize)
        m_axis.tick_params(axis = "both", which = "minor", labelsize = fontsize)
        m_axis.legend([m_simp_line, m_comp_line], [r"Simplified equations", r"Full equations with" + "\n" + r"$P = %d$ hidden units" % (P + 1)], fontsize = fontsize)
        if set_ylabel:
            m_axis.set_ylabel(r"Magnetization $m$", fontsize = fontsize)
        
        q_simp_line = q_axis.plot(alpha_range, q_simp_range, linestyle = "-", color = "C0")[0]
        q_comp_line = q_axis.plot(alpha_range, q_comp_range, linestyle = "--", linewidth = 3, color = "C3")[0]
        q_axis.set_xlim(np.min(alpha_range), np.max(alpha_range))
        q_axis.tick_params(axis = "both", which = "major", labelsize = fontsize)
        q_axis.tick_params(axis = "both", which = "minor", labelsize = fontsize)
        q_axis.legend([q_simp_line, q_comp_line], [r"Simplified equations", r"Full equations with" + "\n" + r"$P = %d$ hidden units" % (P + 1)], fontsize = fontsize)
        if set_ylabel:
            q_axis.set_ylabel(r"Spin-glass overlap $q$", fontsize = fontsize)
        
        set_ylabel = False
    
    fig_axis.set_xlabel(r"Load $\alpha$", fontsize = fontsize)
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
    plt.ylabel(r"Free entropy difference $f_{\mathrm{PBS}} - f_{\mathrm{partial \ PBS}}$")
    
    plt.show()