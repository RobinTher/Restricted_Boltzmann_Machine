import numpy as np
import jax.numpy as jnp
import jax
from Saddle_point_iteration import Iterator, mat_equi_cor

import matplotlib.pyplot as plt
import cmasher as cmr
import re

tol = np.finfo("float32").eps

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

'''
beta_s = 1/1.25
P = 2
P_t = 2
c = 0.7

t = 100
t_step = 1/2
seed = 4

n_beta = 20
n_alpha = 20
T_range = np.linspace(0.53, 1.1, num = n_beta, endpoint = True)
alpha_range = np.linspace(0.045, 0.9, num = n_alpha, endpoint = True)
'''

def phase_diagram_fixed_T_s(beta_s, T_range, alpha_range, c, P, t, t_step, seed):
    '''
    Solve the saddle-point equations over a range of beta and alpha
    with fixed T_s and P_t = P in order to reproduce Figs. (8) and (9).
    Save the results in .npy files.
    '''
    n_beta = len(T_range)
    n_alpha = len(alpha_range)

    n_normal_samples = 8000000
    n_binary_samples = 80000000

    P_t = P

    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key)
    
    m_init = jnp.eye(P, P_t)
    s_init = jnp.eye(P_t)
    q_init = jnp.eye(P_t)

    mat_cor = mat_equi_cor(c, P)

    iterator = Iterator(mat_cor, P, P_t, n_normal_samples, n_binary_samples, key, tol)

    m_range = np.zeros((n_beta, n_alpha))
    q_range = np.zeros((n_beta, n_alpha))

    for i, T in enumerate(T_range):
        for j, alpha in enumerate(alpha_range):
            beta = 1/T
            
            m, s, q, p_M_s, _ = iterator.iterate(t, t_step, beta_s, beta, alpha, m_init, s_init, q_init)
            
            m_range[i, j] = np.mean(np.diagonal(m))
            q_range[i, j] = np.mean(np.diagonal(q))

    with open("./Data/binary_phase_diagram_T_s=%.2f_P=%d_c=%.2f.npy" % (1/beta_s, P, c), "wb") as file:
        np.save(file, m_range)
        np.save(file, q_range)

def plot_phase_diagram_fixed_T_s(T_s, T_range, alpha_range, c_range, P_range, name, plot_m = False):
    '''
    Plot phase diagrams with fixed T_s and P_t = P in order to reproduce Figs. (8) and (9).
    Load the phase diagrams from .npy files written by the function phase_diagram_fixed_T_s.
    '''
    fig, axes = plt.subplots(len(P_range), 3, sharex = True, sharey = True, figsize = (15, 6))
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = "none", which = "both", top = False, bottom = False, left = False, right = False)
    fontsize = 13

    for i, P in enumerate(P_range):
        for c, axis in zip(c_range, axes[i]):
            file_name = "./Data/%s_phase_diagram_P=%d_c=%.2f.npy" % (name, P, c)
            file_name = re.sub("__", "_", file_name, count = 1)
            with open(file_name, "rb") as file:
                _ = np.load(file)
                eigval_range = np.load(file)
            
            file_name = "./Data/%s_phase_diagram_T_s=%.2f_P=%d_c=%.2f.npy" % (name, T_s, P, c)
            file_name = re.sub("__", "_", file_name, count = 1)
            with open(file_name, "rb") as file:
                m_range = np.load(file)
                q_range = np.load(file)
            
            if plot_m:
                image = axis.imshow(m_range, aspect = 2*0.5357142857142857, origin = "lower", cmap = cmr.fall,
                                    extent = (np.min(alpha_range), np.max(alpha_range),
                                              np.min(T_range), np.max(T_range)),
                                    vmin = 0, vmax = 0.75)
            else:
                image = axis.imshow(q_range, aspect = 2*0.5357142857142857, origin = "lower", cmap = cmr.fall,
                                    extent = (np.min(alpha_range), np.max(alpha_range),
                                              np.min(T_range), np.max(T_range)),
                                    vmin = 0, vmax = 0.75)
            axis.plot(T_range**4/eigval_range, T_range, color = "white")
            axis.set_xlim(np.min(alpha_range), np.max(alpha_range))
            axis.tick_params(axis = "both", which = "minor", labelsize = fontsize)
            axis.tick_params(axis = "both", which = "major", labelsize = fontsize)
            if i == 0:
                axis.set_title(r"Correlation $c = %.1f$" % c, fontsize = fontsize)
        
        axis.set_ylabel(r"No. hidden units $P = %d$" % P, fontsize = fontsize, rotation = -90, labelpad = 20)
        axis.yaxis.set_label_position("right")

    colorbar = plt.colorbar(image, ax = axes.ravel().tolist())
    colorbar.ax.tick_params(labelsize = fontsize)
    if plot_m:
        colorbar.set_label(r"Magnetization $m$", fontsize = fontsize)
    else:
        colorbar.set_label(r"Overlap $q$", fontsize = fontsize)
    plt.xlabel(r"Load $\alpha$", fontsize = fontsize)
    plt.ylabel(r"Temperature $T$", fontsize = fontsize)
    plt.show()