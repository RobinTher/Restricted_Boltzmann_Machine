import numpy as np
import jax.numpy as jnp
from functools import partial
import jax
from Saddle_point_iteration import mat_equi_cor, init_spins, hamiltonian_M, probability, accumulate_inv_eigvals

import matplotlib.pyplot as plt
from matplotlib import colors

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def inv_eigval(beta_s_range, c_range, P):
    '''
    Compute the inverse maximum eigenvalue of the matrix S corresponding to teacher patterns
    with uniform correlations for a range of c and beta_s.
    S is defined in Appendix F of the paper.
    Save the results in a .npy file.
    '''

    mat_cor = mat_equi_cor(c_range[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis], P)

    spins_s_T, spins_s = init_spins(P, n_pre_appended_axes = 2)

    H_M_s = hamiltonian_M(spins_s_T, spins_s, mat_cor, beta_s_range)
    p_M_s = probability(H_M_s)

    d_range = jnp.squeeze(jnp.sum(p_M_s * spins_s_T * spins_s, axis = 0))[..., 0, 1]
    eigval_range = (P - 1)**2 * c_range*d_range + (P - 1) * (c_range + d_range) + 1

    inv_eigval_range = 1/eigval_range

    with open("./Data/inv_eigval_range_P=%d.npy" % P, "wb") as file:
        np.save(file, inv_eigval_range)

def random_inv_eigval(beta_s_range, c_range, P, t, seed):
    '''
    Compute the inverse maximum eigenvalue of the matrix S corresponding to teacher patterns
    with random correlations for a range of c and beta_s.
    S is defined in Appendix F of the paper.
    The results are averaged over t iterations.
    The random correlations are defined in Appendix A.3 of the paper.
    Save the results in a .npy file.
    '''
    n_beta = len(beta_s_range)
    n_c = len(c_range)

    key = jax.random.PRNGKey(seed)

    inv_eigval_range = np.zeros((n_beta, n_c))

    mat_cor_cor = mat_equi_cor(c_range[np.newaxis, :, jnp.newaxis, jnp.newaxis, jnp.newaxis], P)

    spins_s_T, spins_s = init_spins(P, n_pre_appended_axes = 2)

    inv_eigval_range, key = jax.lax.fori_loop(0, t, partial(accumulate_inv_eigvals, spins_s_T, spins_s,
                                                            mat_cor_cor, beta_s_range, P, n_beta, n_c),
                                              (inv_eigval_range, key))

    with open("./Data/random_inv_eigval_range_P=%d.npy" % P, "wb") as file:
        np.save(file, inv_eigval_range)

def plot_inv_eigval(T_range_list, c_range_list, P_range):
    '''
    Plot the inverse maximum eigenvalue of the matrix S for a range of c and T.
    S is defined in Appendix F of the paper.
    Load inverse eigenvalues from .npy files written by the functions inv_eigval and random_inv_eigval.
    '''
    file_name_list = ["inv_eigval_range", "random_inv_eigval_range"]
    row_label_list = [r"Uniform correlations", r"Random correlations"]

    fig, axes = plt.subplots(2, 3, sharex = True, sharey = True, figsize = (15, 6))
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = "none", which = "both", top = False, bottom = False, left = False, right = False)
    fontsize = 13

    for i in range(2):
        for P, axis in zip(P_range, axes[i]):
            file_name = file_name_list[i]
            c_range = c_range_list[i]
            T_range = T_range_list[i]
            row_label = row_label_list[i]
            
            with open("./Data/%s_P=%d.npy" % (file_name, P), "rb") as file:
                inv_eigval_range = np.load(file)
        
            image = axis.contourf(c_range, T_range, 1/inv_eigval_range, origin = "lower", cmap = "viridis", levels = 8)
            axis.set_xlim(np.min(c_range), np.max(c_range))
            axis.tick_params(axis = "both", which = "minor", labelsize = fontsize)
            axis.tick_params(axis = "both", which = "major", labelsize = fontsize)
            
            colorbar = plt.colorbar(image, ax = axis)
            colorbar.ax.tick_params(labelsize = fontsize)
            colorbar.set_label(r"Largest eigenvalue $\lambda^{\mathcal{S}}_{\mathrm{max}}$", fontsize = fontsize)
            
            if i == 0:
                axis.set_title(r"No. hidden units $P = %d$" % P, fontsize = fontsize)
        
        axis.set_ylabel(row_label, fontsize = fontsize, rotation = -90, labelpad = 90)
        axis.yaxis.set_label_position("right")

    plt.xlabel(r"Correlation $c$", fontsize = fontsize)
    plt.ylabel(r"Temperature $T$", fontsize = fontsize)
    plt.show()

def plot_critical_load(T_range_list, c_range_list, P_range):
    '''
    Plot the critical load alpha_crit for a range of c and T.
    Load inverse eigenvalues from .npy files written by the functions inv_eigval and random_inv_eigval
    and use them to compute the critical load as described in the paper.
    '''
    file_name_list = ["inv_eigval_range", "random_inv_eigval_range"]
    row_label_list = [r"Uniform correlations", r"Random correlations"]

    fig, axes = plt.subplots(2, 3, sharex = True, sharey = True, figsize = (15, 6))
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = "none", which = "both", top = False, bottom = False, left = False, right = False)
    fontsize = 13
    levels = 10.**np.arange(-16, 1, 1)

    for i in range(2):
        for P, axis in zip(P_range, axes[i]):
            file_name = file_name_list[i]
            c_range = c_range_list[i]
            T_range = T_range_list[i]
            row_label = row_label_list[i]
            
            with open("./Data/%s_P=%d.npy" % (file_name, P), "rb") as file:
                inv_eigval_range = np.load(file)
            
            alpha_crit = T_range[:, np.newaxis]**4*inv_eigval_range
            image = axis.contourf(c_range, T_range, alpha_crit, origin = "lower", cmap = "viridis", levels = levels, norm = colors.LogNorm())
            axis.set_xlim(np.min(c_range), np.max(c_range))
            axis.tick_params(axis = "both", which = "minor", labelsize = fontsize)
            axis.tick_params(axis = "both", which = "major", labelsize = fontsize)
            
            if i == 0:
                axis.set_title(r"No. hidden units $P = %d$" % P, fontsize = fontsize)
        
        axis.set_ylabel(row_label, fontsize = fontsize, rotation = -90, labelpad = 20)
        axis.yaxis.set_label_position("right")

    colorbar = plt.colorbar(image, ax = axes.ravel().tolist())
    colorbar.ax.tick_params(labelsize = fontsize)
    colorbar.set_label(r"Critical load $\alpha_{\mathrm{crit}}$", fontsize = fontsize)
    plt.xlabel(r"Correlation $c$", fontsize = fontsize)
    plt.ylabel(r"Temperature $T$", fontsize = fontsize)
    plt.show()