import numpy as np
from scipy.integrate import quad
from scipy.integrate import simpson
from scipy.special import erfinv

import matplotlib.pyplot as plt
import matplotlib.colors as colors

tol = 2*np.sqrt(np.finfo("float32").eps)

# Setting up plotting

# nishimori_cmap = colors.ListedColormap(["cornflowerblue", "sandybrown"])
# fixed_T_s_cmap = colors.ListedColormap(["cornflowerblue", "mediumpurple", "sandybrown"])

nishimori_cmap = colors.ListedColormap(["C0", "C1"])
fixed_T_s_cmap = colors.ListedColormap(["C0", "C3", "C1"])

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def plot_phase(phases, n_beta, n_alpha, alpha_range, T_range, T_ref = None, nishimori = False):
    '''
    Plot a phase diagram found using the saddle-point equations.
    
    Inputs
    ------
    phases (float array):
        Array of phases as a function of T (axis 0) and alpha (axis 1).
        Each phase is represented by a different number between 0 and 1.
    n_beta (int):
        Size of the phase array along axis 0.
    n_alpha (int):
        Size of the phase array along axis 1.
    alpha_range (float array):
        Range of alpha corresponding to axis 1 of the phase array.
    T_range (float array):
        Range of T corresponding to axis 0 of the phase array.
    nishimori (bool):
        whether we plot the phases on the Nishimori line.
    
    Outputs
    -------
    None
    '''
    fontsize = 13
    
    alpha_min = 0
    alpha_max = np.max(alpha_range)
    T_min = np.around(np.min(T_range))
    T_max = np.around(np.max(T_range))

    if nishimori:
        cmap = nishimori_cmap
    else:
        cmap = fixed_T_s_cmap
    
    plt.matshow(phases, vmin = 0, vmax = 1, origin = "lower", cmap = cmap)

    if nishimori:
        colorbar = plt.colorbar(ticks = [1/4, 3/4], drawedges = True)
        colorbar.ax.set_yticklabels([r"F", r"P"], fontsize = fontsize)
    else:
        colorbar = plt.colorbar(ticks = [1/6, 3/6, 5/6], drawedges = True)
        colorbar.ax.set_yticklabels([r"F", r"SG", r"P"], fontsize = fontsize)
    
    plt.contour(phases, colors = "black")
    
    if T_ref is not None:
        plt.plot(np.full_like(alpha_range, T_ref/T_max * n_beta), color = "white", linestyle = "--", linewidth = 2)
    
    x_labels = np.linspace(alpha_min, alpha_max, num = 5, endpoint = True)
    if np.all(x_labels == np.floor(x_labels)):
        x_labels = x_labels.astype("int32")
    
    y_labels = np.linspace(T_min, T_max, num = 5, endpoint = True)
    if np.all(y_labels == np.floor(y_labels)):
        y_labels = y_labels.astype("int32")
    
    plt.xticks(ticks = np.linspace(0, n_alpha-1, num = 5, endpoint = True),
               labels = x_labels, fontsize = fontsize)
    plt.yticks(ticks = np.linspace(0, n_beta-1, num = 5, endpoint = True),
               labels = y_labels, fontsize = fontsize)
    plt.gca().xaxis.tick_bottom()
    plt.xlabel(r"$\alpha$", fontsize = fontsize)
    plt.ylabel(r"$T$", fontsize = fontsize)

    if nishimori:
        plt.savefig("./Data/phase_diagram_P1_nishimori.png")
    else:
        plt.savefig("./Data/phase_diagram_P1.png")
    
    plt.show()
    
def integral(y, x):
    '''
    Definite integral of y with respect to x.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html.
    Used as a helper function.
    '''
    return simpson(y, x)

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
    
    def update(self, beta_s, beta, alpha, m, q):
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
        m (float):
            Mattis magnetization.
        q (float):
            Overlap between student patterns from different samples that did not converge to a student pattern.
        
        Outputs
        -------
        m (float):
            Updated value of m.
        q (float):
            Updated value of q.
        '''
        m = m[..., np.newaxis]
        q = q[..., np.newaxis]
        
        m_hat = np.squeeze(beta_s*beta*alpha) * integral(1/np.sqrt(2*np.pi) * np.exp(-self.z**2/2)
                                         * np.tanh(beta_s*beta * m + beta * np.sqrt(q) * self.z), self.z)
        
        q_hat = np.squeeze(beta**2*alpha) * integral(1/np.sqrt(2*np.pi) * np.exp(-self.z**2/2)
                                         * np.tanh(beta**2 * m + beta * np.sqrt(q) * self.z)**2, self.z)
        
        m_hat = m_hat[..., np.newaxis]
        q_hat = q_hat[..., np.newaxis]
        
        m = integral(1/np.sqrt(2*np.pi) * np.exp(-self.z**2/2) * np.tanh(m_hat + np.sqrt(q_hat) * self.z), self.z)
    
        q = integral(1/np.sqrt(2*np.pi) * np.exp(-self.z**2/2) * np.tanh(m_hat + np.sqrt(q_hat) * self.z)**2, self.z)
        
        return m, q
    
    def iterate(self, t, beta_s, beta, alpha, m, q):
        '''
        Run update multiple times to solve the saddle-point equations.

        Inputs
        ------
        beta_s (float):
            Inverse temperature of the teacher.
        beta (float):
            Inverse temperature of the student.
        alpha (float):
            Normalized number of examples sent by the teacher to the student.
        m (float):
            Mattis magnetization.
        q (float):
            SG overlap.
        
        Outputs
        -------
        m (float):
            Final value of m.
        q (float):
            Final value of q.
        '''
        m_prev, q_prev = m, q
        m_cur, q_cur = self.update(beta_s, beta, alpha, m_prev, q_prev)

        t_cur = 0
        while np.any(np.abs(m_cur - m_prev) >= tol) and t_cur < t:
            m_prev, q_prev = m_cur, q_cur
            m_cur, q_cur = self.update(beta_s, beta, alpha, m_prev, q_prev)
            t_cur += 1

        m, q = m_cur, q_cur
        return m, q