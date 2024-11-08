import numpy as np
import torch
import gc
from Restricted_Boltzmann_machine_training import RBM, build_m_diag_range

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulation_run_winning_ticket_magnetization(beta, alpha_range, c, N, P, P_t, number_teacher_sampling_steps,
                                       number_monitored_sampling_steps, number_student_sampling_steps,
                                       initial_learning_rate, learning_rate_decay, momentum,
                                       number_student_0_training_epochs, number_student_training_epochs,
                                       number_monitored_training_epochs, number_burn_in_epochs, number_magnetization_samples,
                                       random_number_seed, random_batch_seed):
    '''
    Train three normal student RBMs (0, A and B) on data generated by a normal teacher RBM
    for a range of loads alpha in order to reproduce Figs. (6) and (11).
    The teacher and students have the same number of visible units N.
    The teacher, A and B all have P hidden units, while 0 has P_t hidden units.
    B is initialized with the initial values of the weights of 0 that become the largest after training.
    This function writes the magnetization of B as a function of alpha to a .npy file.
    See Section 3.2.2 of the paper for more details about the training procedure.
    See Restricted_Boltzmann_machine_training.py for the definition of the RBM class
    and the inputs of this function.
    '''
    n_alpha = len(alpha_range)

    monitor_student_sampling = False

    random_number_generator = torch.Generator(device = device)
    random_number_generator.manual_seed(random_number_seed)

    random_batch_generator = torch.Generator(device = "cpu")
    random_batch_generator.manual_seed(random_batch_seed)

    m_A_mean_range = np.zeros(n_alpha)
    m_A_std_range = np.zeros(n_alpha)

    m_B_mean_range = np.zeros(n_alpha)
    m_B_std_range = np.zeros(n_alpha)

    if c is None:
        mat_cor = None
    else:
        mat_cor = (1 - c) * torch.eye(P, P) + c * torch.ones((P, P))
    
    teacher = RBM(N, P, device = device, random_number_generator = random_number_generator,
                random_batch_generator = random_batch_generator).to(device)
    xi_s = teacher.initialize_weights(1., xi_0 = None, binarize = False, mat_cor = mat_cor.to(device))

    student_A = RBM(N, P, device = device, random_number_generator = random_number_generator,
                    random_batch_generator = random_batch_generator, teacher = teacher).to(device)
    student_A.initialize_weights(0., xi_0 = xi_s, P = P, binarize = False, bias_corner = False)

    student_0 = RBM(N, P_t, device = device, random_number_generator = random_number_generator,
                    random_batch_generator = random_batch_generator, teacher = teacher).to(device)
    student_0.initialize_weights(0., xi_0 = xi_s, P = P, binarize = False, bias_corner = False)
    student_0.backup_weights()

    student_B = RBM(N, P, device = device, random_number_generator = random_number_generator,
                    random_batch_generator = random_batch_generator, teacher = teacher).to(device)

    weight_decay = 1

    for i, alpha in enumerate(alpha_range):
        M = int(alpha*N)
        
        data_batch_size = int(0.5*M)
        
        sigma = torch.nn.Parameter(torch.sign(torch.randn((M, N), device = device, generator = random_number_generator)), requires_grad = False)
        
        teacher.gibbs_sample_visible(sigma, beta, number_teacher_sampling_steps, number_monitored_sampling_steps)
        
        loader = torch.utils.data.DataLoader(dataset = sigma, batch_size = data_batch_size, shuffle = True, generator = random_batch_generator)
        
        # Train student A without lottery ticket initialization.
        m_A_list = student_A.langevin_training(loader, beta, alpha, initial_learning_rate, learning_rate_decay,
                                            weight_decay, momentum, number_student_sampling_steps, number_student_training_epochs,
                                            number_monitored_training_epochs, monitor_student_sampling,
                                            number_burn_in_epochs, number_magnetization_samples)
        
        # Train student 0.
        student_0.langevin_training(loader, beta, alpha, initial_learning_rate, learning_rate_decay,
                                    weight_decay, momentum, number_student_sampling_steps, number_student_0_training_epochs,
                                    number_monitored_training_epochs, monitor_student_sampling, number_burn_in_epochs, 0)
        
        # Train student B with magnitude pruning initialization from student 0.
        # Keep the P patterns of student 0 that become the largest after training and prune the rest.
        xi_0 = student_0.backup_xi
        xi_0 = xi_0[:, torch.argsort(torch.sum(student_0.xi**2, axis = 0), descending = True)[: P]]
        
        student_B.initialize_weights(1., xi_0 = xi_0, P = P, binarize = False, bias_corner = False)
        
        m_B_list = student_B.langevin_training(loader, beta, alpha, initial_learning_rate, learning_rate_decay,
                                            weight_decay, momentum, number_student_sampling_steps, number_student_training_epochs,
                                            number_monitored_training_epochs, monitor_student_sampling,
                                            number_burn_in_epochs, number_magnetization_samples)
        
        m_A_range = np.abs(np.array(m_A_list))
        m_B_range = np.abs(np.array(m_B_list))
        
        m_A_range = build_m_diag_range(m_A_range)
        m_B_range = build_m_diag_range(m_B_range)
        
        m_A_mean_range[i] = np.mean(m_A_range)
        m_A_std_range[i] = np.std(m_A_range)
        
        m_B_mean_range[i] = np.mean(m_B_range)
        m_B_std_range[i] = np.std(m_B_range)
        
        del sigma
        
        # Reinitialize weights.
        student_A.initialize_weights(0., xi_0 = xi_s, P = P, binarize = False, bias_corner = False)
        
        student_0.initialize_weights(0., xi_0 = xi_s, P = P, binarize = False, bias_corner = False)

    with open("./Data/simulated_normal_magnetization_magnitude_pruning_c=%.2f_P=%d_beta=%.2f.npy" % (c, P, beta), "wb") as file:
        np.save(file, m_A_mean_range)
        np.save(file, m_A_std_range)
        np.save(file, m_B_mean_range)
        np.save(file, m_B_std_range)

    del teacher
    del student_A
    del student_0
    del student_B

    gc.collect()

def simulation_run_winning_ticket_lead(beta, alpha_range, c, N, P, P_t, number_teacher_sampling_steps,
                                       number_monitored_sampling_steps, number_student_sampling_steps,
                                       initial_learning_rate, learning_rate_decay, momentum,
                                       number_student_0_training_epochs, number_student_training_epochs,
                                       number_monitored_training_epochs, number_burn_in_epochs, number_magnetization_samples,
                                       random_number_seed, random_batch_seed):
    '''
    Train three normal student RBMs (0, A and B) on data generated by a normal teacher RBM
    for a range of loads alpha in order to reproduce Figs. (6) and (11).
    The teacher and students have the same number of visible units N.
    The teacher, A and B all have P hidden units, while 0 has P_t hidden units.
    B is initialized with the initial values of the weights of 0 that become the largest after training.
    This function writes the lead of B over A as a function of the epochs to a .npy file.
    See Section 3.2.2 of the paper for more details about the training procedure.
    See Restricted_Boltzmann_machine_training.py for the definition of the RBM class
    and the inputs of this function.
    '''
    n_alpha = len(alpha_range)

    monitor_student_sampling = False

    random_number_generator = torch.Generator(device = device)
    random_number_generator.manual_seed(random_number_seed)

    random_batch_generator = torch.Generator(device = "cpu")
    random_batch_generator.manual_seed(random_batch_seed)

    m_d_range = np.zeros((n_alpha, P, number_magnetization_samples))

    m_A_mean_range = np.zeros(n_alpha)
    m_A_std_range = np.zeros(n_alpha)

    m_B_mean_range = np.zeros(n_alpha)
    m_B_std_range = np.zeros(n_alpha)

    if c is None:
        mat_cor = None
    else:
        mat_cor = (1 - c) * torch.eye(P, P) + c * torch.ones((P, P))
    
    teacher = RBM(N, P, device = device, random_number_generator = random_number_generator,
                random_batch_generator = random_batch_generator).to(device)
    xi_s = teacher.initialize_weights(1., xi_0 = None, binarize = False, mat_cor = mat_cor.to(device))

    student_A = RBM(N, P, device = device, random_number_generator = random_number_generator,
                    random_batch_generator = random_batch_generator, teacher = teacher).to(device)
    student_A.initialize_weights(0., xi_0 = xi_s, P = P, binarize = False, bias_corner = False)

    student_0 = RBM(N, P_t, device = device, random_number_generator = random_number_generator,
                    random_batch_generator = random_batch_generator, teacher = teacher).to(device)
    student_0.initialize_weights(0., xi_0 = xi_s, P = P, binarize = False, bias_corner = False)
    student_0.backup_weights()

    student_B = RBM(N, P, device = device, random_number_generator = random_number_generator,
                    random_batch_generator = random_batch_generator, teacher = teacher).to(device)

    weight_decay = 1

    for i, alpha in enumerate(alpha_range):
        M = int(alpha*N)
        
        data_batch_size = int(0.5*M)
        
        sigma = torch.nn.Parameter(torch.sign(torch.randn((M, N), device = device, generator = random_number_generator)), requires_grad = False)
        
        teacher.gibbs_sample_visible(sigma, beta, number_teacher_sampling_steps, number_monitored_sampling_steps)
        
        loader = torch.utils.data.DataLoader(dataset = sigma, batch_size = data_batch_size, shuffle = True, generator = random_batch_generator)
        
        m_A_list = student_A.langevin_training(loader, beta, alpha, initial_learning_rate, learning_rate_decay,
                                            weight_decay, momentum, number_student_sampling_steps, number_student_training_epochs,
                                            number_monitored_training_epochs, monitor_student_sampling,
                                            number_burn_in_epochs, number_magnetization_samples)
        
        student_0.langevin_training(loader, beta, alpha, initial_learning_rate, learning_rate_decay,
                                    weight_decay, momentum, number_student_sampling_steps, number_student_0_training_epochs,
                                    number_monitored_training_epochs, monitor_student_sampling, number_burn_in_epochs, 0)
        
        xi_0 = student_0.backup_xi
        xi_0 = xi_0[:, torch.argsort(torch.sum(student_0.xi**2, axis = 0), descending = True)[: P]]
        
        student_B.initialize_weights(1., xi_0 = xi_0, P = P, binarize = False, bias_corner = False)
        
        m_B_list = student_B.langevin_training(loader, beta, alpha, initial_learning_rate, learning_rate_decay,
                                            weight_decay, momentum, number_student_sampling_steps, number_student_training_epochs,
                                            number_monitored_training_epochs, monitor_student_sampling,
                                            number_burn_in_epochs, number_magnetization_samples)
        
        m_A_range = np.abs(np.array(m_A_list))
        m_B_range = np.abs(np.array(m_B_list))
        # (n_samples, P, P_t)
        
        m_A_range = build_m_diag_range(m_A_range)
        m_B_range = build_m_diag_range(m_B_range)
        
        m_d_range[i] = (m_B_range - m_A_range).T
        
        del sigma
        
        # Reinitialize weights
        student_A.initialize_weights(0., xi_0 = xi_s, P = P, binarize = False, bias_corner = False)
        
        student_0.initialize_weights(0., xi_0 = xi_s, P = P, binarize = False, bias_corner = False)

    # Calculate the lead of B over A.
    m_d_med_range = np.median(m_d_range, axis = (0, 1))
    m_d_mad_range = np.mean(np.abs(m_d_range - m_d_med_range), axis = (0, 1))

    with open("./Data/simulated_normal_magnetization_difference_magnitude_pruning_c=%.2f_P=%d_beta=%.2f.npy" % (c, P, beta), "wb") as file:
        np.save(file, m_d_med_range)
        np.save(file, m_d_mad_range)

    del teacher
    del student_A
    del student_0
    del student_B

    gc.collect()

def plot_winning_ticket_lead(beta, alpha_range, P_sim, P_saddle, number_student_training_epochs, c = None):
    '''
    Load the magnetization and magnetization difference from .npy files written
    by the functions simulation_run_winning_ticket_magnetization and simulation_run_winning_ticket_lead.
    Compare the magnetization to the prediction of the saddle-point equations
    and plot the magnetization difference as a function of the epochs
    in order to reproduce Fig. (6) and (11).
    '''
    fig, (overlap_axis, lead_axis) = plt.subplots(nrows = 1, ncols = 2, sharex = False, sharey = False, figsize = (15, 6))
    
    fontsize = 19
    
    if c is None:
        with open("./Data/partial_PSB_normal_magnetization_P=%d_beta=%.2f.npy" % (P_saddle, beta), "rb") as file:
            m_range = np.load(file)
        
        with open("./Data/simulated_normal_magnetization_magnitude_pruning_P=%d_beta=%.2f.npy" % (P_sim, beta), "rb") as file:
            m_A_mean_range = np.load(file)
            m_A_std_range = np.load(file)
            m_B_mean_range = np.load(file)
            m_B_std_range = np.load(file)
        
        with open("./Data/simulated_normal_magnetization_difference_magnitude_pruning_P=%d_beta=%.2f.npy" % (P_sim, beta), "rb") as file:
            m_d_med_range = np.load(file)
            m_d_mad_range = np.load(file)
    else:
        with open("./Data/normal_magnetization_c=%.2f_P=%d_P_t=%d_beta=%.2f.npy" % (c, P_saddle, P_saddle, beta), "rb") as file:
            m_range = np.load(file)
        
        with open("./Data/simulated_normal_magnetization_magnitude_pruning_c=%.2f_P=%d_beta=%.2f.npy" % (c, P_sim, beta), "rb") as file:
            m_A_mean_range = np.load(file)
            m_A_std_range = np.load(file)
            m_B_mean_range = np.load(file)
            m_B_std_range = np.load(file)
        
        with open("./Data/simulated_normal_magnetization_difference_magnitude_pruning_c=%.2f_P=%d_beta=%.2f.npy" % (c, P_sim, beta), "rb") as file:
            m_d_med_range = np.load(file)
            m_d_mad_range = np.load(file)
    
    overlap_axis.errorbar(alpha_range, m_B_mean_range, m_B_std_range, marker = "o", linestyle = "--", capsize = 3, zorder = 2.5, color = "C0")
    overlap_axis.plot(alpha_range, m_range[:, 0], color = "C1")
    overlap_axis.tick_params(axis = "both", which = "both", labelsize = fontsize)
    overlap_axis.set_xlabel(r"Load $\alpha$", fontsize = fontsize)
    overlap_axis.set_ylabel(r"Magnetization $m$", fontsize = fontsize)
    
    number_magnetization_samples = len(m_d_med_range)
    epochs = number_student_training_epochs//number_magnetization_samples*np.arange(number_magnetization_samples)
    
    lead_axis.plot(epochs, np.zeros_like(epochs), linestyle = "--", color = "black")
    lead_axis.plot(epochs, m_d_med_range, linestyle = "-", zorder = 2.5)
    lead_axis.fill_between(epochs, m_d_med_range - m_d_mad_range, m_d_med_range + m_d_mad_range, alpha = 0.5, zorder = 2.4, edgecolor = "C0")
    lead_axis.set_xlim(left = epochs[0], right = epochs[-1])
    lead_axis.tick_params(axis = "both", which = "both", labelsize = fontsize)
    lead_axis.yaxis.tick_right()
    lead_axis.set_xlabel(r"Epoch", fontsize = fontsize)
    lead_axis.set_ylabel(r"Lead $m_B - m_A$ of B over A", fontsize = fontsize, rotation = -90, labelpad = 20)
    lead_axis.yaxis.set_label_position("right")
    plt.tight_layout()
    plt.show()