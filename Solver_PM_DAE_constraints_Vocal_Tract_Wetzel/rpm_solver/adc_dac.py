import numpy as np


def discretize_Bo0(solver, signal, t_max, t_min=0):
    # Time at begining of each frames
    t_frames = np.linspace(t_min, t_max, int((t_max-t_min)/solver.time_step))
    # Array initialization
    d_signal = np.zeros((len(t_frames), solver.struct.n_io, solver.p_order))
    # Fill the array using 0 order blocking
    d_signal[:, :, 0] = signal(t_frames)
    return d_signal


def discretize_Bo1(solver, signal, t_max, t_min=0):
    # Time at begining of each frames
    Nframes = int((t_max-t_min)/solver.time_step)
    t_frames = np.linspace(t_min, t_max+solver.time_step, Nframes + 1)
    # Array initialization
    d_signal = np.zeros((len(t_frames)-1, solver.struct.n_io, solver.p_order))

    # Value of the order 1 polynomial at 0
    basis_value = solver.basis.proj_diff_0[0, 1]
    # Derivative of the order 1 polynomial
    basis_diff = solver.basis.proj_diff_0[1, 1]

    # Evaluation of the signal at sampling points
    sampled_signal = signal(t_frames)
    # Derivative
    diff_sampled_signal = sampled_signal[1:] - sampled_signal[:-1]
    # Derivative coefficients
    d_signal[:, :, 1] = diff_sampled_signal / (solver.time_step * basis_diff)
    # Constant coefficients
    d_signal[:, :, 0] = sampled_signal[:-1] - d_signal[:, :, 1] * basis_value
    return d_signal
