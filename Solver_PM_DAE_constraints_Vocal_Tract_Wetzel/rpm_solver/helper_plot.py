import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_basis(solver):
    """Plots both basis of the solver in two different subplots.

    Args:
        solver (object): instance of the solver class

    Returns:
        object: matplotlib figure
    """
    evaluate_proj = solver.basis.evaluate_proj
    evaluate_regul = solver.basis.evaluate_regul
    x = np.linspace(0, 1, 500)
    y_proj = evaluate_proj(x).T

    fig = plt.figure(figsize=(12, 6))
    plt.suptitle("Basis functions")

    plt.subplot(1, 2, 1)
    plt.title("Functions of the projection step")
    for n in range(len(y_proj)):
        plt.plot(x, y_proj[n], label=f'Basis function {n}')
        plt.plot(x-1, y_proj[n])
        plt.plot(x+1, y_proj[n])
    plt.xlabel('$ tau $')
    plt.legend()

    x2 = np.linspace(-1, 0, 500)
    y_regul = evaluate_regul(x).T
    plt.subplot(1, 2, 2)
    plt.title("Basis functions of the regularization step")
    for n in range(int(len(y_regul)/2)):
        i_alpha_1 = n+int(len(y_regul)/2)
        plt.plot(x, y_regul[n], label=f'Basis function {n}')
        plt.plot(x, y_regul[i_alpha_1], label=f'Basis function {n}')
        plt.plot(x2, y_regul[n], label=f'Basis function {i_alpha_1}',
                 linestyle="--")
        plt.plot(x2, y_regul[i_alpha_1], label=f'Basis function {i_alpha_1}',
                 linestyle="--")
    plt.xlabel('$ tau $')
    plt.legend()

    plt.tight_layout()
    return fig


def plot_basis_P(solver):
    """Plots the basis of the projector P.

    Args:
        solver (object): instance of the solver class

    Returns:
        object: matplotlib figure
    """
    # Evaluating basis function values
    evaluate_proj = solver.basis.evaluate_proj
    x = np.linspace(0, 1, 500)
    y_proj = evaluate_proj(x).T

    # Creating figure and colormap
    fig = plt.figure(figsize=(12, 6))
    plt.title("Basis functions of the projector P")
    plt.xlim((-1, 2))
    miny = np.min(y_proj)*1.2
    maxy = np.max(y_proj)*1.2
    plt.ylim(miny, maxy)

    norm = mpl.colors.Normalize(vmin=0, vmax=len(y_proj))
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.autumn)

    # Add background colors
    plt.axvspan(-1, 0, facecolor='pink', alpha=0.1)
    plt.axvspan(1, 2, facecolor='pink', alpha=0.1)

    for n in range(len(y_proj)):
        i = len(y_proj)-1-n
        alpha = 1 - i/(3*len(y_proj))
        color = cmap.to_rgba(i)
        if i == 0:
            lw = 3
        else:
            lw = 2
        plt.plot(x, y_proj[i], label=f'Basis function {i+1}',
                 c=color, alpha=alpha, linewidth=lw)
        plt.plot(x-1, y_proj[i], c=color, alpha=alpha-0.2, linewidth=lw)
        plt.plot(x+1, y_proj[i], c=color, alpha=alpha-0.2, linewidth=lw)

    # Add vertical lines to separate time frames
    plt.vlines([0, 1], [miny, miny], [maxy, maxy], colors='black',
               linestyles='dashed')

    # Label and legends
    plt.xlabel('$ tau $')
    plt.legend()

    return fig


def plot_gradients(solver, x, labels):
    x = np.tile(x, (solver.struct.n_state, 1))
    gradients = solver.struct.grad_H(x.T).T
    plt.figure()
    plt.title("Flows as functions of states")
    for i, gradient in enumerate(gradients):
        plt.plot(x[0], gradient, label=labels[i])
    plt.legend()


def plot_error_energy(solver, x, t):
    plt.figure()
    plt.title("Error on stored Energy")
    plt.plot(t, solver.struct.H(x) - solver.struct.H(x)[0])
    plt.xlabel("Time")
    plt.ylabel("Error on energy conservation")


def plot_flows_trajectories(solver, dx_proj, N_points=10):
    """Plot flows evolution using resynthesis operation
    to obtain flows values during frames from projection
    coefficients.

    Args:
        solver (object): instance of the RPM solver class
        dx_proj (array): array of projected flows coefficients of the
                        projection step.
                        Size : [n, solver.n_state, solver.p_order]
        N_points (int): number of points per frame for the plot.
    """
    # Number of frames
    steps = len(dx_proj)
    # Intermediate points
    tau = np.linspace(0, 1, N_points)
    # Computation of basis function values at intermediate points
    proj_points = solver.basis.evaluate_proj(tau).T
    # Synthesis
    synth_proj = np.zeros((steps, solver.struct.n_state, N_points))
    for step in range(steps):
        synth_proj[step] = dx_proj[step] @ proj_points

    # We want to plot trajectories using recontruction
    # from the projection coefficients
    plt.figure(figsize=(12, 6))
    for state in range(solver.struct.n_state):
        plt.subplot(1, solver.struct.n_state, state+1)
        for step in range(steps):
            if step == 0:
                plt.plot((step+tau)*solver.time_step, synth_proj[step, state], color='r',
                         label='Projection step')
            else:
                plt.plot((step+tau)*solver.time_step, synth_proj[step, state], color='r')
        plt.legend()


def plot_phase_diagram_2D_proj(solver, x, dx_proj, N_points=10,
                               indexes=[0, 1]):
    # First we evaluate integration of the basis functions at intermediate
    # points.
    taui = np.linspace(0, 1, N_points, endpoint=False)
    integrals = np.zeros((N_points, solver.k_order + solver.p_order))
    for i, tau in enumerate(taui):
        integrals[i] = solver.quad.integrate(solver.basis.evaluate_all, 0, tau)
    integrals_P = integrals[:, :solver.p_order]

    steps = len(x)
    int_states_non_regul = np.zeros((len(x) * N_points, solver.struct.n_state))
    for step in range(steps):
        first_ind = step*N_points
        last_ind = first_ind+N_points
        x0 = x[step]
        for state in range(solver.struct.n_state):
            int_states_non_regul[first_ind:last_ind, state] = \
                x0[state] +\
                solver.time_step * dx_proj[step, state]@integrals_P.T

    fig = plt.figure(figsize=(12, 6))
    plt.suptitle("Phase diagram")

    plt.plot(int_states_non_regul[:, indexes[0]],
             int_states_non_regul[:, indexes[1]])
    x0_points = np.arange(0, len(x)*N_points, N_points)
    plt.scatter(int_states_non_regul[x0_points, indexes[0]],
                int_states_non_regul[x0_points, indexes[1]], c='r')

    plt.title("Without regularization")
    return fig


def plot_gradients_phase_proj(solver, x, dx_proj, N_points=10,
                              indexes=[0, 1]):
    # First we evaluate integration of the basis functions at intermediate
    # points.
    taui = np.linspace(0, 1, N_points, endpoint=False)
    integrals = np.zeros((N_points, solver.k_order + solver.p_order))
    for i, tau in enumerate(taui):
        integrals[i] = solver.quad.integrate(solver.basis.evaluate_all, 0, tau)
    integrals_P = integrals[:, :solver.p_order]

    steps = len(x)
    int_states_non_regul = np.zeros((len(x) * N_points, solver.struct.n_state))
    for step in range(steps):
        first_ind = step*N_points
        last_ind = first_ind+N_points
        x0 = x[step]
        for state in range(solver.struct.n_state):
            int_states_non_regul[first_ind:last_ind, state] = \
                x0[state] +\
                solver.time_step * dx_proj[step, state]@integrals_P.T

    # Computation of gradients from the state
    gradients_non_regul = solver.struct.grad_H(int_states_non_regul)

    fig = plt.figure(figsize=(12, 6))
    plt.suptitle("Gradient phase diagram")

    plt.plot(gradients_non_regul[:, indexes[0]],
             gradients_non_regul[:, indexes[1]])
    x0_points = np.arange(0, len(x)*N_points, N_points)
    plt.scatter(gradients_non_regul[x0_points, indexes[0]],
                gradients_non_regul[x0_points, indexes[1]], c='r')

    plt.title("Without regularization")
    return fig
