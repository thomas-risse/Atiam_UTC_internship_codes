import numpy as np
import scipy.linalg as spe


"""Functions associated with numerical quadrature
for the computation of the projected gradient and
all associated integrals.
"""


def hessian_basis_term(solver):
    """Computes outer product of basis quad points
    and basis int points so that the hessian can later
    be evaluated with one einsum step.

    Args:
        solver (object): solver instance

    Returns:
        array: inner product of basis functions values
               and integrals at each integration point.
               Size : [quad_order, proj_order, proj_order]
    """
    return np.einsum('ai, aj -> aij',
                     solver.basis_quad_points.T,
                     solver.basis_int.T)


def _compute_intermediate_states(solver, x0, dx):
    """Computes the value of states variables at quadrature points
    in a frame using values of x0 the state at the begining of the frame
    and dx the projected flows

    Args:
        x0 (array): states (non projected) at the begining of the frame.
                        Size: [solver.n_state]
        dx (2D array): projected flows for the frame.
                        Size: [solver.n_state, solver.p_order]

    Returns:
        2D array: estimated state (non projected) values at
                    quadrature points.
                    Size: [solver.n_state, solver.quad_order]
    """
    inter_states = np.tile(x0, (solver.quad_order, 1)).T
    # compute change of state compared to begining of frame
    return inter_states + solver.time_step * dx @ solver.basis_int


def _proj_gradient(solver, int_states):
    """Compute projected gradients of the system using state
    at the begining of the frame and estimation of the flows projection
    coefficients dx.
    projGradient[i,j] is the the projection
    of the gradient i on the basis function j


    Args:
        int_states: estimated state values at quadrature points.
                    Size: [solver.n_state, solver.quad_order]
    Returns:
        2D array: array of projected gradients coefficients
                    of size [solver.n_state, solver.p_order]
    """
    # Initialization of the array
    proj_gradient = np.zeros((solver.n_state, solver.p_order))
    # Evaluation of gradients at each quadrature points
    # size(gradHint_states) = [solver.n_states, solver.quad_order]
    grad_H_int_states = solver.gradients(int_states.T).T
    # Add weights to the quadrature points to perform integration
    w_quad_points = solver.basis_quad_points*solver.quad.quad_weights
    proj_gradient = grad_H_int_states @ w_quad_points.T
    """for i in range(solver.p_order):
        to_integrate = grad_H_int_states * solver.basis_quad_points[i, :]
        proj_gradient[:, i] = solver.quad.quadrature_array(to_integrate)"""
    return proj_gradient


def _fToOptimize(solver, dx, l_mults, int_states):
    """Function to optimize at each projection step,
    so that dx - S*grad(dx) = 0

    Args:
        dx (2D array): projected flows coefficients for the frame.
                        Size: [solver.n_state, solver.p_order]
        int_states: estimated state values at quadrature points.
                        Size: [solver.n_state, solver.quad_order]

    Returns:
        array: error for each row of the system. Size: [solver.proj_size]
    """
    left_side = np.concatenate((dx.flatten(),
                               np.zeros(solver.cons_proj_size)), axis=0)
    right_side = np.concatenate((_proj_gradient(solver, int_states).flatten(),
                                l_mults.flatten()), axis=0)
    return left_side - solver.S_proj @ right_side


def _jacobi(solver, int_states):
    """Computes the derivatives of fToOptimize
    with respect to the projected flows variables

    Args:
        int_states: estimated state values at quadrature points.
                    Size: [solver.n_state, solver.quad_order]

    Returns:
        2D array: Jacobian matrix of the problem.
                    Size: [solver.proj_size, solver.proj_size]
    """
    # Hamiltonian related term
    # Compute unprojected hessian at integration points
    hessian_H = solver.hessian(int_states.T)
    # hessian_H = np.moveaxis(hessian_H, [0, 1, 2], [1, 2, 0])
    # Compute kronecker product for each integration point
    to_integrate = np.einsum('aij, akl -> aikjl',
                             hessian_H,
                             solver.hessian_basis_term).\
        reshape(solver.quad_order,
                solver.p_order*solver.n_state,
                solver.p_order*solver.n_state)
    # Quadrature weights
    weights = solver.quad.quad_weights
    # Hamiltonian related terms
    H_right_term = solver.time_step * np.einsum('aij, a -> ij',
                                                to_integrate, weights)
    H_left_term = np.diag(np.ones(solver.H_proj_size))

    # Constraints related terms
    L_right_term = np.diag(np.ones(solver.cons_proj_size))
    L_left_term = np.zeros((solver.cons_proj_size, solver.cons_proj_size))

    # Assembling
    right_term = spe.block_diag(H_right_term, L_right_term)
    left_term = spe.block_diag(H_left_term, L_left_term)

    # Returning jacobian matrix
    return left_term - solver.S_proj @ right_term


def f_and_jac_quad(solver, x0, p_flows):
    """Computes value of f and associated jacobian
    for the solver given and values of state at begining
    of frame and projected state flows.

    Args:
        solver (object): instance of the rpm solver class.
        x0 (array): states (non projected) at the begining of the frame.
                        Size: [solver.n_state]
        p_flows (2D array): projected flows for the frame.
                        Size: [solver.full_size_unproj, solver.p_order]

    Returns:
        array: value of f(x0, dx). Size: [solver.proj_size]
        2D array: value of the jacobian.
                  Size: [solver.proj_size, solver.proj_size]
    """
    p_flows_H = p_flows[0:solver.n_state]
    p_flows_L = p_flows[solver.n_state:]
    int_states = _compute_intermediate_states(solver, x0, p_flows_H)
    return _fToOptimize(solver, p_flows_H, p_flows_L, int_states),\
        _jacobi(solver, int_states)

# Computation of un-projected gradient


def _func_weighted_sum(funcs, weights):
    """Utility function used to create a function
    corresponding to the weighted sum of several functions

    Args:
        funcs (array): array of functions to add
        weights (array): array of weights

    Returns:
        function: weighted sum function
    """
    def sum_func(x):
        out = 0
        for i, weight in enumerate(weights):
            out += weight*funcs[i](x)
        return out
    return sum_func


def compute_state(solver, x0, dx, tau):
    """Given dx an x0 fr the frame, computes
    the state at a given time tau between 0 and 1 using
    quadrature of order solver.quad_order.

    Args:
        x0 (array): states at the begining of the frame.
            Size: [solver.n_state]
        dx (2D array): projected state flows.
            Size: [solver.n_state, solver.p_order]
        tau (float): time at which the estimation is needed

    Returns:
        array: values of the states Size: [solver.n_state]
    """
    states_tau = np.zeros((solver.n_state))
    for state in range(solver.n_state):
        # Function to integrate to get state
        basis_functions = solver.basis._build_basis_functions()
        to_int = _func_weighted_sum(basis_functions,
                                    dx[state])
        # State at instant tau
        states_tau[state] = x0[state] + solver.time_step * \
            solver.quad.integrate(to_int, 0, tau)
    return states_tau


def _compute_gradient(solver, x0, dx, tau):
    """Given dx and x0 for the frame, computes
    the gradient at a given time tau between 0 and 1

    Args:
        x0 (array): states at the begining of the frame.
            Size: [solver.n_state]
        dx (2D array): projected state flows.
            Size: [solver.n_state, solver.p_order]
        tau (float): time at which the estimation is needed

    Returns:
        array: values of the gradients. Size: [solver.n_state]
    """
    # compute state at instants tau
    states_tau = compute_state(solver, x0, dx, tau)
    states_tau = states_tau.reshape(solver.n_state)
    return solver.gradients(states_tau).flatten()


def check_proj_gradient(solver, x0, dx, tau):
    """Computes uprojected gradients value as well at projected ones and sum
    of projected values on all basis functions for times tau

    Args:
        x0 (array): initial state
        dx (2D array): projected flows
        tau (array): times at which an estimation is wanted
            (between 0 and 1)

    Returns:
        2D array: Unprojected gradients values.
            Size: [len(tau), solver.n_state]
        3D array: Projected gradients values.
            Size: [len(tau), solver.n_state, solver.p_order]
        2D array: Sum of projected gradients values.
            Size: [len(tau), solver.n_state]
    """
    # Compute intermediate states for integration
    int_states = _compute_intermediate_states(solver, x0, dx)
    # Compute projected gradient values
    proj_gradient_coeffs = _proj_gradient(solver, int_states)
    # Number of point to evaluate
    Ntau = len(tau)

    # Arrays to store results
    unproj_gradients = np.zeros((Ntau, solver.n_state))

    for i, ti in enumerate(tau):
        unproj_gradients[i] = _compute_gradient(solver, x0, dx, ti)
    basis_eval = solver.basis.evaluate_all(tau)
    basis_eval = np.repeat(basis_eval[:, np.newaxis, :],
                           solver.n_state,
                           axis=1)
    proj_gradient_coeffs2 = np.repeat(proj_gradient_coeffs[np.newaxis, :, :],
                                      len(tau),
                                      axis=0)
    proj_gradients = basis_eval * proj_gradient_coeffs2
    sum_proj_gradients = np.sum(proj_gradients, axis=2)
    # Returns three arrays of values
    return unproj_gradients, proj_gradient_coeffs, sum_proj_gradients


# Regularization step

def regularize(solver, x0, x1, proj_coeffs):
    """Apply the regularization step on the projected system.

    Args:
        solver (object): instance of the rpm solver class
        x0 (array): states at begining of frame. Size: [solver.n_state]
        x1 (array): states at end of frame. Size: [solver.n_state]
        proj_coeffs (array): Coefficients of the projection step.
                             Size: [solver.full_size_unproj, solver.p_order]

    Returns:
        array: regularization coefficients. Size: [solver.full_size_unproj,
            solver.k_order]
    """
    # Separation of coefficients
    p_coeffs_H = proj_coeffs[0:solver.n_state]

    # Initialization
    regul_order = int(solver.k_order / 2)
    regul_0 = np.zeros((regul_order, solver.n_state))
    regul_1 = np.zeros((regul_order, solver.n_state))

    SH = solver.S[0:solver.n_state, 0:solver.n_state]
    for order in range(regul_order):
        if order == 0:
            # Values of flows
            f_tau0 = solver.basis.get_proj_diff_0(p_coeffs_H, order)
            f_tau1 = solver.basis.get_proj_diff_1(p_coeffs_H, order)

            # Values of gradients
            dH_tau0 = solver.gradients(x0)
            dH_tau1 = solver.gradients(x1)

            # Computation of regularization coefficients
            diff_f_0 = SH @ dH_tau0
            regul_0[order, :] = diff_f_0 \
                - f_tau0
            diff_f_1 = SH @ dH_tau1
            regul_1[order, :] = diff_f_1\
                - f_tau1
        else:
            # Values of flows
            f_tau0 = solver.basis.get_proj_diff_0(p_coeffs_H, order)
            f_tau1 = solver.basis.get_proj_diff_1(p_coeffs_H, order)

            # Values of hessians
            dH_tau0 = solver.hessian(x0)
            dH_tau1 = solver.hessian(x1)

            diff2_f_0 = np.dot(SH @ dH_tau0,
                               diff_f_0)
            diff2_f_1 = np.dot(SH @ dH_tau1,
                               diff_f_1)

            # Regularization coeffs computation
            regul_0[order, :] = diff2_f_0 \
                - f_tau0
            regul_1[order, :] = diff2_f_1 \
                - f_tau1
    return np.concatenate((regul_0.T, regul_1.T), axis=1)


