import sympy as sp
import numpy as np


def build_H_np(H, states):
    """Function used to generate a numpy compatible Hamiltonian function from the
    sympy version of it.

    Args:
        H (sympy expression): hamiltonian
        states (list): list of states

    Returns:
        function: numpy hamiltonian function
    """
    lambda_H = sp.lambdify(states, H)

    def H_np(x):
        """Hamiltonian function.

        Args:
            x (array): array of states values of shape [N_points, n_states].

        Returns:
            array: values of the hamiltonian evaluated at given points. 
        """
        if x.ndim == 1:
            return lambda_H(*x)
        else:
            results = np.zeros((len(x)))
            for i, xi in enumerate(x):
                results[i] = lambda_H(*xi)
            return results
    return H_np


def compute_diffs(H, states, max_order, mode="lambda"):
    """Function used to generate numpy compatible functions 
    to compute derivatives of the hamiltonian wrt the states from the
    sympy version of H.

    Args:
        H (sympy expression): hamiltonian
        states (list): list of states

    Returns:
        array: array of functions used to evaluate the derivatives of H.
    """
    diffs = [0]*max_order
    diffs[0] = sp.tensor.array.derive_by_array(H, states)

    for i in range(1, max_order):
        diffs[i] = sp.tensor.array.derive_by_array(diffs[i-1], states)

    f_diffs = [sp.lambdify(states, diffs[i])
               for i in range(max_order)]

    # Make sure that returned arrays are numpy arrays
    # and that an input numpy array containing the state is accepted
    def build_f(order):
        l_dims = [len(states) for i in range(order+1)]

        def f(x):
            if x.ndim == 1:
                return np.array(f_diffs[order](*x))
            else:
                results = np.zeros((len(x), *l_dims))
                for i, xi in enumerate(x):
                    results[i] = np.array(f_diffs[order](*xi))
                return results
        return f

    f_diffs_np = [build_f(i) for i in range(max_order)]
    if mode == "lambda":
        return f_diffs_np
    else:
        return diffs
