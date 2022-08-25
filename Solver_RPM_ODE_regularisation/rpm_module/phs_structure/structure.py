
import pickle

"""This file provides tools to store PHS structures in order to be able to
re-use them when needed.
"""


def build_struct_dict(S, H, states, parameters, n_constraints, about=""):
    """Build and returns a dictionnary containing the descritpion
    of the PHS.

    Args:
        S (array): interconnexion matrix
        H (sympy expression): hamiltonian expression with variables
                              states and parameters.
        states (array): array of sympy variables (state variables)
        parameters (array): array of sympy variables (tunable parameters).
        n_constraints (int): number of constraints
        about (str, optional): Description of the system.
                               Defaults to "".

    Returns:
        dict: descritpion of the phs stored in a dict
    """
    phs_struct = {"S": S,
                  "H": H,
                  "States": states,
                  "Parameters": parameters,
                  "Constraints": n_constraints,
                  "About": about}
    return phs_struct


def store(phs_struct, filename):
    """Stores a PHS struct in a file with pickle.

    Args:
        phs_struct (dict): structure of the PHS
        filename (string): filename
    """
    with open(filename, 'wb') as f:
        pickle.dump(phs_struct, f)  # , indent=4, sort_keys=True, default=vars)


def load(filename):
    """Reads a PHS struct from a file (pickle format).

    Args:
        filename (string): name of the file to read

    Returns:
        dict: structure of the PHS
    """
    with open(filename, 'rb') as f:
        phs_struct = pickle.load(f)
    return phs_struct
