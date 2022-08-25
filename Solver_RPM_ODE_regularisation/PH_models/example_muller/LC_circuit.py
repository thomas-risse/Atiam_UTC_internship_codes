import numpy as np
import sympy as sp
import rpm_module as rpm

# Functions presented here are used to build le conservative autonomous LC
# circuit used as exemple in Müller's Phd.  

def build_linear(C_0, L_0):
    #### Build a phs structure ####
    # Interconnexion matrix
    S = np.array([[0, -1],
                  [1, 0]])

    # Symbols declaration
    q, phi = sp.symbols('q, phi', real=True)
    # States variables
    states = [q, phi]
    # Additional parameters
    parameters = [C_0, L_0]

    # Hamiltonian expression

    def hamiltonian(q, phi):
        return (q)**2 / (2*C_0) + phi**2 / (2*L_0)

    H = hamiltonian(*states)

    # Number of constraints
    n_constraints = 0

    # String with informations about the phs
    about = "One capacitance and one inductance in a closed loop."

    # Create dictionnary containing the structure
    phs_struct = rpm.struct.build_struct_dict(S,
                                              H,
                                              states,
                                              parameters,
                                              n_constraints,
                                              about)

    # Add the L matrix to desribe hamiltonian of linear systems
    phs_struct["L"] = np.array([[1/C_0, 0],
                                [0, 1/L_0]])

    return phs_struct


def build_non_linear(C_0, L_0, I_s=1):
    #### Build a phs structure ####
    # Interconnexion matrix
    S = np.array([[0, -1],
                  [1, 0]])

    # Symbols declaration
    q, phi, C0, L0, IS = sp.symbols('q, phi, C_0, L_0, I_s', real=True)
    # States variables
    states = [q, phi]
    # Additional parameters
    parameters = [C0, L0, IS]

    # Hamiltonian expression
    def hamiltonian(q, phi):
        return (q)**2 / (2*C0) + L0*IS**2*sp.ln(sp.cosh(phi/(L0*IS)))

    H = hamiltonian(*states)

    # Number of constraints
    n_constraints = 0

    # String with informations about the phs
    about = "One capacitance and one inductance in a closed loop."

    # Create dictionnary containing the structure
    phs_struct = rpm.struct.build_struct_dict(S,
                                              H,
                                              states,
                                              parameters,
                                              n_constraints,
                                              about)

    phs_struct["H"] = phs_struct["H"].subs([(C0, C_0), (L0, L_0), (IS, I_s)])

    return phs_struct