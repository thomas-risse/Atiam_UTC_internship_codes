from models.models import PhsModel
import numpy as np


class LinearLLC(PhsModel):
    def __init__(self, L0, L1, C0):
        """Initialize the class with necesssary
        arguments
        """
        # Number of state variables, dissipative elements,
        # constraints equations and input/output pair
        self.n_state = 3  # state
        self.n_diss = 0  # dissipative elements
        self.n_cons = 1  # constraints
        self.n_io = 0  # input/output pairs
        # Full size of the system
        self.full_size = self.n_state +\
            self.n_diss +\
            self.n_cons +\
            self.n_io

        # Interconnection matrix as a 2D numpy array
        self.S = np.array([[0, 0, -1, -1],
                           [0, 0, 0, 1],
                           [1, 0, 0, 0],
                           [1, -1, 0, 0]])

        # Interconnection matrix splitted
        self._split_S()

        # Hamiltonian parameters
        self.L0 = L0
        self.L1 = L1
        self.C0 = C0

    def H(self, states):
        return 0.5 * (states[:, 0]**2/self.L0 +
                      states[:, 1]**2/self.L1 +
                      states[:, 2]**2/self.C0)

    def grad_H(self, states):
        grad = np.zeros_like(states)
        grad[:, 0] = states[:, 0]/self.L0
        grad[:, 1] = states[:, 1]/self.L1
        grad[:, 2] = states[:, 2]/self.C0
        return grad

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))
        hess[:, 0, 0] = 1/self.L0
        hess[:, 1, 1] = 1/self.L1
        hess[:, 2, 2] = 1/self.C0
        return hess


class NonLinearLLC1(LinearLLC):
    def __init__(self, E0, phi0, E1, phi1, C0):
        super().__init__(0, 0, C0)
        self.E0 = E0
        self.E1 = E1
        self.phi0 = phi0
        self.phi1 = phi1

    def H(self, states):
        return self.E1 * np.log(np.cosh(states[:, 0] / self.phi1)) +\
               self.E0 * np.log(np.cosh(states[:, 1] / self.phi0)) +\
               0.5 * states[:, 2]**2/self.C0

    def grad_H(self, states):
        grad = np.zeros_like(states)
        grad[:, 0] = self.E0 / self.phi0 * np.tanh(states[:, 0] / self.phi0)
        grad[:, 1] = self.E1 / self.phi1 * np.tanh(states[:, 1] / self.phi1)
        grad[:, 2] = states[:, 2]/self.C0
        return grad

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))
        hess[:, 0, 0] = self.E0 / self.phi0**2 *\
            (1 - np.tanh(states[:, 0] / self.phi0)**2)
        hess[:, 1, 1] = self.E1 / self.phi1**2 *\
            (1 - np.tanh(states[:, 1] / self.phi1)**2)
        hess[:, 1, 1] = 1/self.C0
        return hess


class NonLinearLLC2(LinearLLC):
    def __init__(self, L0, E0, phi0, C0):
        super().__init__(L0, 0, C0)
        self.E0 = E0
        self.phi0 = phi0

    def H(self, states):
        return 0.5 * states[:, 0]**2/self.L0 +\
               self.E0 * np.log(np.cosh(states[:, 1] / self.phi0)) +\
               0.5 * states[:, 2]**2/self.C0

    def grad_H(self, states):
        grad = np.zeros_like(states)
        grad[:, 0] = states[:, 0]/self.L0
        grad[:, 1] = self.E0 / self.phi0 * np.tanh(states[:, 1] / self.phi0)
        grad[:, 2] = states[:, 2]/self.C0
        return grad

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))
        hess[:, 0, 0] = 1/self.L0
        hess[:, 1, 1] = self.E0 / self.phi0**2 *\
            (1 - np.tanh(states[:, 1] / self.phi0)**2)
        hess[:, 1, 1] = 1/self.C0
        return hess
