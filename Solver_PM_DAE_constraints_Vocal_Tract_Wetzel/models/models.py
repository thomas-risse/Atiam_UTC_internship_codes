from cgi import parse_multipart
from os import stat
from importlib_metadata import FreezableDefaultDict
import numpy as np
import scipy as sp
# This file contains the template class to store phs models in a
# suitable way for the solver, aswell as some examples.


class PhsModel:
    def __init__(self):
        """Initialize the class with necesssary
        arguments
        """
        # Number of state variables, dissipative elements,
        # constraints equations and input/output pair
        self.n_state = 2  # state
        self.n_diss = 0  # dissipative elements
        self.n_cons = 0  # constraints
        self.n_io = 0  # input/output pairs
        # Full size of the system
        self.full_size = self.n_state +\
            self.n_diss +\
            self.n_cons +\
            self.n_io

        # Interconnection matrix as a 2D numpy array
        self.S = self._init_S()

        # Interconnection matrix splitted
        self._split_S()

    def H(self, states):
        """Function returning the values of the hamiltonian
        correponding to the different given values of the state.

        Args:
            states (array): states values. Size: [len(states), self.n_state]
        Returns:
            array: hamiltonian values. Size: [len(states)]
        """
        return np.ones(len(states))

    def grad_H(self, states):
        """Function returning the values of the gradients of the hamiltonian
        for the given states values.

        Args:
            states (array): states values. Size: [len(states), self.n_state]

        Returns:
            array: gradient values. Size: [len(states), self.n_state]
        """
        return np.ones((len(states), self.n_state))

    def hess_H(self, states):
        """Function returning the values of the hessian of the hamiltonian
        for the given states values.

        Args:
            states (array): states values. Size: [len(states), self.n_state]

        Returns:
            array: gradient values.
                Size: [len(states), self.n_state, self.n_state]
        """
        return np.ones((len(states), self.n_state, self.n_state))

    def zw(self, w, states):
        """Function returning the values of the dissipation laws z(w, x)
        for the given dissipation variables w and states values.

        Args:
            w (array): dissipation variables values.
                Size: [len(w), self.n_diss]
            states (array): states values. Size: [len(w), self.n_state]

        Returns:
            array: values of z(w, x).
                Size: [len(w), self.n_diss]
        """
        return np.ones((len(w), self.n_diss))

    def grad_zw_w(self, w, states):
        """Function returning the gradient of the dissipation laws z(w, x)
        for the given dissipation variables w and states values with
        respect to w.

        Args:
            w (array): dissipation variables values.
                Size: [len(w), self.n_diss]
            states (array): states values. Size: [len(w), self.n_state]

        Returns:
            array: gradient of z(w, x).
                Size: [len(w), self.n_diss, self.n_diss]
        """
        return np.ones((len(w), self.n_diss, self.n_diss))

    def grad_zw_x(self, w, states):
        """Function returning the gradient of the dissipation laws z(w, x)
        for the given dissipation variables w and states values with
        respect to the state x.

        Args:
            w (array): dissipation variables values.
                Size: [len(w), self.n_diss]
            states (array): states values. Size: [len(w), self.n_state]

        Returns:
            array: gradient of z(w, x).
                Size: [len(w), self.n_diss, self.n_state]
        """
        return np.ones((len(w), self.n_diss, self.n_state))

    def hess_zw(self, w, states):
        """Function returning the hessian of the dissipation laws z(w)
        for the given dissipation variables w and states values.

        Args:
            w (array): dissipation variables values.
                Size: [len(w), self.n_diss]
            states (array): states values. Size: [len(w), self.n_state]

        Returns:
            array: hessian of z(w, x).
                Size: [len(w), self.n_diss, self.n_state+self.n_diss]
        """
        return np.ones((len(w), self.n_diss,
                        self.n_state+self.n_diss, self.n_state+self.n_diss))

    def _init_S(self):
        """Initializes interconnection matrix.
        """
        self.S = np.zeros((self.full_size, self.full_size))

    def _set_Jxx(self, Jxx):
        """Utility function to build S. Sets Jxx.

        Args:
            Jxx (array): Jxx. Size : [self.n_state, self.n_state]
        """
        self.S[0:self.n_state, 0:self.n_state] = Jxx

    def _set_Jww(self, Jww):
        """Utility function to build S. Sets Jww.

        Args:
            Jww (array): Jww. Size : [self.n_diss, self.n_diss]
        """
        i0 = self.n_state
        i1 = self.n_state+self.n_diss
        self.S[i0:i1, i0:i1] = Jww

    def _set_Jll(self, Jll):
        """Utility function to build S. Sets Jll.

        Args:
            Jll (array): Jll. Size : [self.n_diss, self.n_diss]
        """
        i0 = self.n_state+self.n_diss
        i1 = self.n_state+self.n_diss+self.n_cons
        self.S[i0:i1, i0:i1] = Jll

    def _set_Jyu(self, Jyu):
        """Utility function to build S. Sets Jyu.

        Args:
            Jyu (array): Jyu. Size : [self.n_io, self.n_io]
        """
        i0 = self.n_state+self.n_diss+self.n_cons
        self.S[i0:, i0:] = Jyu

    def _set_Jxw(self, Jxw):
        """Utility function to build S. Sets Jxw and Jwx.

        Args:
            Jxw (array): Jxw. Size : [self.n_state, self.n_diss]
        """
        self.S[0:self.n_state, self.n_state:self.n_state+self.n_diss] = Jxw
        self.S[self.n_state:self.n_state+self.n_diss, 0:self.n_state] = -Jxw.T

    def _set_Jxl(self, Jxl):
        """Utility function to build S. Sets Jxl and Jlx.

        Args:
            Jxl (array): Jxl. Size : [self.n_state, self.n_cons]
        """
        i0 = self.n_state+self.n_diss
        i1 = self.n_state+self.n_diss+self.n_cons
        self.S[0:self.n_state, i0:i1] = Jxl
        self.S[i0:i1, 0:self.n_state] = -Jxl.T

    def _set_Jxu(self, Jxu):
        """Utility function to build S. Sets Jxu and Jux.

        Args:
            Jxu (array): Jxu. Size : [self.n_state, self.n_io]
        """
        i0 = self.n_state+self.n_diss+self.n_cons
        self.S[0:self.n_state, i0:] = Jxu
        self.S[i0:, 0:self.n_state] = -Jxu.T

    def _set_Jwl(self, Jwl):
        """Utility function to build S. Sets Jwl and Jlw.

        Args:
            Jwl (array): Jwl. Size : [self.n_diss, self.n_cons]
        """
        i0 = self.n_state+self.n_diss
        i1 = i0 + self.n_cons
        self.S[self.n_state:i0, i0:i1] = Jwl
        self.S[i0:i1, self.n_state:i0] = -Jwl.T

    def _set_Jwu(self, Jwu):
        """Utility function to build S. Sets Jwu and Juw.

        Args:
            Jwu (array): Jwu. Size : [self.n_diss, self.n_io]
        """
        i0 = self.n_state+self.n_diss
        i1 = i0 + self.n_cons
        self.S[self.n_state:i0, i1:] = Jwu
        self.S[i1:, self.n_state:i0] = -Jwu.T

    def _set_Jlu(self, Jlu):
        """Utility function to build S. Sets Jlu and Jul.

        Args:
            Jlu (array): Jlu. Size : [self.n_cons, self.n_io]
        """
        i0 = self.n_state+self.n_diss
        i1 = i0 + self.n_cons
        self.S[i0:i1, i1:] = Jlu
        self.S[i1:, i0:i1] = -Jlu.T

    def _split_S(self):
        """Splits the interconnection matrix into
        diferent parts.
        """
        self.Sxx = self.S[0:self.n_state, 0:self.n_state]
        self.Sxw = self.S[0:self.n_state,
                          self.n_state:self.n_state+self.n_diss]
        self.Sxl = self.S[0:self.n_state,
                          self.n_state+self.n_diss:
                          self.n_state+self.n_diss+self.n_cons]
        self.Sxu = self.S[0:self.n_state,
                          self.n_state+self.n_diss+self.n_cons:]

        self.Swx = self.S[self.n_state:self.n_state+self.n_diss,
                          0:self.n_state]
        self.Sww = self.S[self.n_state:self.n_state+self.n_diss,
                          self.n_state:self.n_state+self.n_diss]
        self.Swu = self.S[self.n_state:self.n_state+self.n_diss,
                          self.n_state+self.n_diss+self.n_cons:]

        self.Slx = self.S[self.n_state+self.n_diss:
                          self.n_state+self.n_diss+self.n_cons,
                          0:self.n_state]
        self.Slu = self.S[self.n_state+self.n_diss:
                          self.n_state+self.n_diss+self.n_cons,
                          self.n_state + self.n_diss + self.n_cons:]

        self.Syx = self.S[self.n_state+self.n_diss+self.n_cons:,
                          0:self.n_state]
        self.Syl = self.S[self.n_state+self.n_diss+self.n_cons:,
                          self.n_state+self.n_diss:
                          self.n_state+self.n_diss+self.n_cons]

        self.Syw = self.S[self.n_state+self.n_diss+self.n_cons:,
                          self.n_state:self.n_state+self.n_diss]
        self.Syu = self.S[self.n_state+self.n_diss+self.n_cons:,
                          self.n_state+self.n_diss+self.n_cons:]

    def build_p_struct(self, p_order):
        """Builds projected interconnection matrices given
        the projection order.

        Args:
            p_order (int): projection order
        """
        # Computes projected system sizes
        # Number of state variables, dissipative elements,
        # constraints equations and input/output pair
        self.proj_n_state = p_order * self.n_state
        self.proj_n_diss = p_order * self.n_diss
        self.proj_n_cons = p_order * self.n_cons
        self.proj_n_io = p_order * self.n_io
        # Full size of the system
        self.proj_full_size = self.proj_n_state +\
            self.proj_n_diss +\
            self.proj_n_cons +\
            self.proj_n_io

        # Index of first element of each part
        self.proj_i_0 = 0

        self.proj_i_1 = self.proj_n_state
        self.proj_i_2 = self.proj_n_state +\
            self.proj_n_diss

        self.proj_i_3 = self.proj_n_state +\
            self.proj_n_diss + self.proj_n_cons

        # Projected S
        self.proj_S = np.kron(self.S, np.eye(p_order))

        self.proj_Sxx = self.proj_S[:self.proj_i_1,
                                    :self.proj_i_1]
        self.proj_Sxw = self.proj_S[:self.proj_i_1,
                                    self.proj_i_1:self.proj_i_2]
        self.proj_Sxl = self.proj_S[:self.proj_i_1,
                                    self.proj_i_2:self.proj_i_3]
        self.proj_Sxu = self.proj_S[:self.proj_i_1,
                                    self.proj_i_3:]

        self.proj_Swx = self.proj_S[self.proj_i_1:self.proj_i_2,
                                    :self.proj_i_1]
        self.proj_Sww = self.proj_S[self.proj_i_1:self.proj_i_2,
                                    self.proj_i_1:self.proj_i_2]
        self.proj_Swu = self.proj_S[self.proj_i_1:self.proj_i_2,
                                    self.proj_i_3:]

        self.proj_Slx = self.proj_S[self.proj_i_2:self.proj_i_3,
                                    :self.proj_i_1]
        self.proj_Slu = self.proj_S[self.proj_i_2:self.proj_i_3,
                                    self.proj_i_3:]

        self.proj_Syx = self.proj_S[self.proj_i_3:,
                                    :self.proj_i_1]
        self.proj_Syw = self.proj_S[self.proj_i_3:,
                                    self.proj_i_1:self.proj_i_2]
        self.proj_Syl = self.proj_S[self.proj_i_3:,
                                    self.proj_i_2:self.proj_i_3]
        self.proj_Syu = self.proj_S[self.proj_i_3:,
                                    self.proj_i_3::]

    def adim(self, tref, Href, Mref, Wref, Zref, Uref, Yref):
        """Modifies the PHS to adimension it with given values.

        Args:
            tref (float): time reference
            Href (float): energy reference
            Mref (array): states references. Size: [self.n_state]
            Wref (aray): dissipation references.
                Size: [self.n_diss]
            Zref (array): dissipation laws references.
                Size: [self.n_diss]
            Uref (array): Input references. Size: [self.n_io]
            Yref (array): Output references. Size: [self.n_io]
        """
        self.tref = tref
        self.Href = Href
        self.Mref = Mref
        self.Wref = Wref
        self.Zref = Zref
        self.Uref = Uref
        self.Yref = Yref
        # New interconnexion
        Fref = np.zeros((self.full_size))
        Fref[:self.n_state] = Mref/tref
        i0 = self.n_state
        i1 = self.n_diss + self.n_state
        Fref[i0:i1] = Wref
        i0 = i1
        Fref[i0:] = Yref

        Eref = np.zeros((self.full_size))
        Eref[:self.n_state] = 1/Mref*Href
        i0 = self.n_state
        i1 = self.n_state + self.n_diss
        Eref[i0:i1] = Zref
        i0 = i1
        Eref[i0:] = Uref

        self.S = np.diag(1/Fref) @ \
            self.S @ np.diag(Eref)

        # New H
        Hdim = self.H

        def H(states):
            return 1/self.Href * Hdim(Mref * states)
        self.H = H

        gradHdim = self.grad_H

        def grad_H(states):
            return 1/self.Href * Mref * \
                gradHdim(Mref * states)
        self.grad_H = grad_H

        hessHdim = self.hess_H

        def hess_H(states):
            return 1/self.Href * np.diag(Mref*Mref) @\
                hessHdim(Mref * states)
        self.hess_H = hess_H

        # New z
        zdim = self.zw

        def zw(w, states):
            return 1/Zref * zdim(Wref*w, Mref*states)
        if self.n_diss != 0:
            self.zw = zw

        grad_zw_w_dim = self.grad_zw_w

        def grad_zw_w(w, states):
            z0 = np.swapaxes(Wref * grad_zw_w_dim(Wref*w, Mref*states), 1, 2)
            zw = 1/Zref * z0
            return np.swapaxes(zw, 1, 2)
        if self.n_diss != 0:
            self.grad_zw_w = grad_zw_w

        grad_zw_x_dim = self.grad_zw_x

        def grad_zw_x(w, states):
            z0 = np.swapaxes(Mref * grad_zw_x_dim(Wref*w, Mref*states), 1, 2)
            zw = 1/Zref * z0
            return np.swapaxes(zw, 1, 2)
        if self.n_diss != 0:
            self.grad_zw_x = grad_zw_x

        self._split_S()

    def find_0_grad(self, Href, xtilde):
        """Function used to find the value of Mref such
        that Mref/Href*gradH(xtilde) = 1.

        Args:
            Href (float): Reference energy.
            xtilde (array): Estimated amplitude of the dimensionned states.

        Returns:
            array: Mref such that Mref/Href*gradH(xtilde) = 1
        """
        grad_xtilde = self.grad_H(xtilde)[0, :]
        return Href/grad_xtilde

    def get_parameters(self):
        """Ouputs a dictionnary with some informations about the model.

        Returns:
            dict: dictionnary containing informations
        """
        structure_parameters = {'n_states': self.n_state,
                                'n_diss': self.n_diss,
                                'n_cons': self.n_cons,
                                'n_io': self.n_io,
                                'S': self.S
                                }
        parameters = {'Structure': structure_parameters,
                      'Name': self.__class__.__name__
                      }
        return parameters


class LinearLC(PhsModel):
    def __init__(self, C0, L0):
        # Number of state variables, dissipative elements,
        # constraints equations and input/output pair
        self.n_state = 2  # state
        self.n_diss = 0  # dissipative elements
        self.n_cons = 0  # constraints
        self.n_io = 0  # input/output pairs
        # Full size of the system
        self.full_size = 2

        # Interconnection matrix
        self.S = np.array([[0, -1],
                           [1, 0]])
        self._split_S()

        # Hamiltonian parameters
        self.C0 = C0
        self.L0 = L0

    def H(self, states):
        return 0.5 * (states[:, 0]**2/self.C0 +
                      states[:, 1]**2/self.L0)

    def grad_H(self, states):
        states[:, 0] = states[:, 0]/self.C0
        states[:, 1] = states[:, 1]/self.L0
        return states

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))
        hess[:, 0, 0] = 1/self.C0
        hess[:, 1, 1] = 1/self.L0
        return hess


class LinearRLC(PhsModel):
    def __init__(self, C0, L0, R0):
        # Number of state variables, dissipative elements,
        # constraints equations and input/output pair
        self.n_state = 2  # state
        self.n_diss = 1  # dissipative elements
        self.n_cons = 0  # constraints
        self.n_io = 1  # input/output pairs
        # Full size of the system
        self.full_size = 4

        # Interconnection matrix
        self.S = np.array([[0, -1, 1, 0],
                           [1, 0, 0, 0],
                           [-1, 0, 0, -1],
                           [0, 0, 1, 0]])
        self._split_S()

        # Hamiltonian parameters
        self.C0 = C0
        self.L0 = L0

        # Dissipation parameters
        self.R0 = R0

    def H(self, states):
        return 0.5 * (states[:, 0]**2/self.C0 +
                      states[:, 1]**2/self.L0)

    def grad_H(self, states):
        states[:, 0] = states[:, 0]/self.C0
        states[:, 1] = states[:, 1]/self.L0
        return states

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))
        hess[:, 0, 0] = 1/self.C0
        hess[:, 1, 1] = 1/self.L0
        return hess

    def zw(self, w, states):
        return w/self.R0

    def grad_zw_w(self, w, states):
        return np.ones((len(w), self.n_diss, self.n_diss)) / self.R0

    def grad_zw_x(self, w, states):
        return np.zeros((len(w), self.n_diss, self.n_state))


class NonLinearRLC(LinearRLC):
    def __init__(self, C0, E0, phi0, R0):
        super().__init__(C0, 0, R0)
        self.E0 = E0
        self.phi0 = phi0

    def H(self, states):
        return 0.5 * states[:, 0]**2/self.C0 +\
            self.E0 * np.log(np.cosh(states[:, 1] / self.phi0))

    def grad_H(self, states):
        grad = np.zeros_like(states)
        grad[:, 0] = states[:, 0]/self.C0
        grad[:, 1] = self.E0 / self.phi0 * np.tanh(states[:, 1] / self.phi0)
        return grad

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))
        hess[:, 0, 0] = 1/self.C0
        hess[:, 1, 1] = self.E0 / self.phi0**2 *\
            (1 - np.tanh(states[:, 1] / self.phi0)**2)
        return hess


class LinearTriangle(PhsModel):
    def __init__(self, C0, C1, L0):
        # Number of state variables, dissipative elements,
        # constraints equations and input/output pair
        self.n_state = 3  # state
        self.n_diss = 0  # dissipative elements
        self.n_cons = 1  # constraints
        self.n_io = 0  # input/output pairs
        # Full size of the system
        self.full_size = 4

        # Interconnection matrix
        self.S = np.array([[0, 0, -1, -1],
                           [0, 0, 0, -1],
                           [1, 0, 0, 0],
                           [1, 1, 0, 0]])
        self._split_S()

        # Hamiltonian parameters
        self.C0 = C0
        self.C1 = C1
        self.L0 = L0

    def H(self, states):
        return 0.5 * (states[:, 0]**2/self.C0 +
                      states[:, 1]**2/self.C1 +
                      states[:, 2]**2/self.L0)

    def grad_H(self, states):
        states[:, 0] = states[:, 0]/self.C0
        states[:, 1] = states[:, 1]/self.C1
        states[:, 2] = states[:, 2]/self.L0
        return states

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))
        hess[:, 0, 0] = 1/self.C0
        hess[:, 1, 1] = 1/self.C1
        hess[:, 2, 2] = 1/self.L0
        return hess


class AdimTest(PhsModel):
    """This system is a MSD oscillator with a non-linear spring
    and an amplitude dependant dissipation law. It is used to test
    the adimensionment.
    """

    def __init__(self, M, k1, k2, alpha):
        # Number of state variables, dissipative elements,
        # constraints equations and input/output pair
        self.n_state = 2  # state
        self.n_diss = 1  # dissipative elements
        self.n_cons = 0  # constraints
        self.n_io = 0  # input/output pairs
        # Full size of the system
        self.full_size = 3

        # Interconnection matrix
        self.S = np.array([[0, -1, -1],
                           [1, 0, 0],
                           [1, 0, 0]])
        self._split_S()

        # Hamiltonian parameters
        self.M = M
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha

    def H(self, states):
        return 0.5 * (states[:, 0]**2/self.M +
                      states[:, 1]**2*self.k1 + states[:, 1]**4*self.k2)

    def grad_H(self, states):
        gradH = np.zeros_like(states)
        gradH[:, 0] = states[:, 0]/self.M
        gradH[:, 1] = states[:, 1]*self.k1 + 2*states[:, 1]**3*self.k2
        return gradH

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))
        hess[:, 0, 0] = 1/self.M
        hess[:, 1, 1] = self.k1 + 6*states[:, 1]**2*self.k2
        return hess

    def zw(self, w, states):
        zw = np.zeros_like(w)
        zw[:, 0] = self.alpha*w[:, 0] * np.abs(states[:, 1])
        return zw

    def grad_zw_w(self, w, states):
        zw = np.zeros((len(w), self.n_diss, self.n_diss))
        zw[:, 0, 0] = self.alpha * np.abs(states[:, 1])
        return zw

    def grad_zw_x(self, w, states):
        zw = np.zeros((len(w), self.n_diss, self.n_state))
        zw[:, 0, 0] = 0
        zw[:, 0, 1] = self.alpha*w[:, 0] * np.sign(states[:, 1])
        return zw
