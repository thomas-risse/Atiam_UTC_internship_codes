from models.models import PhsModel
import numpy as np
import copy

""" THis file contains class definitions for 
the SHP models of the vocal tract proposed by Victor
Wetzel in his thesis.
"""

class SingleTract(PhsModel):
    """Single tract of the model without walls.
    """
    def __init__(self, l0, Sw):
        """Initialize the class with necesssary
        arguments.

        Args:
            l0 (float): semi length of the tract.
            Sw (float): surface 2*l0*L0
        """
        # Number of state variables, dissipative elements,
        # constraints equations and input/output pair
        self.n_state = 5  # state
        self.n_diss = 2  # dissipative elements
        self.n_cons = 0  # constraints
        self.n_io = 3  # input/output pairs
        # Full size of the system
        self.full_size = self.n_state +\
            self.n_diss +\
            self.n_cons +\
            self.n_io

        # Interconnection matrix as a 2D numpy array
        self.S = np.zeros((self.full_size, self.full_size))
        # Indexes of 1
        ind1 = [[0, 1, 1, 2, 3, 4, 5, 6, 9],
                [7, 3, 5, 8, 0, 2, 0, 2, 1]]
        self.S[ind1[0], ind1[1]] = 1
        self.S[ind1[1], ind1[0]] = -1

        # Interconnection matrix splitted
        self._split_S()

        # Hamiltonian parameters
        self.l0 = l0 # semi length of the tract
        self.Sw = Sw # surface 2*l0*L0

        # General parameters
        self.rho0 = 1.292  # kg.m-3
        self.gamma = 1.4
        self.P0 = 101325  # Pa

    def H(self, states):
        H1 = 0.5 / self.l0**2 * states[:, 3] *\
            (states[:, 0]**2 + states[:, 1]**2 - states[:, 0]*states[:, 1])
        H2 = 3 * states[:, 2]**2 / (2 * states[:, 3])
        rho_m = states[:, 3] / (self.Sw*states[:, 4])
        H3 = self.P0 * self.Sw * states[:, 4] *\
            (self.gamma/2 * (rho_m/self.rho0-1)**2 - 1)
        return H1 + H2 + H3

    def grad_H(self, states):
        grad = np.zeros_like(states)
        vl = states[:, 0]
        vr = states[:, 1]
        piy = states[:, 2]
        m = states[:, 3]
        h = states[:, 4]

        term1 = -1+m/(self.Sw*h*self.rho0)

        grad[:, 0] = (2*vl-vr)*m/(2*self.l0**2)
        grad[:, 1] = (2*vr-vl)*m/(2*self.l0**2)
        grad[:, 2] = 3*piy/m
        grad[:, 3] = (vl**2+vr**2-vl*vr) / (2 * self.l0**2)\
            - 3*piy**2/(2*m**2) + self.P0*self.gamma*term1/self.rho0
        grad[:, 4] = self.P0 * self.Sw * (self.gamma/2 * term1**2 - 1)\
            - self.P0*self.gamma*m*term1/(h*self.rho0)
        return grad

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))

        vl = states[:, 0]
        vr = states[:, 1]
        piy = states[:, 2]
        m = states[:, 3]
        h = states[:, 4]

        hess[:, 0, 0] = m / self.l0**2
        hess[:, 0, 1] = -m / (2*self.l0**2)
        hess[:, 0, 3] = (2*vl - vr) / (2*self.l0**2)

        hess[:, 1, 0] = -m / (2*self.l0**2)
        hess[:, 1, 1] = m / self.l0**2
        hess[:, 1, 3] = (2*vr - vl) / (2*self.l0**2)

        hess[:, 2, 2] = 3/m
        hess[:, 2, 3] = -3*piy/m**2

        hess[:, 3, 0] = (2*vl-vr)/(2*self.l0**2)
        hess[:, 3, 1] = (2*vr-vl)/(2*self.l0**2)
        hess[:, 3, 2] = -3*piy/m**2
        hess[:, 3, 3] = 3*piy**2 / m**3 + self.P0 * self.gamma /\
            (self.rho0**2 * self.Sw * h)
        hess[:, 3, 4] = -self.P0 * self.gamma * m /\
            ((self.rho0 * h)**2 * self.Sw)

        hess[:, 4, 3] = -self.P0 * self.gamma * m /\
            ((self.rho0 * h)**2 * self.Sw)
        hess[:, 4, 4] = self.P0*self.gamma*m**2 / (self.Sw*h**3*self.rho0**2)
        return hess

    def zw(self, w, states):
        zw = np.zeros_like(w)

        mdot = w[:, 0]
        vy = w[:, 1]

        piy = states[:, 2]
        m = states[:, 3]

        zw[:, 0] = vy*piy/m
        zw[:, 1] = -piy*mdot/m
        return zw

    def grad_zw_w(self, w, states):
        piy = states[:, 2]
        m = states[:, 3]

        grad_zw = np.zeros((len(w), self.n_diss, self.n_diss))
        grad_zw[:, 0, 1] = piy/m
        grad_zw[:, 1, 0] = -piy/m
        return grad_zw

    def grad_zw_x(self, w, states):
        mdot = w[:, 0]
        vy = w[:, 1]

        piy = states[:, 2]
        m = states[:, 3]

        grad_zw = np.zeros((len(w), self.n_diss, self.n_state))
        grad_zw[:, 0, 2] = vy/m
        grad_zw[:, 0, 3] = -vy*piy/m**2

        grad_zw[:, 1, 2] = -mdot/m
        grad_zw[:, 1, 3] = piy*mdot/m**2
        return grad_zw


class SingleTractWithWall(PhsModel):
    """Single tract of the model without walls.
    """
    def __init__(self, l0, Sw, k, r):
        """Initialize the class with necesssary
        arguments.

        Args:
            l0 (float): semi length of the tract.
            Sw (float): surface 2*l0*L0.
            k (float): stiffness of the wall.
            r (float): damping coefficient of the wall
        """
        # Number of state variables, dissipative elements,
        # constraints equations and input/output pair
        self.n_state = 6  # state
        self.n_diss = 3  # dissipative elements
        self.n_cons = 0  # constraints
        self.n_io = 3  # input/output pairs
        # Full size of the system
        self.full_size = self.n_state +\
            self.n_diss +\
            self.n_cons +\
            self.n_io

        # Interconnection matrix as a 2D numpy array
        self.S = np.zeros((self.full_size, self.full_size))
        # Indexes of 1
        ind1 = [[0, 1, 1, 3, 4, 2, 6, 7, 2, 11, 10, 10],
                [9, 3, 6, 0, 2, 5, 0, 2, 8, 1, 5, 8]]
        self.S[ind1[0], ind1[1]] = 1
        self.S[ind1[1], ind1[0]] = -1

        # Interconnection matrix splitted
        self._split_S()

        # Hamiltonian parameters
        self.l0 = l0
        self.Sw = Sw
        self.L0 = self.Sw / (2*self.l0)
        self.k = k
        self.r = r

        # General parameters
        self.rho0 = 1.292  # kg.m-3
        self.gamma = 1.4
        self.P0 = 101325  # Pa

    def H(self, states):
        H1 = 0.5 / self.l0**2 * states[:, 3] *\
            (states[:, 0]**2 + states[:, 1]**2 - states[:, 0]*states[:, 1])
        H2 = 3 * states[:, 2]**2 / (2 * states[:, 3])
        rho_m = states[:, 3] / (self.Sw*states[:, 4])
        H3 = self.P0 * self.Sw * states[:, 4] *\
            (self.gamma/2 * (rho_m/self.rho0-1)**2 - 1)
        Hmec = 0.5 * self.k * states[:, 5]**2 - self.P0*self.Sw * states[:, 5]
        return H1 + H2 + H3 + Hmec

    def grad_H(self, states):
        grad = np.zeros_like(states)
        vl = states[:, 0]
        vr = states[:, 1]
        piy = states[:, 2]
        m = states[:, 3]
        h = states[:, 4]

        term1 = -1+m/(self.Sw*h*self.rho0)
        rho = m/(self.Sw*h)

        grad[:, 0] = (2*vl-vr)*m/(2*self.l0**2)
        grad[:, 1] = (2*vr-vl)*m/(2*self.l0**2)
        grad[:, 2] = 3*piy/m
        grad[:, 3] = (vl**2+vr**2-vl*vr) / (2 * self.l0**2)\
            - 3*piy**2/(2*m**2) + self.P0*self.gamma*term1/self.rho0
        grad[:, 4] = self.P0*self.Sw*(self.gamma/2*(1-(rho/self.rho0)**2)-1)
        grad[:, 5] = self.k * states[:, 5] - self.P0*self.Sw
        return grad

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))

        vl = states[:, 0]
        vr = states[:, 1]
        piy = states[:, 2]
        m = states[:, 3]
        h = states[:, 4]

        hess[:, 0, 0] = m / self.l0**2
        hess[:, 0, 1] = -m / (2*self.l0**2)
        hess[:, 0, 3] = (2*vl - vr) / (2*self.l0**2)

        hess[:, 1, 0] = -m / (2*self.l0**2)
        hess[:, 1, 1] = m / self.l0**2
        hess[:, 1, 3] = (2*vr - vl) / (2*self.l0**2)

        hess[:, 2, 2] = 3/m
        hess[:, 2, 3] = -3*piy/m**2

        hess[:, 3, 0] = (2*vl-vr)/(2*self.l0**2)
        hess[:, 3, 1] = (2*vr-vl)/(2*self.l0**2)
        hess[:, 3, 2] = -3*piy/m**2
        hess[:, 3, 3] = 3*piy**2 / m**3 + self.P0 * self.gamma /\
            (self.rho0**2 * self.Sw * h)
        hess[:, 3, 4] = -self.P0 * self.gamma * m /\
            ((self.rho0 * h)**2 * self.Sw)

        hess[:, 4, 3] = -self.P0 * self.gamma * m /\
            ((self.rho0 * h)**2 * self.Sw)
        hess[:, 4, 4] = self.P0*self.gamma*m**2 / (self.Sw*h**3*self.rho0**2)

        hess[:, 5, 5] = self.k
        return hess

    def zw(self, w, states):
        zw = np.zeros_like(w)

        mdot = w[:, 0]
        vy = w[:, 1]

        piy = states[:, 2]
        m = states[:, 3]

        zw[:, 0] = vy*piy/m
        zw[:, 1] = -piy*mdot/m
        zw[:, 2] = self.r*w[:, 2]
        return zw

    def grad_zw_w(self, w, states):
        piy = states[:, 2]
        m = states[:, 3]

        grad_zw = np.zeros((len(w), self.n_diss, self.n_diss))
        grad_zw[:, 0, 1] = piy/m
        grad_zw[:, 1, 0] = -piy/m

        grad_zw[:, 2, 2] = self.r
        return grad_zw

    def grad_zw_x(self, w, states):
        mdot = w[:, 0]
        vy = w[:, 1]

        piy = states[:, 2]
        m = states[:, 3]

        grad_zw = np.zeros((len(w), self.n_diss, self.n_state))
        grad_zw[:, 0, 2] = vy/m
        grad_zw[:, 0, 3] = -vy*piy/m**2

        grad_zw[:, 1, 2] = -mdot/m
        grad_zw[:, 1, 3] = piy*mdot/m**2
        return grad_zw


class MultipleTracts(SingleTract):
    """Multiples tracts of the model without walls. All
    tracts are assumed to have same l0 and L0.
    """
    def __init__(self, N, l0, Sw):
        """Initializes the class.

        Args:
            N (int): number of tracts.
            l0 (float): semi length of one tract.
            Sw (float): Surface 2*L0*l0 of one tract.
        """
        self.N_tracts = N

        # We first populate the class with a single tract model
        super().__init__(l0, Sw)

        # We make a copy of the current assembled model to ease assembly
        self.tractl = SingleTract(l0, Sw)
        # Building of a single tract model to assemble iteratively
        # with the self model
        self.tractr = SingleTract(l0, Sw)

        self.connect(self.tractl, self. tractr)

        # Assembly loop
        for i in range(self.N_tracts-2):
            self.tractl = copy.copy(self)
            self.connect(self.tractl, self.tractr)

    def connect(self, tractl, tractr):
        """Assemble the current model with the tract phs
        given in tractr.

        Args:
            tractl (object): left track assembly
            tractr (object): right tract assembly
        """
        # New sizes
        full_n_state = tractl.n_state + tractr.n_state
        full_n_diss = tractl.n_diss + tractr.n_diss
        full_n_cons = tractl.n_cons + tractr.n_cons + 1
        full_n_io = tractl.n_io + tractr.n_io - 2

        # First, we build the new interconnection matrix
        new_Sxx = np.block(
            [[tractl.Sxx, np.zeros((tractl.n_state, tractr.n_state))],
             [np.zeros((tractr.n_state, tractl.n_state)), tractr.Sxx]])
        new_Sxw = np.block(
            [[tractl.Sxw, np.zeros((tractl.n_state, tractr.n_diss))],
             [np.zeros((tractr.n_state, tractl.n_diss)), tractr.Sxw]])
        # We need to add the new constraints
        newC = np.zeros((full_n_state, 1))
        ind_qr = tractl.n_state-4
        ind_ql = tractl.n_state
        newC[ind_qr] = -1
        newC[ind_ql] = 1
        new_Sxl = np.block(
            [[tractl.Sxl, newC[:tractl.n_state]],
             [np.zeros((tractr.n_state, tractl.n_cons)), newC[tractl.n_state:]]])
        # and to remove unnecessary inputs
        Sxul = tractl.Sxu[:, :-1]
        Sxur = tractr.Sxu[:, 1:]
        new_Sxu = np.block(
            [[Sxul, np.zeros((tractl.n_state, tractr.n_io-1))],
             [np.zeros((tractr.n_state, tractl.n_io-1)), Sxur]])

        # Others matrices are symmetrics or empty

        # Finally, we set the computed values
        self.n_state = full_n_state
        self.n_diss = full_n_diss
        self.n_cons = full_n_cons
        self.n_io = full_n_io
        self.full_size = self.n_state +\
            self.n_diss +\
            self.n_cons +\
            self.n_io

        self._init_S()
        self._set_Jxx(new_Sxx)
        self._set_Jxw(new_Sxw)
        self._set_Jxl(new_Sxl)
        self._set_Jxu(new_Sxu)
        self._split_S()

    def separate_states(self, states):
        states_l = states[:, :self.tractl.n_state]
        states_r = states[:, self.tractr.n_state:]
        return states_l, states_r

    def separate_diss(self, states):
        diss_l = states[:, :self.tractl.n_diss]
        diss_r = states[:, self.tractl.n_diss:]
        return diss_l, diss_r

    def H(self, states):
        states_l, states_r = self.separate_states(states)
        return self.tractl.H(states_l) + self.tractr.H(states_r)

    def grad_H(self, states):
        states_l, states_r = self.separate_states(states)
        grad_l = self.tractl.grad_H(states_l)
        grad_r = self.tractr.grad_H(states_r)
        return np.concatenate((grad_l, grad_r), axis=1)

    def hess_H(self, states):
        states_l, states_r = self.separate_states(states)
        hess_l = self.tractl.hess_H(states_l)
        hess_r = self.tractr.hess_H(states_r)
        hess = np.block(
            [[hess_l, np.zeros((len(states), self.tractl.n_state, self.tractr.n_state))],
             [np.zeros((len(states), self.tractr.n_state, self.tractl.n_state)), hess_r]])
        return hess

    def zw(self, w, states):
        states_l, states_r = self.separate_states(states)
        diss_l, diss_r = self.separate_diss(w)
        zw_l = self.tractl.zw(diss_l, states_l)
        zw_r = self.tractr.zw(diss_r, states_r)
        return np.concatenate((zw_l, zw_r), axis=1)

    def grad_zw_w(self, w, states):
        states_l, states_r = self.separate_states(states)
        diss_l, diss_r = self.separate_diss(w)
        grad_zw_l = self.tractl.grad_zw_w(diss_l, states_l)
        grad_zw_r = self.tractr.grad_zw_w(diss_r, states_r)
        grad_zw = np.block(
            [[grad_zw_l, np.zeros((len(states), self.tractl.n_diss, self.tractr.n_diss))],
             [np.zeros((len(states), self.tractr.n_diss, self.tractl.n_diss)), grad_zw_r]])
        return grad_zw

    def grad_zw_x(self, w, states):
        states_l, states_r = self.separate_states(states)
        diss_l, diss_r = self.separate_diss(w)
        grad_zw_l = self.tractl.grad_zw_x(diss_l, states_l)
        grad_zw_r = self.tractr.grad_zw_x(diss_r, states_r)
        grad_zw = np.block(
            [[grad_zw_l, np.zeros((len(states), self.tractl.n_diss, self.tractr.n_state))],
             [np.zeros((len(states), self.tractr.n_diss, self.tractl.n_state)), grad_zw_r]])
        return grad_zw


class MultipleTractsWithWalls(SingleTractWithWall):
    """Multiples tracts of the model with walls. All
    tracts are assumed to have same l0 and L0. Causlity of the input
    is not reversed, control is then done with input enthalpy.
    """
    def __init__(self, N, l0, Sw, k, r):
        """Initializes the class.

        Args:
            N (int): number of tracts.
            l0 (float): semi length of one tract.
            Sw (float): Surface 2*L0*l0 of one tract.
            k (float): stiffness of the walls.
            r (float): damping coefficient of the walls.
        """
        self.N_tracts = N

        # We first populate the class with a single tract model
        super().__init__(l0, Sw, k, r)

        # We make a copy of the current assembled model to ease assembly
        self.tractl = SingleTractWithWall(l0, Sw, k, r)
        # Building of a single tract model to assemble iteratively
        # with the self model
        self.tractr = SingleTractWithWall(l0, Sw, k, r)

        self.connect(self.tractl, self. tractr)

        # Assembly loop
        for i in range(self.N_tracts-2):
            self.tractl = copy.copy(self)
            self.connect(self.tractl, self.tractr)

        # Adding the radiation damping
        self.connect_with_radiation()

    def connect(self, tractl, tractr):
        """Assemble the current model with the tract phs
        given in tractr.

        Args:
            tractl (object): left track assembly
            tractr (object): right tract assembly
        """
        # New sizes
        full_n_state = tractl.n_state + tractr.n_state
        full_n_diss = tractl.n_diss + tractr.n_diss
        full_n_cons = tractl.n_cons + tractr.n_cons + 1
        full_n_io = tractl.n_io + tractr.n_io - 2

        # First, we build the new interconnection matrix
        new_Sxx = np.block(
            [[tractl.Sxx, np.zeros((tractl.n_state, tractr.n_state))],
             [np.zeros((tractr.n_state, tractl.n_state)), tractr.Sxx]])
        new_Sxw = np.block(
            [[tractl.Sxw, np.zeros((tractl.n_state, tractr.n_diss))],
             [np.zeros((tractr.n_state, tractl.n_diss)), tractr.Sxw]])
        # We need to add the new constraints
        newC = np.zeros((full_n_state, 1))
        ind_qr = tractl.n_state-5
        ind_ql = tractl.n_state
        newC[ind_qr] = -1
        newC[ind_ql] = 1
        new_Sxl = np.block(
            [[tractl.Sxl, newC[:tractl.n_state]],
             [np.zeros((tractr.n_state, tractl.n_cons)), newC[tractl.n_state:]]])
        # and to remove unnecessary inputs
        Sxul = tractl.Sxu[:, :-1]
        Sxur = tractr.Sxu[:, 1:]
        new_Sxu = np.block(
            [[Sxul, np.zeros((tractl.n_state, tractr.n_io-1))],
             [np.zeros((tractr.n_state, tractl.n_io-1)), Sxur]])

        # Others matrices are symmetrics or empty

        # Finally, we set the computed values
        self.n_state = full_n_state
        self.n_diss = full_n_diss
        self.n_cons = full_n_cons
        self.n_io = full_n_io
        self.full_size = self.n_state +\
            self.n_diss +\
            self.n_cons +\
            self.n_io

        self._init_S()
        self._set_Jxx(new_Sxx)
        self._set_Jxw(new_Sxw)
        self._set_Jxl(new_Sxl)
        self._set_Jxu(new_Sxu)
        self._split_S()

    def connect_with_radiation(self):
        """Adds a radiation condition at the en dof the vocal tract.

        Args:
            tractl (object): tract assembly
        """
        self.n_diss += 1
        self.n_io -= 1
        Sxx_old = self.Sxx
        Sxw_old = self.Sxw
        Sxl_old = self.Sxl
        Sxu_old = self.Sxu

        # New interconnection matrix
        self._init_S()
        self._set_Jxx(Sxx_old)
        self._set_Jxw(np.concatenate((Sxw_old, Sxu_old[:, -1:]), axis=1))
        self._set_Jxl(Sxl_old)
        self._set_Jxu(Sxu_old[:, :-1])
        self._split_S()

        # Change of zw
        zw_old = self.zw
        grad_zw_w_old = self.grad_zw_w
        grad_zw_x_old = self.grad_zw_x

        def zw(w, states):
            zw = np.zeros_like(w)
            # Old zw
            zw[:, :-1] = zw_old(w[:, :-1], states)
            # Turbulence zw
            zw[:, -1:] = self.zwrad(w[:, -1:], states[:, -self.tractr.n_state:])
            return zw
        self.zw = zw

        def grad_zw_w(w, states):
            grad_zw = np.zeros((len(w), self.n_diss, self.n_diss))
            # Old
            grad_zw[:, :-1, :-1] = grad_zw_w_old(w[:, :-1], states)
            # Turbulence
            grad_zw[:, -1:, -1:] = \
                self.grad_zwrad_w(w[:, -1:], states[:, -self.tractr.n_state:])
            return grad_zw
        self.grad_zw_w = grad_zw_w

        def grad_zw_x(w, states):
            grad_zw = np.zeros((len(w), self.n_diss, self.n_state))
            # Old
            grad_zw[:, :-1, :] = grad_zw_x_old(w[:, :-1], states)
            # Turbulence
            grad_zw[:, -1:, -self.tractr.n_state:] = \
                self.grad_zwrad_x(w[:, -1:], states[:, -self.tractr.n_state:])
            return grad_zw
        self.grad_zw_x = grad_zw_x

    def separate_states(self, states):
        states_l = states[:, :self.tractl.n_state]
        states_r = states[:, self.tractl.n_state:]
        return states_l, states_r

    def separate_diss(self, states):
        diss_l = states[:, :self.tractl.n_diss]
        diss_r = states[:, self.tractl.n_diss:]
        return diss_l, diss_r

    def H(self, states):
        states_l, states_r = self.separate_states(states)
        return self.tractl.H(states_l) + self.tractr.H(states_r)

    def grad_H(self, states):
        states_l, states_r = self.separate_states(states)
        grad_l = self.tractl.grad_H(states_l)
        grad_r = self.tractr.grad_H(states_r)
        return np.concatenate((grad_l, grad_r), axis=1)

    def hess_H(self, states):
        states_l, states_r = self.separate_states(states)
        hess_l = self.tractl.hess_H(states_l)
        hess_r = self.tractr.hess_H(states_r)
        hess = np.block(
            [[hess_l, np.zeros((len(states), self.tractl.n_state, self.tractr.n_state))],
             [np.zeros((len(states), self.tractr.n_state, self.tractl.n_state)), hess_r]])
        return hess

    def zwrad(self, w, states):
        """Dissipation law of te radiation condition

        Args:
            w (array): value of wturb
            states (array): states of the rightmost tract
        """
        zw = np.zeros_like(w)
        rho = states[:, 3] / (self.tractr.Sw*states[:, 4])
        zw[:, 0] = 1 / (2*rho) *\
            (w[:, 0]/(states[:, 4]*self.tractl.L0))**2\
            * (w[:, 0] > 0)
        return zw

    def grad_zwrad_w(self, w, states):
        grad_zw = np.zeros((len(w), 1, 1))
        rho = states[:, 3] / (self.tractr.Sw*states[:, 4])
        grad_zw[:, 0, 0] = 1 * w[:, 0] / (rho) \
            * (1/((states[:, 4])*self.tractr.L0))**2\
            * (w[:, 0] > 0)
        return grad_zw

    def grad_zwrad_x(self, w, states):
        grad_zw = np.zeros((len(w), 1, self.tractr.n_state))
        h = states[:, 4]
        m = states[:, 3]
        grad_zw[:, 0, 3] = - 1 * w[:, 0]**2 * self.tractr.Sw / \
            (2*self.tractr.l0**2*h*m**2)*(w[:, 0] > 0)

        grad_zw[:, 0, 4] = - 1 * w[:, 0]**2 * self.tractr.Sw / \
            (2*self.tractr.l0**2*h**2*m)*(w[:, 0] > 0)
        return grad_zw

    def zw(self, w, states):
        states_l, states_r = self.separate_states(states)
        diss_l, diss_r = self.separate_diss(w)
        zw_l = self.tractl.zw(diss_l, states_l)
        zw_r = self.tractr.zw(diss_r, states_r)
        return np.concatenate((zw_l, zw_r), axis=1)

    def grad_zw_w(self, w, states):
        states_l, states_r = self.separate_states(states)
        diss_l, diss_r = self.separate_diss(w)
        grad_zw_l = self.tractl.grad_zw_w(diss_l, states_l)
        grad_zw_r = self.tractr.grad_zw_w(diss_r, states_r)
        grad_zw = np.block(
            [[grad_zw_l, np.zeros((len(states), self.tractl.n_diss, self.tractr.n_diss))],
             [np.zeros((len(states), self.tractr.n_diss, self.tractl.n_diss)), grad_zw_r]])
        return grad_zw

    def grad_zw_x(self, w, states):
        states_l, states_r = self.separate_states(states)
        diss_l, diss_r = self.separate_diss(w)
        grad_zw_l = self.tractl.grad_zw_x(diss_l, states_l)
        grad_zw_r = self.tractr.grad_zw_x(diss_r, states_r)
        grad_zw = np.block(
            [[grad_zw_l, np.zeros((len(states), self.tractl.n_diss, self.tractr.n_state))],
             [np.zeros((len(states), self.tractr.n_diss, self.tractl.n_state)), grad_zw_r]])
        return grad_zw
    
    def get_parameters(self):
        add_parameters = {
            "Tracts": self.N_tracts,
            "K": self.k,
            "l0": self.l0,
            "L0": self.L0,
            "Sw": self.Sw,
            "r": self.r,
            "rho0": self.rho0,
            "gamma": self.gamma,
            "P0": self.P0,
        }
        all_parameters = dict(**add_parameters)
        return all_parameters


class MultipleTractsWithWallsVflow(MultipleTractsWithWalls):
    """Multiples tracts of the model with walls. All
    tracts are assumed to have same l0 and L0. Causlity of the input
    is reversed, control is then done with input volume flow.
    """
    def __init__(self, N, l0, Sw, k, r, diss=False):
        """Initializes the class.

        Args:
            N (int): number of tracts.
            l0 (float): semi length of one tract.
            Sw (float): Surface 2*L0*l0 of one tract.
            k (float): stiffness of the walls.
            r (float): damping coefficient of the walls.
        """
        super().__init__(N, l0, Sw, k, r)

        # Reversing causality of the input using a Lagrange multiplier
        self.n_cons += 1
        self.full_size += 1
        self._init_S()
        self._set_Jxx(self.Sxx)
        self._set_Jxw(self.Sxw)
        self._set_Jww(self.Sww)
        new_Sxl = np.zeros((self.n_state, self.n_cons))
        new_Sxl[:, :-1] = self.Sxl
        new_Sxl[0, -1] = -1
        self._set_Jxl(new_Sxl)
        new_Slu = np.zeros((self.n_cons, self.n_io))
        new_Slu[:-1, :] = self.Slu
        new_Slu[-1, 0] = -1
        self._set_Jlu(new_Slu)
        new_Syx = np.zeros((self.n_io, self.n_state))
        new_Syx[1:, :] = self.Syx[1:, :]
        self._set_Jxu(-new_Syx.T)
        self._split_S()

    def get_parameters(self):
        base_parameters = super().get_parameters()
        add_parameters = {
            "Tracts": self.N_tracts,
            "K": self.k,
            "l0": self.l0,
            "L0": self.L0,
            "Sw": self.Sw,
            "r": self.r,
            "rho0": self.rho0,
            "gamma": self.gamma,
            "P0": self.P0,
        }
        all_parameters = dict(base_parameters, **add_parameters)
        return all_parameters
