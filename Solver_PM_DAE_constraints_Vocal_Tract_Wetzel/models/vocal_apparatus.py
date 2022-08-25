from models.models import PhsModel
import numpy as np
from IPython.display import display
import models.pyphsModels.vocal_apparatus_model_pyphs as pyphsmodels

"""This file defines classes for the SHP models of the vocal apparatus proposed 
by Hélie and Silva in 2017. 
"""
class VocalFold(PhsModel):
    def __init__(self, m, k, kappa, r, S_sub, S_sup, i):
        # Number of state variables, dissipative elements,
        # constraints equations and input/output pair
        self.n_state = 3  # state
        self.n_diss = 1  # dissipative elements
        self.n_cons = 0  # constraints
        self.n_io = 3  # input/output pairs
        # Full size of the system
        self.full_size = self.n_state + self.n_diss + self.n_cons + self.n_io

        # Hamiltonian parameters
        self.m = m
        self.k = k
        self.kappa = kappa
        # Dissipation parameter
        self.r = r
        # Interconnection parameters
        self.S_sub = S_sub
        self.S_sup = S_sup

        # Interconnection matrix as a 2D numpy array
        self.S = np.array(
            [
                [0, -1, 1, -1, -self.S_sub, -self.S_sup, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0],
                [self.S_sub, 0, 0, 0, 0, 0, 0],
                [self.S_sup, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0],
            ]
        )

        # Interconnection matrix splitted
        self._split_S()

        # Associated pyphs symbolic version
        self.pyphs = pyphsmodels.VocalFold(i)
        self.pyphs.define_parameters(m, k, kappa, r, S_sub, S_sup)

    def H(self, states):
        return 0.5 * (
            states[:, 0] ** 2 / self.m
            + states[:, 1] ** 2 * self.k
            + states[:, 2] ** 2 * self.kappa
        )

    def grad_H(self, states):
        grad_H = np.zeros_like(states)
        grad_H[:, 0] = states[:, 0] / self.m
        grad_H[:, 1] = states[:, 1] * self.k
        grad_H[:, 2] = states[:, 2] * self.kappa
        return grad_H

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))
        hess[:, 0, 0] = 1 / self.m
        hess[:, 1, 1] = self.k
        hess[:, 2, 2] = self.kappa
        return hess

    def zw(self, w, states):
        return w * self.r

    def grad_zw_w(self, w, states):
        return np.ones((len(w), self.n_diss, self.n_diss)) * self.r

    def grad_zw_x(self, w, states):
        return np.zeros((len(w), self.n_diss, self.n_state))


class GFlow(PhsModel):
    def __init__(self, rho, L0, l0):
        # Number of state variables, dissipative elements,
        # constraints equations and input/output pair
        self.n_state = 4  # state
        self.n_diss = 5  # dissipative elements
        self.n_cons = 0  # constraints
        self.n_io = 4  # input/output pairs
        # Full size of the system
        self.full_size = self.n_state + self.n_diss + self.n_cons + self.n_io

        # Hamiltonian parameters
        self.rho = rho
        self.L0 = L0
        self.l0 = l0
        self.rhoL0l0 = self.rho * self.L0 * self.l0

        # Interconnection matrix as a 2D numpy array
        self._init_S()

        rhol2 = 1 / (2 * self.rho * self.l0)
        self._set_Jxw(
            np.array(
                [
                    [0, 0, 0, 0, -rhol2],
                    [0, -1, 0, 0, 0],
                    [0, 0, 0, -1, 0],
                    [0, 0, 1, 0, 0],
                ]
            )
        )

        self._set_Jxu(
            np.array(
                [
                    [rhol2, -rhol2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
        )

        Ll = self.L0 * self.l0
        self._set_Jww(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, Ll],
                    [0, 0, 0, 0, 0],
                    [0, 0, -Ll, 0, 0],
                ]
            )
        )

        self._set_Jwu(
            np.array(
                [
                    [0, 0, -1, 1],
                    [0, 0, 0, 0],
                    [Ll, Ll, -0.5, -0.5],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
        )

        # Interconnection matrix splitted
        self._split_S()
        self.pyphs = pyphsmodels.GFlowIndependant()
        self.pyphs.define_parameters(self.rho, self.L0, self.l0)

    def H(self, states):
        v0 = states[:, 0]
        vym = states[:, 1]
        vh = states[:, 2]
        h = states[:, 3]
        m = 2 * self.rhoL0l0 * h
        m_3 = m * (1 + 4 * self.l0 ** 2 / h ** 2) / 12
        return 0.5 * ((v0 ** 2 + vym ** 2) * m + vh ** 2 * m_3)

    def grad_H(self, states):
        v0 = states[:, 0]
        vym = states[:, 1]
        vh = states[:, 2]
        h = states[:, 3]
        m = 2 * self.rho * self.l0 * self.L0 * h
        m_3 = m * (1 + 4 * self.l0 ** 2 / h ** 2) / 12
        grad_H = np.zeros_like(states)
        grad_H[:, 0] = v0 * m
        grad_H[:, 1] = vym * m
        grad_H[:, 2] = vh * m_3
        grad_H[:, 3] = (
            self.rhoL0l0 * (v0 ** 2 + vym ** 2)
            + self.rhoL0l0 * (1 - 4 * self.l0 ** 2 / h ** 2) / 12 * vh ** 2
        )
        return grad_H

    def hess_H(self, states):
        v0 = states[:, 0]
        vym = states[:, 1]
        vh = states[:, 2]
        h = states[:, 3]
        m = 2 * self.rhoL0l0 * h
        m_3 = m * (1 + 4 * self.l0 ** 2 / h ** 2) / 12
        hess = np.zeros((len(states), self.n_state, self.n_state))

        hess[:, 0, 0] = m
        hess[:, 0, 3] = 2 * self.rhoL0l0 * v0

        hess[:, 1, 1] = m
        hess[:, 1, 3] = 2 * self.rhoL0l0 * vym

        hess[:, 2, 2] = m_3
        hess[:, 2, 3] = (
            self.rhoL0l0 * (1 - 4 * self.l0 ** 2 / h ** 2) / 12 * vh
        )

        hess[:, 3, 0] = 2 * self.rhoL0l0 * (v0)
        hess[:, 3, 1] = 2 * self.rhoL0l0 * (vym)
        hess[:, 3, 2] = (
            1 / 6 * self.rhoL0l0 * vh * (1 - 4 * self.l0 ** 2 / h ** 2)
        )
        hess[:, 3, 3] = 3 / 4 * self.rhoL0l0 * self.l0 ** 2 / h ** 3
        return hess

    def zw(self, w, states):
        zw = np.zeros_like(w)

        gx00 = w[:, 0]
        gx01 = w[:, 1]
        gx10 = w[:, 2]
        gx11 = w[:, 3]

        h = states[:, 3]
        m = 2 * self.rhoL0l0 * h
        m_3 = m * (1 + 4 * self.l0 ** 2 / h ** 2) / 12

        zw[:, 0] = gx01 / m
        zw[:, 1] = -gx00 / m

        zw[:, 2] = gx11 / m_3
        zw[:, 3] = -gx10 / m_3

        wG = w[:, 4]
        zw[:, 4] = 0.5 * self.rho * (wG / (self.L0 * h)) ** 2 * (wG > 0)
        return zw

    def grad_zw_w(self, w, states):
        gradzw = np.zeros((len(w), self.n_diss, self.n_diss))

        h = states[:, 3]
        m = 2 * self.rhoL0l0 * h
        m_3 = m * (1 + 4 * self.l0 ** 2 / h ** 2) / 12

        gradzw[:, 0, 1] = 1 / m
        gradzw[:, 1, 0] = -1 / m
        gradzw[:, 2, 3] = 1 / m_3
        gradzw[:, 3, 2] = -1 / m_3

        wG = w[:, 4]
        gradzw[:, 4, 4] = wG * self.rho * (1 / (self.L0 * h)) ** 2 * (wG > 0)
        return gradzw

    def grad_zw_x(self, w, states):
        gradzwx = np.zeros((len(w), self.n_diss, self.n_state))

        h = states[:, 3]
        m = 2 * self.rhoL0l0 * h

        gx00 = w[:, 0]
        gx01 = w[:, 1]
        gx10 = w[:, 2]
        gx11 = w[:, 3]

        gradzwx[:, 0, 3] = -gx01 / (m * h)
        gradzwx[:, 1, 3] = gx00 / (m * h)
        factor = 12 * (
            -1 / (2 * self.rhoL0l0 * (h ** 2 + 4 * self.l0 ** 2))
            + 4
            * self.l0
            / (
                self.L0
                * self.rho
                * h ** 4
                * (1 + 4 * self.l0 ** 2 / h ** 2) ** 2
            )
        )
        gradzwx[:, 2, 3] = gx11 * factor
        gradzwx[:, 3, 3] = -gx10 * factor

        wG = w[:, 4]
        gradzwx[:, 4, 3] = (-self.rho * (wG / (self.L0)) ** 2 / h ** 3) * (
            wG > 0
        )
        return gradzwx


class GFlowMomentum(PhsModel):
    def __init__(self, rho, L0, l0, h0, hr):
        # Number of state variables, dissipative elements,
        # constraints equations and input/output pair
        self.n_state = 4  # state
        self.n_diss = 3  # dissipative elements
        self.n_cons = 0  # constraints
        self.n_io = 4  # input/output pairs
        # Full size of the system
        self.full_size = self.n_state + self.n_diss + self.n_cons + self.n_io

        # Hamiltonian parameters
        self.rho = rho
        self.L0 = L0
        self.l0 = l0
        self.rhoL0l0 = self.rho * self.L0 * self.l0
        self.h0 = h0
        self.hr = hr
        self.mu0 = self.rho * self.L0 * 2 * self.l0

        # Pyphs model
        self.pyphs = pyphsmodels.GFlowMomentumIndependant()
        self.pyphs.define_parameters(
            self.rho, self.L0, self.l0, self.h0, self.hr
        )

        self._init_S()
        self.S = np.array(self.pyphs.core.M, dtype=np.float64)
        self._split_S()

    def H(self, states):
        pix = states[:, 0]
        piy = states[:, 1]
        piexp = states[:, 2]
        h = states[:, 3]
        htotal = self.hr + h
        m = self.mu0 * htotal
        m33 = m / 12 * (1 + 4 * self.l0 ** 2 / htotal ** 2)
        H = (htotal / self.h0) ** 2 * (pix ** 2 + piy ** 2) / (
            2 * self.mu0 * self.h0
        ) + piexp ** 2 / (8 * m33)
        return H

    def grad_H(self, states):
        pix = states[:, 0]
        piy = states[:, 1]
        piexp = states[:, 2]
        h = states[:, 3]
        htotal = self.hr + h
        rholL = 2 * self.rho * self.l0 * self.L0
        gradH = np.zeros_like(states)
        gradH[:, 0] = pix * htotal ** 2 / (rholL * self.h0 ** 3)
        gradH[:, 1] = piy * htotal ** 2 / (rholL * self.h0 ** 3)
        gradH[:, 2] = (
            3
            * piexp
            / (rholL * (htotal) * (4 * self.l0 ** 2 / (htotal) ** 2 + 1))
        )
        gradH[:, 3] = (pix ** 2 + piy ** 2) * (2 * htotal) / (
            2 * rholL * self.h0 ** 3
        ) + 3 * piexp ** 2 * (
            2
            * self.l0
            / (
                self.L0
                * self.rho
                * (htotal) ** 4
                * (4 * self.l0 ** 2 / (htotal) ** 2 + 1) ** 2
            )
            - 1
            / (
                2
                * rholL
                * (htotal) ** 2
                * (4 * self.l0 ** 2 / (htotal) ** 2 + 1)
            )
        )
        return gradH

    def hess_H(self, states):
        pix = states[:, 0]
        piy = states[:, 1]
        piexp = states[:, 2]
        h = states[:, 3]
        htotal = h + self.hr

        hess = np.zeros((len(states), self.n_state, self.n_state))

        term1 = 4 * self.l0 ** 2 / htotal ** 2 + 1

        hess[:, 0, 0] = htotal ** 2 / (2 * self.rhoL0l0 * self.h0 ** 3)
        hess[:, 0, 3] = pix * htotal / (self.rhoL0l0 * self.h0 ** 3)

        hess[:, 1, 1] = htotal ** 2 / (2 * self.rhoL0l0 * self.h0 ** 3)
        hess[:, 1, 3] = piy * htotal / (self.rhoL0l0 * self.h0 ** 3)

        hess[:, 2, 2] = 3 / (2 * self.rhoL0l0 * htotal * term1)
        hess[:, 2, 3] = piexp * (
            12 * self.l0 / (self.L0 * self.rho * htotal ** 4 * term1 ** 2)
            - 3 / (2 * self.rhoL0l0 * htotal ** 2 * term1)
        )

        hess[:, 3, 0] = hess[:, 0, 3]
        hess[:, 3, 1] = hess[:, 1, 3]
        hess[:, 3, 2] = hess[:, 2, 3]
        hess[:, 3, 3] = (pix ** 2 + piy ** 2) / (
            2 * self.rhoL0l0 * self.h0 ** 3
        ) + piexp ** 2 * (
            96 * self.l0 ** 3 / (self.L0 * self.rho * htotal ** 7 * term1 ** 3)
            - 30 * self.l0 / (self.L0 * self.rho * htotal ** 5 * term1 ** 2)
            + 3 / (2 * self.rhoL0l0 * htotal ** 3 * term1)
        )
        return hess

    def zw(self, w, states):
        zw = np.zeros_like(w)

        h = states[:, 3]
        htotal = h + self.hr

        factor = self.h0 / htotal
        wG = w[:, 0]
        zw[:, 0] = 0.5 * self.rho * (wG / (self.L0 * htotal)) ** 2 * (wG > 0)

        zw[:, 1] = w[:, 2] * factor
        zw[:, 2] = -w[:, 1] * factor
        return zw

    def grad_zw_w(self, w, states):
        gradzw = np.zeros((len(w), self.n_diss, self.n_diss))

        h = states[:, 3]
        htotal = h + self.hr
        factor = self.h0 / htotal

        wG = w[:, 0]
        gradzw[:, 0, 0] = (
            wG * self.rho * (1 / (self.L0 * htotal)) ** 2 * (wG > 0)
        )
        gradzw[:, 1, 2] = factor
        gradzw[:, 2, 1] = -factor
        return gradzw

    def grad_zw_x(self, w, states):
        gradzwx = np.zeros((len(w), self.n_diss, self.n_state))

        h = states[:, 3]
        htotal = h + self.hr

        wG = w[:, 0]
        gx00 = w[:, 1]
        gx01 = w[:, 2]

        gradzwx[:, 0, 3] = (
            -self.rho * (wG / (self.L0)) ** 2 / htotal ** 3
        ) * (wG > 0)
        gradzwx[:, 1, 3] = self.h0 * gx01 / (htotal) ** 2
        gradzwx[:, 2, 3] = -self.h0 * gx00 / (htotal) ** 2
        return gradzwx


class VocalTractAc(PhsModel):
    def __init__(self, n_res, ai, wi, qi):
        # Number of resonances
        self.n_res = n_res

        # Number of state variables, dissipative elements,
        # constraints equations and input/output pair
        self.n_state = 2 * self.n_res  # state
        self.n_diss = self.n_res  # dissipative elements
        self.n_cons = 0  # constraints
        self.n_io = 1  # input/output pairs
        # Full size of the system
        self.full_size = self.n_state + self.n_diss + self.n_cons + self.n_io

        # Hamiltonian parameters
        self.ai = ai
        self.wi = wi
        # Dissipation parameter
        self.qi = qi

        # Interconnection matrix as a 2D numpy array
        self._init_S()

        # Jx
        Jx = np.block(
            [
                [np.zeros((self.n_res, self.n_res)), -np.eye(self.n_res)],
                [np.eye(self.n_res), np.zeros((self.n_res, self.n_res))],
            ]
        )
        self._set_Jxx(Jx)
        # K
        K = np.block(
            [[-np.eye(self.n_res)], [np.zeros((self.n_res, self.n_res))]]
        )
        self._set_Jxw(K)

        # Gx
        Gx = np.block(
            [[np.ones((self.n_res, 1))], [np.zeros((self.n_res, 1))]]
        )
        self._set_Jxu(Gx)

        # Interconnection matrix splitted
        self._split_S()

        # Pyphs symbolic model
        self.pyphs = pyphsmodels.VocalTract(self.n_res)
        self.pyphs.define_parameters(self.ai, self.wi, self.qi)

    def H(self, states):
        return (
            0.5
            * (
                states[:, : self.n_res] ** 2 * self.ai
                + states[:, self.n_res :] ** 2 * self.wi ** 2 / self.ai
            ).flatten()
        )

    def grad_H(self, states):
        grad_H = np.zeros_like(states)
        grad_H[:, : self.n_res] = states[:, : self.n_res] * self.ai
        grad_H[:, self.n_res :] = (
            states[:, self.n_res :] * self.wi ** 2 / self.ai
        )
        return grad_H

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))
        hess[:, : self.n_res, : self.n_res] = np.tile(
            np.diag(self.ai), (len(states), 1, 1)
        )
        hess[:, self.n_res :, self.n_res :] = np.tile(
            np.diag(self.wi ** 2 / self.ai), (len(states), 1, 1)
        )
        return hess

    def zw(self, w, states):
        return w * self.qi * self.wi / self.ai

    def grad_zw_w(self, w, states):
        diag = np.diag(self.qi * self.wi / self.ai)
        return np.tile(diag, (len(w), 1, 1))

    def grad_zw_x(self, w, states):
        return np.zeros((len(w), self.n_diss, self.n_state))


class Larynx(PhsModel):
    def __init__(self, VocalFold1, VocalFold2, GFlow):
        self.VocalFold1 = VocalFold1
        self.VocalFold2 = VocalFold2
        self.GFlow = GFlow

        # System Size
        self.n_state = VocalFold1.n_state + VocalFold1.n_state + GFlow.n_state
        self.n_diss = VocalFold1.n_diss + VocalFold1.n_diss + GFlow.n_diss
        self.n_cons = 0
        self.n_io = 2

        # Full size of the system
        self.full_size = self.n_state + self.n_diss + self.n_cons + self.n_io

        # Assembly using the pyphs models
        self.vocal_apparatus = self._connect_pyphs()
        display(self.vocal_apparatus.M)

        # Interconnection matrix
        self.S = np.array(self.vocal_apparatus.M, dtype=float)

        self._split_S()

    def _connect_pyphs(self):
        self.SubGlottalCavity = pyphsmodels.SubGlottalCavity()
        self.SupraGlottalCavity = pyphsmodels.SupraGlottalCavity()

        Gflow = self.GFlow.pyphs.core
        Vfold1 = self.VocalFold1.pyphs.core
        Vfold2 = self.VocalFold2.pyphs.core
        Supglottal = self.SupraGlottalCavity.core
        Subglottal = self.SubGlottalCavity.core

        vocal_apparatus = Vfold1 + Vfold2 + Gflow + Subglottal + Supglottal

        # Connection bewtween vocal folds and glottal flow

        F1_p = vocal_apparatus.y.index(Vfold1.y[2])
        Flp = vocal_apparatus.u.index(Gflow.u[2])
        vocal_apparatus.add_connector((Flp, F1_p), alpha=-1)

        F2_p = vocal_apparatus.y.index(Vfold2.y[2])
        Frp = vocal_apparatus.u.index(Gflow.u[3])
        vocal_apparatus.add_connector((Frp, F2_p), alpha=-1)

        # Connection of sub-glottal cavity
        P1sub = vocal_apparatus.u.index(Vfold1.u[0])
        P1sub_2 = vocal_apparatus.y.index(Subglottal.y[1])
        vocal_apparatus.add_connector((P1sub, P1sub_2), alpha=-1)

        P2sub = vocal_apparatus.u.index(Vfold2.u[0])
        P2sub_2 = vocal_apparatus.y.index(Subglottal.y[2])
        vocal_apparatus.add_connector((P2sub, P2sub_2), alpha=-1)

        Ptot_minus = vocal_apparatus.u.index(Gflow.u[0])
        Ptot_minus_2 = vocal_apparatus.y.index(Subglottal.y[3])
        vocal_apparatus.add_connector((Ptot_minus, Ptot_minus_2), alpha=-1)

        # Connection of supra-glottal cavity
        P1sup = vocal_apparatus.u.index(Vfold1.u[1])
        P1sup_2 = vocal_apparatus.y.index(Supglottal.y[1])
        vocal_apparatus.add_connector((P1sup, P1sup_2), alpha=1)

        P2sup = vocal_apparatus.u.index(Vfold2.u[1])
        P2sup_2 = vocal_apparatus.y.index(Supglottal.y[2])
        vocal_apparatus.add_connector((P2sup, P2sup_2), alpha=1)

        Ptot_plus = vocal_apparatus.u.index(Gflow.u[1])
        Ptot_plus_2 = vocal_apparatus.y.index(Supglottal.y[3])
        vocal_apparatus.add_connector((Ptot_plus, Ptot_plus_2), alpha=1)

        vocal_apparatus.connect()
        return vocal_apparatus

    def separate_states(self, states):
        states_VFold1 = states[:, : self.VocalFold1.n_state]
        i0 = self.VocalFold1.n_state
        i1 = i0 + self.VocalFold2.n_state
        states_VFold2 = states[:, i0:i1]
        i0 = i1
        i1 = i0 + self.GFlow.n_state
        states_Gflow = states[:, i0:i1]
        return states_Gflow, states_VFold1, states_VFold2

    def separate_diss(self, w):
        sdiss_VFold1 = w[:, : self.VocalFold1.n_diss]
        i0 = self.VocalFold1.n_diss
        i1 = i0 + self.VocalFold2.n_diss
        sdiss_VFold2 = w[:, i0:i1]
        i0 = i1
        i1 = i0 + self.GFlow.n_diss
        sdiss_Gflow = w[:, i0:i1]
        return sdiss_Gflow, sdiss_VFold1, sdiss_VFold2

    def H(self, states):
        (states_Gflow, states_VFold1, states_VFold2,) = self.separate_states(
            states
        )
        return (
            self.GFlow.H(states_Gflow)
            + self.VocalFold1.H(states_VFold1)
            + self.VocalFold2.H(states_VFold2)
        )

    def grad_H(self, states):
        (states_Gflow, states_VFold1, states_VFold2,) = self.separate_states(
            states
        )
        grad_H = np.zeros_like(states)
        grad_H[:, 0 : self.VocalFold1.n_state] = self.VocalFold1.grad_H(
            states_VFold1
        )
        i0 = self.VocalFold1.n_state
        i1 = i0 + self.VocalFold2.n_state
        grad_H[:, i0:i1] = self.VocalFold2.grad_H(states_VFold2)
        i0 = i1
        i1 = i0 + self.GFlow.n_state
        grad_H[:, i0:i1] = self.GFlow.grad_H(states_Gflow)
        return grad_H

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))
        (states_Gflow, states_VFold1, states_VFold2,) = self.separate_states(
            states
        )
        i0 = 0
        i1 = self.VocalFold1.n_state
        hess[:, i0:i1, i0:i1] = self.VocalFold1.hess_H(states_VFold1)
        i0 = i1
        i1 = i0 + self.VocalFold2.n_state
        hess[:, i0:i1, i0:i1] = self.VocalFold2.hess_H(states_VFold2)
        i0 = i1
        i1 = i0 + self.GFlow.n_state
        hess[:, i0:i1, i0:i1] = self.GFlow.hess_H(states_Gflow)
        return hess

    def zw(self, w, states):
        (states_Gflow, states_VFold1, states_VFold2,) = self.separate_states(
            states
        )
        w_Gflow, w_VFold1, w_VFold2 = self.separate_diss(w)

        zw = np.zeros_like(w)
        zw[:, 0 : self.VocalFold1.n_diss] = self.VocalFold1.zw(
            w_VFold1, states_VFold1
        )
        i0 = self.VocalFold1.n_diss
        i1 = i0 + self.VocalFold2.n_diss
        zw[:, i0:i1] = self.VocalFold2.zw(w_VFold2, states_VFold2)
        i0 = i1
        i1 = i0 + self.GFlow.n_diss
        zw[:, i0:i1] = self.GFlow.zw(w_Gflow, states_Gflow)
        return zw

    def grad_zw_w(self, w, states):
        (states_Gflow, states_VFold1, states_VFold2,) = self.separate_states(
            states
        )
        w_Gflow, w_VFold1, w_VFold2 = self.separate_diss(w)
        zw_w = np.zeros((len(w), self.n_diss, self.n_diss))
        i0 = 0
        i1 = self.VocalFold1.n_diss
        zw_w[:, i0:i1, i0:i1] = self.VocalFold1.grad_zw_w(
            w_VFold1, states_VFold1
        )
        i0 = i1
        i1 = i0 + self.VocalFold2.n_diss
        zw_w[:, i0:i1, i0:i1] = self.VocalFold2.grad_zw_w(
            w_VFold2, states_VFold2
        )
        i0 = i1
        i1 = i0 + self.GFlow.n_diss
        zw_w[:, i0:i1, i0:i1] = self.GFlow.grad_zw_w(w_Gflow, states_Gflow)
        return zw_w

    def grad_zw_x(self, w, states):
        (states_Gflow, states_VFold1, states_VFold2,) = self.separate_states(
            states
        )
        w_Gflow, w_VFold1, w_VFold2 = self.separate_diss(w)
        zw_x = np.zeros((len(w), self.n_diss, self.n_state))
        i0x = 0
        i1x = self.VocalFold1.n_state
        i0w = 0
        i1w = self.VocalFold1.n_diss
        zw_x[:, i0w:i1w, i0x:i1x] = self.VocalFold1.grad_zw_x(
            w_VFold1, states_VFold1
        )
        i0x = i1x
        i1x = i0x + self.VocalFold2.n_state
        i0w = i1w
        i1w = i0w + self.VocalFold2.n_diss
        zw_x[:, i0w:i1w, i0x:i1x] = self.VocalFold2.grad_zw_x(
            w_VFold2, states_VFold2
        )
        i0x = i1x
        i1x = i0x + self.GFlow.n_state
        i0w = i1w
        i1w = i0w + self.GFlow.n_diss
        zw_x[:, i0w:i1w, i0x:i1x] = self.GFlow.grad_zw_x(w_Gflow, states_Gflow)
        return zw_x


class VocalApparatus(PhsModel):
    def __init__(self, VocalFold1, VocalFold2, GFlow, VocalTract):
        self.VocalFold1 = VocalFold1
        self.VocalFold2 = VocalFold2
        self.GFlow = GFlow
        self.VocalTract = VocalTract

        # System Size
        self.n_state = (
            VocalFold1.n_state
            + VocalFold1.n_state
            + GFlow.n_state
            + VocalTract.n_state
        )
        self.n_diss = (
            VocalFold1.n_diss
            + VocalFold1.n_diss
            + GFlow.n_diss
            + VocalTract.n_diss
        )
        self.n_cons = 0
        self.n_io = 1

        # Full size of the system
        self.full_size = self.n_state + self.n_diss + self.n_cons + self.n_io

        # Assembly using the pyphs models
        self.vocal_apparatus = self._connect_pyphs()
        display(self.vocal_apparatus.M)

        # Interconnection matrix
        self.S = np.array(self.vocal_apparatus.M, dtype=float)

        self._split_S()

    def _connect_pyphs(self):
        self.SubGlottalCavity = pyphsmodels.SubGlottalCavity()
        self.SupraGlottalCavity = pyphsmodels.SupraGlottalCavity()

        Gflow = self.GFlow.pyphs.core
        Vfold1 = self.VocalFold1.pyphs.core
        Vfold2 = self.VocalFold2.pyphs.core
        Vtract = self.VocalTract.pyphs.core
        Supglottal = self.SupraGlottalCavity.core
        Subglottal = self.SubGlottalCavity.core

        vocal_apparatus = (
            Vfold1 + Vfold2 + Gflow + Vtract + Subglottal + Supglottal
        )

        # Connection bewtween vocal folds and glottal flow

        F1_p = vocal_apparatus.y.index(Vfold1.y[2])
        Flp = vocal_apparatus.u.index(Gflow.u[2])
        vocal_apparatus.add_connector((Flp, F1_p), alpha=-1)

        F2_p = vocal_apparatus.y.index(Vfold2.y[2])
        Frp = vocal_apparatus.u.index(Gflow.u[3])
        vocal_apparatus.add_connector((Frp, F2_p), alpha=-1)

        # Connection of sub-glottal cavity
        P1sub = vocal_apparatus.u.index(Vfold1.u[0])
        P1sub_2 = vocal_apparatus.y.index(Subglottal.y[1])
        vocal_apparatus.add_connector((P1sub, P1sub_2), alpha=-1)

        P2sub = vocal_apparatus.u.index(Vfold2.u[0])
        P2sub_2 = vocal_apparatus.y.index(Subglottal.y[2])
        vocal_apparatus.add_connector((P2sub, P2sub_2), alpha=-1)

        Ptot_minus = vocal_apparatus.u.index(Gflow.u[0])
        Ptot_minus_2 = vocal_apparatus.y.index(Subglottal.y[3])
        vocal_apparatus.add_connector((Ptot_minus, Ptot_minus_2), alpha=-1)

        # Connection of supra-glottal cavity
        Pac = vocal_apparatus.y.index(Vtract.y[0])
        Pac2 = vocal_apparatus.u.index(Supglottal.u[0])
        vocal_apparatus.add_connector((Pac, Pac2), alpha=-1)

        P1sup = vocal_apparatus.u.index(Vfold1.u[1])
        P1sup_2 = vocal_apparatus.y.index(Supglottal.y[1])
        vocal_apparatus.add_connector((P1sup, P1sup_2), alpha=1)

        P2sup = vocal_apparatus.u.index(Vfold2.u[1])
        P2sup_2 = vocal_apparatus.y.index(Supglottal.y[2])
        vocal_apparatus.add_connector((P2sup, P2sup_2), alpha=1)

        Ptot_plus = vocal_apparatus.u.index(Gflow.u[1])
        Ptot_plus_2 = vocal_apparatus.y.index(Supglottal.y[3])
        vocal_apparatus.add_connector((Ptot_plus, Ptot_plus_2), alpha=1)

        vocal_apparatus.connect()
        return vocal_apparatus

    def separate_states(self, states):
        states_VFold1 = states[:, : self.VocalFold1.n_state]
        i0 = self.VocalFold1.n_state
        i1 = i0 + self.VocalFold2.n_state
        states_VFold2 = states[:, i0:i1]
        i0 = i1
        i1 = i0 + self.GFlow.n_state
        states_Gflow = states[:, i0:i1]
        i0 = i1
        states_VTract = states[:, i0:]
        return states_Gflow, states_VFold1, states_VFold2, states_VTract

    def separate_diss(self, w):
        sdiss_VFold1 = w[:, : self.VocalFold1.n_diss]
        i0 = self.VocalFold1.n_diss
        i1 = i0 + self.VocalFold2.n_diss
        sdiss_VFold2 = w[:, i0:i1]
        i0 = i1
        i1 = i0 + self.GFlow.n_diss
        sdiss_Gflow = w[:, i0:i1]
        i0 = i1
        sdiss_VTract = w[:, i0:]
        return sdiss_Gflow, sdiss_VFold1, sdiss_VFold2, sdiss_VTract

    def H(self, states):
        (
            states_Gflow,
            states_VFold1,
            states_VFold2,
            states_VTract,
        ) = self.separate_states(states)
        return (
            self.GFlow.H(states_Gflow)
            + self.VocalFold1.H(states_VFold1)
            + self.VocalFold2.H(states_VFold2)
            + self.VocalTract.H(states_VTract)
        )

    def grad_H(self, states):
        (
            states_Gflow,
            states_VFold1,
            states_VFold2,
            states_VTract,
        ) = self.separate_states(states)
        grad_H = np.zeros_like(states)
        grad_H[:, 0 : self.VocalFold1.n_state] = self.VocalFold1.grad_H(
            states_VFold1
        )
        i0 = self.VocalFold1.n_state
        i1 = i0 + self.VocalFold2.n_state
        grad_H[:, i0:i1] = self.VocalFold2.grad_H(states_VFold2)
        i0 = i1
        i1 = i0 + self.GFlow.n_state
        grad_H[:, i0:i1] = self.GFlow.grad_H(states_Gflow)
        i0 = i1
        grad_H[:, i0:] = self.VocalTract.grad_H(states_VTract)
        return grad_H

    def hess_H(self, states):
        hess = np.zeros((len(states), self.n_state, self.n_state))
        (
            states_Gflow,
            states_VFold1,
            states_VFold2,
            states_VTract,
        ) = self.separate_states(states)
        i0 = 0
        i1 = self.VocalFold1.n_state
        hess[:, i0:i1, i0:i1] = self.VocalFold1.hess_H(states_VFold1)
        i0 = i1
        i1 = i0 + self.VocalFold2.n_state
        hess[:, i0:i1, i0:i1] = self.VocalFold2.hess_H(states_VFold2)
        i0 = i1
        i1 = i0 + self.GFlow.n_state
        hess[:, i0:i1, i0:i1] = self.GFlow.hess_H(states_Gflow)
        i0 = i1
        i1 = i0 + self.VocalTract.n_state
        hess[:, i0:i1, i0:i1] = self.VocalTract.hess_H(states_VTract)
        return hess

    def zw(self, w, states):
        (
            states_Gflow,
            states_VFold1,
            states_VFold2,
            states_VTract,
        ) = self.separate_states(states)
        w_Gflow, w_VFold1, w_VFold2, w_VTract = self.separate_diss(w)

        zw = np.zeros_like(w)
        zw[:, 0 : self.VocalFold1.n_diss] = self.VocalFold1.zw(
            w_VFold1, states_VFold1
        )
        i0 = self.VocalFold1.n_diss
        i1 = i0 + self.VocalFold2.n_diss
        zw[:, i0:i1] = self.VocalFold2.zw(w_VFold2, states_VFold2)
        i0 = i1
        i1 = i0 + self.GFlow.n_diss
        zw[:, i0:i1] = self.GFlow.zw(w_Gflow, states_Gflow)
        i0 = i1
        zw[:, i0:] = self.VocalTract.zw(w_VTract, states_VTract)
        return zw

    def grad_zw_w(self, w, states):
        (
            states_Gflow,
            states_VFold1,
            states_VFold2,
            states_VTract,
        ) = self.separate_states(states)
        w_Gflow, w_VFold1, w_VFold2, w_VTract = self.separate_diss(w)
        zw_w = np.zeros((len(w), self.n_diss, self.n_diss))
        i0 = 0
        i1 = self.VocalFold1.n_diss
        zw_w[:, i0:i1, i0:i1] = self.VocalFold1.grad_zw_w(
            w_VFold1, states_VFold1
        )
        i0 = i1
        i1 = i0 + self.VocalFold2.n_diss
        zw_w[:, i0:i1, i0:i1] = self.VocalFold2.grad_zw_w(
            w_VFold2, states_VFold2
        )
        i0 = i1
        i1 = i0 + self.GFlow.n_diss
        zw_w[:, i0:i1, i0:i1] = self.GFlow.grad_zw_w(w_Gflow, states_Gflow)
        i0 = i1
        i1 = i0 + self.VocalTract.n_diss
        zw_w[:, i0:i1, i0:i1] = self.VocalTract.grad_zw_w(
            w_VTract, states_VTract
        )
        return zw_w

    def grad_zw_x(self, w, states):
        (
            states_Gflow,
            states_VFold1,
            states_VFold2,
            states_VTract,
        ) = self.separate_states(states)
        w_Gflow, w_VFold1, w_VFold2, w_VTract = self.separate_diss(w)
        zw_x = np.zeros((len(w), self.n_diss, self.n_state))
        i0x = 0
        i1x = self.VocalFold1.n_state
        i0w = 0
        i1w = self.VocalFold1.n_diss
        zw_x[:, i0w:i1w, i0x:i1x] = self.VocalFold1.grad_zw_x(
            w_VFold1, states_VFold1
        )
        i0x = i1x
        i1x = i0x + self.VocalFold2.n_state
        i0w = i1w
        i1w = i0w + self.VocalFold2.n_diss
        zw_x[:, i0w:i1w, i0x:i1x] = self.VocalFold2.grad_zw_x(
            w_VFold2, states_VFold2
        )
        i0x = i1x
        i1x = i0x + self.GFlow.n_state
        i0w = i1w
        i1w = i0w + self.GFlow.n_diss
        zw_x[:, i0w:i1w, i0x:i1x] = self.GFlow.grad_zw_x(w_Gflow, states_Gflow)
        i0x = i1x
        i0w = i1w
        zw_x[:, i0w:, i0x:] = self.VocalTract.grad_zw_x(
            w_VTract, states_VTract
        )
        return zw_x
