from pyphs import Core
import numpy as np

from models.pyphsModels import PyphsBase


class VocalTractEnthalpyControlled(PyphsBase):
    def __init__(self):
        self.core = Core("Single vocal tract")
        # Storing elements
        # Symbols declaration
        # States
        vl, vr, piy, m, h = self.core.symbols(
            ["\\nu_l", "\\nu_r", "\\pi_y", "m", "h"]
        )
        # Hamiltonian parameters
        self.l0, self.P0, self.Sw, self.gamma, self.rho0 = self.core.symbols(
            ["l_0", "P_0", "S_w", "\\gamma", "\\rho_0"]
        )

        # Adding storages components
        H1 = 0.5 / self.l0 ** 2 * m * (vl ** 2 + vr ** 2 - vl * vr)
        H2 = 3 * piy ** 2 / (2 * m)
        rho_m = m / (self.Sw * h)
        H3 = (
            self.P0
            * self.Sw
            * h
            * (self.gamma / 2 * (rho_m / self.rho0 - 1) ** 2 - 1)
        )
        H = H1 + H2 + H3
        self.core.add_storages([vl, vr, piy, m, h], H)

        # Dissipative elements
        # Symbols declaration
        dotm, doty = self.core.symbols(["\\frac{dm}{dt}", "\\frac{dy}{dt}"])

        # Adding dissipation law
        zw1 = doty * piy / m
        zw2 = -dotm * piy / m
        self.core.add_dissipations([dotm, doty], [zw1, zw2])

        # Ports
        # Symbol declaration
        ql, qr, vw, psil, psir, Fw = self.core.symbols(
            ["-q_l", "q_r", "-v_w", "\\psi_L", "\\psi_R", "F_w"]
        )
        self.core.add_ports([psil, Fw, psir], [ql, vw, qr])

        # Interconnection matrix
        # Initialization
        self.core.init_M()

        # Indexes of 1
        ind1 = [[0, 1, 1, 2, 3, 4, 5, 6, 9], [7, 3, 5, 8, 0, 2, 0, 2, 1]]
        Sarray = np.zeros_like(self.core.M)
        Sarray[ind1[0], ind1[1]] = 1
        Sarray[ind1[1], ind1[0]] = -1
        self.core.M[:, :] = Sarray

        # self.core.move_port(1, 2)

    def define_parameters(self, l0, Sw, rho0=1.292, P0=101325, gamma=1.4):
        subs = {
            self.l0: l0,
            self.Sw: Sw,
            self.rho0: rho0,
            self.P0: P0,
            self.gamma: gamma,
        }
        self.core.substitute(subs=subs)


class VocalTractPressureControlled(PyphsBase):
    def __init__(self):
        self.core = Core("Single vocal tract")
        # Storing elements
        # Symbols declaration
        # States
        vl, vr, piy, m, h = self.core.symbols(
            ["\\nu_l", "\\nu_r", "\\pi_y", "m", "h"]
        )
        # Hamiltonian parameters
        self.l0, self.P0, self.Sw, self.gamma, self.rho0 = self.core.symbols(
            ["l_0", "P_0", "S_w", "\\gamma", "\\rho_0"]
        )

        # Adding storages components
        H1 = 0.5 / self.l0 ** 2 * m * (vl ** 2 + vr ** 2 - vl * vr)
        H2 = 3 * piy ** 2 / (2 * m)
        rho_m = m / (self.Sw * h)
        H3 = (
            self.P0
            * self.Sw
            * h
            * (self.gamma / 2 * (rho_m / self.rho0 - 1) ** 2 - 1)
        )
        H = H1 + H2 + H3
        self.core.add_storages([vl, vr, piy, m, h], H)

        # Dissipative elements
        # Symbols declaration
        dotm, doty, qL, Plgyr = self.core.symbols(
            ["\\frac{dm}{dt}", "\\frac{dy}{dt}", "-q_L", "P_L"]
        )

        # Adding dissipation law
        zw1 = doty * piy / m
        zw2 = -dotm * piy / m

        # Pressure control gyrator
        zw3 = 1 / self.rho0 * Plgyr  # psiL
        zw4 = -1 / self.rho0 * qL  # ul
        self.core.add_dissipations(
            [dotm, doty, qL, Plgyr], [zw1, zw2, zw3, zw4]
        )

        # Ports
        # Symbol declaration
        ul, qr, vw, Pl, psir, Fw = self.core.symbols(
            ["-u_l", "q_r", "-v_w", "P_L", "\\psi_R", "F_w"]
        )
        self.core.add_ports([Pl, Fw, psir], [ul, vw, qr])

        # Interconnection matrix
        # Initialization
        self.core.init_M()

        # Indexes of 1
        ind1 = [
            [1, 1, 2, 3, 4, 5, 6, 11, 0, 8],
            [3, 5, 10, 0, 2, 0, 2, 1, 7, 9],
        ]
        Sarray = np.zeros_like(self.core.M)
        Sarray[ind1[0], ind1[1]] = 1
        Sarray[ind1[1], ind1[0]] = -1
        self.core.M[:, :] = Sarray

        # self.core.move_port(1, 2)

    def define_parameters(self, l0, Sw, rho0=1.292, P0=101325, gamma=1.4):
        subs = {
            self.l0: l0,
            self.Sw: Sw,
            self.rho0: rho0,
            self.P0: P0,
            self.gamma: gamma,
        }
        self.core.substitute(subs=subs)


class Wall(PyphsBase):
    def __init__(self):
        self.core = Core("Single vocal tract")
        # Storing elements
        # Symbols declaration
        # States
        (xi,) = self.core.symbols(["\\xi"])
        # Hamiltonian parameters
        (self.k,) = self.core.symbols(["k"])

        # Adding storages components
        H = 0.5 * xi ** 2 * self.k
        self.core.add_storages([xi], H)

        # Dissipative elements
        # Symbols declaration
        w, self.alpha = self.core.symbols(["w", "alpha"])

        # Adding dissipation law
        zw = self.alpha * w
        self.core.add_dissipations([w], [zw])

        # Ports
        # Symbol declaration
        Fext, Fm, vext, vm = self.core.symbols(
            ["F_{{ext}}", "F_m", "v_{{ext}}", "v_m"]
        )
        self.core.add_ports([vext, vm], [Fext, Fm])

        # Interconnection matrix
        # Initialization
        self.core.init_M()

        # Indexes of 1
        ind1 = [[2, 2, 3, 3], [0, 1, 0, 1]]
        Sarray = np.zeros_like(self.core.M)
        Sarray[ind1[0], ind1[1]] = 1
        Sarray[ind1[1], ind1[0]] = -1
        self.core.M[:, :] = Sarray

    def define_parameters(self, k, alpha):
        subs = {
            self.k: k,
            self.alpha: alpha,
        }
        self.core.substitute(subs=subs)


class EnthalpyTractWithWall(PyphsBase):
    def __init__(self):
        self.Tract = VocalTractEnthalpyControlled()
        self.Wall = Wall()

        # Assembly using the pyphs models
        self.core = self.Tract.core + self.Wall.core

        Fw = self.core.u.index(self.Tract.core.u[1])
        Fext = self.core.y.index(self.Wall.core.y[0])
        self.core.add_connector((Fw, Fext), alpha=-1)

        self.core.connect()
        self.core.move_port(1, 2)


class PressureTractWithWall(PyphsBase):
    def __init__(self):
        self.Tract = VocalTractPressureControlled()
        self.Wall = Wall()

        # Assembly using the pyphs models
        self.core = self.Tract.core + self.Wall.core

        Fw = self.core.u.index(self.Tract.core.u[1])
        Fext = self.core.y.index(self.Wall.core.y[0])
        self.core.add_connector((Fw, Fext), alpha=-1)

        self.core.connect()
        self.core.move_port(1, 2)
