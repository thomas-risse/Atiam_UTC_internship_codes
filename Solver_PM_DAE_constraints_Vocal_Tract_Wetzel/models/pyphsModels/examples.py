import sympy as sp
import numpy as np
from pyphs import Core
from models.pyphsModels.modelspyphs import PyphsBase


class MSDLinear(PyphsBase):
    def __init__(self):
        self.core = Core("MSD")

        # States
        l, pi = self.core.symbols(["l", "\\pi"])

        # Hamiltonian parameters
        self.m, self.k =\
            self.core.symbols(["m", "k"])

        # Adding storages components
        H = 0.5 * (pi**2 / self.m + l**2 * self.k)
        self.core.add_storages([pi, l], H)

        # Dissipative elements
        # Symbols declaration
        wd, self.r = self.core.symbols(["w_{d}", "r"])

        # Adding dissipation law
        zw = self.r*wd
        self.core.add_dissipations(wd, zw)

        # Ports
        # Symbol declaration
        v_ext, F_ext = self.core.symbols(
            ["v_{ext}", "F_{ext}"])
        self.core.add_ports([F_ext], [v_ext])

        # Interconnection matrix
        # Initialization
        self.core.init_M()

        self.core.M = sp.Matrix([[0, 1, 1, 1],
                                 [-1, 0, 0, 0],
                                 [-1, 0, 0, 0],
                                 [-1, 0, 0, 0]])


class DuffingAdim(PyphsBase):
    def __init__(self):
        self.core = Core("Duffing")

        # States
        l, pi = self.core.symbols(["l", "\\pi"])

        # Hamiltonian parameters
        self.m, self.k1, self.k2 =\
            self.core.symbols(["m", "k_1", "k_2"])

        # Adding storages components
        H = 0.5 * (pi**2 / self.m +
                   l**2 * self.k1 + l**4 * self.k2)
        self.core.add_storages([pi, l], H)

        # Dissipative elements
        # Symbols declaration
        wd, self.r = self.core.symbols(["w_{d}", "r"])

        # Adding dissipation law
        zw = self.r*wd*sp.Abs(l)
        self.core.add_dissipations(wd, zw)

        # Interconnection matrix
        # Initialization
        self.core.init_M()

        self.core.M = sp.Matrix([[0, -1, -1],
                                 [1, 0, 0],
                                 [1, 0, 0]])
