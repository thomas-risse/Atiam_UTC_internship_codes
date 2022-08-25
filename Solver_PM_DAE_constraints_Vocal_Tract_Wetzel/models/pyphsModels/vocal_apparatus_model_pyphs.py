import numpy as np
from pyphs import Core
from sympy import Heaviside, Abs

from models.pyphsModels.modelspyphs import PyphsBase


class VocalFold(PyphsBase):
    def __init__(self, i):
        self.core = Core("Vocal fold")
        # Storing elements
        # Symbols declaration
        # States
        pi, xi, zeta = self.core.symbols(
            [f"\\pi_{i}", "\\xi_{i}", "\\zeta_{i}"]
        )
        # Hamiltonian parameters
        self.m, self.k, self.kappa = self.core.symbols(
            [f"m_{i}", "k_{i}", "\\kappa_{i}"]
        )

        # Adding storages components
        H = 0.5 * (
            pi ** 2 / self.m + xi ** 2 * self.k + zeta ** 2 * self.kappa
        )
        self.core.add_storages([pi, xi, zeta], H)

        # Dissipative elements
        # Symbols declaration
        wF, self.r = self.core.symbols([f"w_{{F{i}}}", "r_{i}"])

        # Adding dissipation law
        zw = self.r * wF
        self.core.add_dissipations(wF, zw)

        # Ports
        # Symbol declaration
        Psub, Psup, v, Qsub, Qsup, F = self.core.symbols(
            [
                f"P_{i}^{{sub}}",
                f"P_{i}^{{sup}}",
                f"v_{i}",
                f"-Q_{i}^{{sub}}",
                f"Q_{i}^{{sup}}",
                f"-F_{i}^p",
            ]
        )
        self.core.add_ports([Psub, Psup, v], [Qsub, Qsup, F])

        # Interconnection matrix
        # Initialization
        self.core.init_M()

        # Jx
        self.core.set_Mxx([[0, -1, 1], [1, 0, 0], [-1, 0, 0]])
        # K
        self.core.set_Jxw([-1, 0, 0])

        # Gx
        self.Ssub, self.Ssup = self.core.symbols(
            [f"S_{{sub}}^{i}", f"S_{{sup}}^{i}"]
        )
        self.core.set_Jxy([[-self.Ssub, -self.Ssup, 0], [0, 0, 0], [0, 0, 1]])

        # Jw , Jy, Gw = 0

    def define_parameters(self, m, k, kappa, r, Ssub, Ssup):
        subs = {
            self.m: m,
            self.k: k,
            self.kappa: kappa,
            self.r: r,
            self.Ssub: Ssub,
            self.Ssup: Ssup,
        }
        self.core.substitute(subs=subs)


class GFlow(PyphsBase):
    def __init__(self):
        self.core = Core("Glottal flow")

        # Storing elements
        # Symbols declaration
        # States
        v0, vym, h_dot, h = self.core.symbols(["v_0", "v_y", "\\dot{h}", "h"])
        # Hamiltonian parameters
        self.l, self.L, self.rho = self.core.symbols(["l", "L", "rho"])

        # Adding storages components
        m = 2 * self.rho * self.l * self.L * h
        m_3 = m * (1 + 4 * self.l ** 2 / h ** 2) / 12
        H = 0.5 * ((v0 ** 2 + vym ** 2) * m + h_dot ** 2 * m_3)
        self.core.add_storages([v0, vym, h_dot, h], H)

        # Dissipative elements
        # Symbols declaration
        wG = self.core.symbols("w_{G}")

        # Adding dissipation law
        zw = 0.5 * self.rho * (wG / (self.L * h)) ** 2 * Heaviside(wG)
        self.core.add_dissipations(wG, zw)

        # Ports
        # Symbol declaration
        Pminus, Pplus, Fpl, Fpr, Qminus, Qplus, vl, vr = self.core.symbols(
            [
                "P_{tot}^-",
                "P^+",
                "F_l^p",
                "F_r^p",
                "-Q^-",
                "Q^+",
                "-v_l",
                "-v_r",
            ]
        )
        self.core.add_ports([Pminus, Pplus, Fpl, Fpr], [Qminus, Qplus, vl, vr])

        # Interconnection matrix
        # Initialization
        self.core.init_M()

        # Jx
        self.core.set_Mxx(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, -1 / m_3],
                [0, 0, 1 / m_3, 0],
            ]
        )
        # K
        self.core.set_Jxw([-self.L * h / m, 0, self.L * h / m_3, 0])

        # Gx
        self.core.set_Jxy(
            [
                [self.L * h / m, -self.L * h / m, 0, 0],
                [0, 0, -1 / m, 1 / m],
                [
                    self.L * self.l / m_3,
                    self.L * self.l / m_3,
                    -0.5 / m_3,
                    -0.5 / m_3,
                ],
                [0, 0, 0, 0],
            ]
        )

        # Jw , Jy, Gw = 0

    def define_parameters(self, rho, L0, l0):
        subs = {self.rho: rho, self.L: L0, self.l: l0}
        self.core.substitute(subs=subs)


class GFlowIndependant(PyphsBase):
    def __init__(self):
        self.core = Core("Glottal flow")

        # Storing elements
        # Symbols declaration
        # States
        v0, vym, h_dot, h = self.core.symbols(["v_0", "v_y", "\\dot{h}", "h"])
        # Hamiltonian parameters
        self.l, self.L, self.rho = self.core.symbols(["l", "L", "rho"])

        # Adding storages components
        m = 2 * self.rho * self.l * self.L * h
        m_3 = m * (1 + 4 * self.l ** 2 / h ** 2) / 12
        H = 0.5 * ((v0 ** 2 + vym ** 2) * m + h_dot ** 2 * m_3)
        self.core.add_storages([v0, vym, h_dot, h], H)

        # Dissipative elements
        # Gyrators to remove m and m3 from interconnection matrix
        gx00, gx01, gx10, gx11 = self.core.symbols(
            [
                "g_{{x00}}",  # F_r^p - F_l^p
                "g_{{x01}}",  # grad_2(H)
                "g_{{x10}}",  # -grad_4(H) + Ll(zg+Ptot+ + Ptot-)
                # - 0.5*(Flp - Frp)
                "g_{{x11}}",
            ]
        )  # grad_3(H)
        zg00 = gx01 / m
        zg01 = -gx00 / m

        zg10 = gx11 / m_3
        zg11 = -gx10 / m_3

        self.core.add_dissipations(
            [gx00, gx01, gx10, gx11], [zg00, zg01, zg10, zg11]
        )

        # Dissipation
        # Symbols declaration
        wG = self.core.symbols("w_{G}")

        # Adding dissipation law
        zw = 0.5 * self.rho * (wG / (self.L * h)) ** 2 * Heaviside(wG)
        self.core.add_dissipations(wG, zw)

        # Ports
        # Symbol declaration
        Pminus, Pplus, Fpl, Fpr, Qminus, Qplus, vl, vr = self.core.symbols(
            [
                "P_{tot}^-",
                "P^+",
                "F_l^p",
                "F_r^p",
                "-Q^-",
                "Q^+",
                "-v_l",
                "-v_r",
            ]
        )
        self.core.add_ports([Pminus, Pplus, Fpl, Fpr], [Qminus, Qplus, vl, vr])

        # Interconnection matrix
        # Initialization
        self.core.init_M()

        # Jx
        self.core.set_Mxx(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        # K
        rhol2 = 1 / (2 * self.rho * self.l)
        self.core.set_Jxw(
            [
                [0, 0, 0, 0, -rhol2],
                [0, -1, 0, 0, 0],
                [0, 0, 0, -1, 0],
                [0, 0, 1, 0, 0],
            ]
        )

        # Gx
        self.core.set_Jxy(
            [[rhol2, -rhol2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        Ll = self.l * self.L
        self.core.set_Jww(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, Ll],
                [0, 0, 0, 0, 0],
                [0, 0, -Ll, 0, 0],
            ]
        )

        self.core.set_Jwy(
            [
                [0, 0, -1, 1],
                [0, 0, 0, 0],
                [Ll, Ll, -0.5, -0.5],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

    def define_parameters(self, rho, L0, l0):
        subs = {self.rho: rho, self.L: L0, self.l: l0}
        self.core.substitute(subs=subs)


class GlottalFlow_straightchannel_momentumh0(Core):
    """
    Glottal flow model based on a parametrization of the velocity field using
    - the mean axial flow (pseudo momentum Pi_x)
    - the mean transverse flow (pseudo momentum Pi_y)
    - the transverse expansion motion (momentum Pi_exp)a
    - the channel height h+hr
    Pseudo momentum Pi_x and Pi_y are defined using the arbitrary height h0 (see parameters)
    Pi_x = (2*rho*ell0*L0) * h0 * V_x
    Pi_y = (2*rho*ell0*L0) * h0 * V_y
    while Pi_exp is defined using the effective mass for expansion
    Pi_exp = 2*m3*V_exp with
    m3 = (2*rho*ell0*L0) * (h+hr) / 12 * (1 + 4*ell0**2/(h+hr)**2)

    The parameters are:
    - the density of the fluid: rho
    - the width of the channel (out-of-plane z dimension): L0
    - the half-length of the channel: ell0
    - the height of the channel at rest: hr
    - an arbitrary positive height used in the pseudo momentum: h0

    The flow is assumed to be potential, incompressible and inviscid.
    However the model includes the existence of a jet downstream of the channel,
    the jet dissipating the kinetic energy of the flow without recovering pressure.
    This is modelled as a dissipative component with variable w_turb.
    """

    def __init__(self, label="glottis", **kwargs):
        Core.__init__(self, label)

        # Parameters
        rho, L0, ell0, hr, h0 = self.symbols("\\rho L_0 l_0 h_r h_0")
        # Internal and dissipation variables
        Pix, Piy, Piexp, h, wturb = self.symbols(
            "\\Pi_x \\Pi_y \\Pi_{exp} h w_{turb}"
        )
        # Input variables
        Ptotu, Pd, Fl, Fr = self.symbols("P_{totu} P_d F_l F_r")
        # Output variables
        Qu, Qd, vl, vr = self.symbols("Q_u Q_d v_l v_r")

        htotal = hr + h
        mu0 = rho * L0 * 2 * ell0
        m = mu0 * htotal
        m33 = m / 12 * (1 + 4 * ell0 ** 2 / htotal ** 2)
        H = (htotal / h0) ** 2 * (Pix ** 2 + Piy ** 2) / (
            2 * mu0 * h0
        ) + Piexp ** 2 / (8 * m33)
        zturb = rho / 2 * ((wturb + Abs(wturb)) / (2 * L0 * htotal)) ** 2

        self.add_storages([Pix, Piy, Piexp, h], H)
        self.add_dissipations(wturb, zturb)
        self.add_ports(Ptotu, -Qu)
        self.add_ports(Pd, Qd)
        self.add_ports(Fl, -vl)
        self.add_ports(Fr, -vr)

        self.set_Jxx([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -2], [0, 0, 2, 0]])
        self.set_Jxw([-L0 * h0, 0, 2 * L0 * ell0, 0])
        self.set_Jxy(
            [
                [L0 * h0, -L0 * h0, 0, 0],
                [0, 0, -h0 / htotal, h0 / htotal],
                [2 * L0 * ell0, 2 * L0 * ell0, -1, -1],
                [0, 0, 0, 0],
            ]
        )

        self.subs.update({self.symbols(k): v for k, v in kwargs.items()})


class GFlowMomentum(PyphsBase):
    def __init__(self):
        self.core = GlottalFlow_straightchannel_momentumh0()

    def define_parameters(self, rho, L0, l0, h0, hr):
        subs = {
            self.core.symbols("rho"): rho,
            self.core.symbols("L_0"): L0,
            self.core.symbols("l_0"): l0,
            self.core.symbols("h_r"): hr,
            self.core.symbols("h_0"): h0,
        }
        self.core.substitute(subs=subs)


class GFlowMomentumIndependant(GFlowMomentum):
    def __init__(self):
        super().__init__()
        wg00, wg01 = self.core.symbols(["w_{g00}", "w_{g01}"])
        hr, h0, h, L0, l0 = self.core.symbols(
            ["h_r", "h_0", "h", "L_0", "l_0"]
        )
        htotal = hr + h
        factor = h0 / htotal
        zwg00 = factor * wg01
        zwg01 = -factor * wg00
        self.core.add_dissipations([wg00, wg01], [zwg00, zwg01])
        self.core.init_M()
        self.core.set_Jxx(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -2], [0, 0, 2, 0]]
        )
        self.core.set_Jxw(
            [[-L0 * h0, 0, 0], [0, 0, 1], [2 * L0 * l0, 0, 0], [0, 0, 0]]
        )
        self.core.set_Jxy(
            [
                [L0 * h0, -L0 * h0, 0, 0],
                [0, 0, 0, 0],
                [2 * L0 * l0, 2 * L0 * l0, -1, -1],
                [0, 0, 0, 0],
            ]
        )
        self.core.set_Jyw([[0, 0, 0], [0, 0, 0], [0, -1, 0], [0, 1, 0]])


class VocalTract(PyphsBase):
    def __init__(self, N):
        self.N = N

        self.core = Core("Vocal tract")

        # Storing elements
        # Symbols declaration
        # States
        pi_names = [f"\\frac{{p_{i}}}{{a_{i}}}" for i in range(self.N)]
        int_pi_names = [f"\\int_0^tp_{i}" for i in range(self.N)]
        states = self.core.symbols([*pi_names, *int_pi_names])
        # Hamiltonian parameters
        self.ai = self.core.symbols([f"a_{i}" for i in range(self.N)])
        self.wi = self.core.symbols([f"w_{i}" for i in range(self.N)])

        # Adding storages components
        H = 0
        self.core.add_storages(states, H)
        for i in range(N):
            H += 0.5 * (
                states[i] ** 2 * self.ai[i]
                + self.wi[i] ** 2 / self.ai[i] * states[N + i] ** 2
            )
            self.core.add_storages([], H)

        # Dissipative elements
        # Symbols declaration
        wi = self.core.symbols([f"p_{i}" for i in range(self.N)])
        self.qi = self.core.symbols([f"q_{i}" for i in range(self.N)])

        # Adding dissipation law
        for i in range(self.N):
            zw = self.qi[i] * self.wi[i] / self.ai[i] * wi[i]
            self.core.add_dissipations([wi[i]], [zw])

        # Ports
        # Symbol declaration
        Qac, Pac = self.core.symbols(["Q_{ac}", "-P_{ac}"])
        self.core.add_ports([Qac], [Pac])

        # Interconnection matrix
        # Initialization
        self.core.init_M()

        # Jx
        Jx = np.block(
            [
                [np.zeros((self.N, self.N)), -np.eye(self.N)],
                [np.eye(self.N), np.zeros((self.N, self.N))],
            ]
        )
        self.core.set_Mxx(Jx)
        # K
        K = np.block([[-np.eye(self.N)], [np.zeros((self.N, self.N))]])
        self.core.set_Jxw(K)

        # Gx
        x = np.block([[np.ones((self.N, 1))], [np.zeros((self.N, 1))]])
        self.core.set_Jxy(x)

        # Jw , Jy, Gw = 0

    def define_parameters(self, ai, wi, qi):
        subs = dict(zip(self.ai, ai))
        self.core.substitute(subs=subs)
        subs = dict(zip(self.wi, wi))
        self.core.substitute(subs=subs)
        subs = dict(zip(self.qi, qi))
        self.core.substitute(subs=subs)


class SubGlottalCavity(PyphsBase):
    def __init__(self):
        self.core = Core("Sub-glottal cavity")
        # Input variables
        Psub, Qsubl, Qsubr, Qu = self.core.symbols(
            ["P_{sub}", "Q_{sub_l}", "Q_{sub_r}", "Q_u"]
        )
        # Output variables
        Qsub, Psubl, Psubr, Ptotu = self.core.symbols(
            ["Q_{sub}", "P_{sub_l}", "P_{sub_r}", "P_{totu}"]
        )

        self.core.add_ports(Psub, -Qsub)
        self.core.add_ports(Qsubl, Psubl)
        self.core.add_ports(Qsubr, Psubr)
        self.core.add_ports(Qu, Ptotu)
        self.core.set_Jyy(
            [[0, -1, -1, -1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
        )


class SupraGlottalCavity(PyphsBase):
    def __init__(self):
        self.core = Core("Supra-glottal cavity")
        # Input variables
        Qac, Qsupl, Qsupr, Qd = self.core.symbols(
            ["Q_ac", "Q_sup_l", "Q_sup_r", "Q_d"]
        )
        # Output variables
        Pac, Psupl, Psupr, Pd = self.core.symbols(
            ["P_ac", "P_sup_l", "P_sup_r", "P_d"]
        )

        self.core.add_ports(Pac, Qac)
        self.core.add_ports(Qsupl, -Psupl)
        self.core.add_ports(Qsupr, -Psupr)
        self.core.add_ports(Qd, -Pd)
        self.core.set_Jyy(
            [[0, 1, 1, 1], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0]]
        )


class VocalApparatus(PyphsBase):
    def __init__(self, n_tract):
        self.n_tract = n_tract

        # Builds elements
        self.VFold1 = VocalFold(1)
        self.VFold2 = VocalFold(2)
        self.GFlow = GFlowMomentumIndependant()
        self.VTract = VocalTract(self.n_tract)
        self.SubGlottal = SubGlottalCavity()
        self.SupGlottal = SupraGlottalCavity()

        # Creates core
        self.core = (
            self.VFold1.core
            + self.VFold2.core
            + self.GFlow.core
            + self.VTract.core
            + self.SubGlottal.core
            + self.SupGlottal.core
        )

        self.connect()

    def connect(self):
        Gflow = self.GFlow.core
        Vfold1 = self.VFold1.core
        Vfold2 = self.VFold2.core
        Vtract = self.VTract.core
        Supglottal = self.SupGlottal.core
        Subglottal = self.SubGlottal.core

        # Connection bewtween vocal folds and glottal flow

        F1_p = self.core.y.index(Vfold1.y[2])
        Flp = self.core.u.index(Gflow.u[2])
        self.core.add_connector((Flp, F1_p), alpha=-1)

        F2_p = self.core.y.index(Vfold2.y[2])
        Frp = self.core.u.index(Gflow.u[3])
        self.core.add_connector((Frp, F2_p), alpha=-1)

        # Connection of sub-glottal cavity
        P1sub = self.core.u.index(Vfold1.u[0])
        P1sub_2 = self.core.y.index(Subglottal.y[1])
        self.core.add_connector((P1sub, P1sub_2), alpha=-1)

        P2sub = self.core.u.index(Vfold2.u[0])
        P2sub_2 = self.core.y.index(Subglottal.y[2])
        self.core.add_connector((P2sub, P2sub_2), alpha=-1)

        Ptot_minus = self.core.u.index(Gflow.u[0])
        Ptot_minus_2 = self.core.y.index(Subglottal.y[3])
        self.core.add_connector((Ptot_minus, Ptot_minus_2), alpha=-1)

        # Connection of supra-glottal cavity
        Pac = self.core.y.index(Vtract.y[0])
        Pac2 = self.core.u.index(Supglottal.u[0])
        self.core.add_connector((Pac, Pac2), alpha=-1)

        P1sup = self.core.u.index(Vfold1.u[1])
        P1sup_2 = self.core.y.index(Supglottal.y[1])
        self.core.add_connector((P1sup, P1sup_2), alpha=1)

        P2sup = self.core.u.index(Vfold2.u[1])
        P2sup_2 = self.core.y.index(Supglottal.y[2])
        self.core.add_connector((P2sup, P2sup_2), alpha=1)

        Ptot_plus = self.core.u.index(Gflow.u[1])
        Ptot_plus_2 = self.core.y.index(Supglottal.y[3])
        self.core.add_connector((Ptot_plus, Ptot_plus_2), alpha=1)

        self.core.connect()
