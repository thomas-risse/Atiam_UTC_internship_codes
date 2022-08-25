import numpy as np
import scipy.special as spe


class Basis():
    """Template class for basis generation and handling

    """
    def __init__(self, proj_order, regul_order, time_step):
        self.p_order = proj_order
        self.k_order = regul_order * 2
        self.full_size = self.k_order + self.p_order
        self.time_step = time_step

    def evaluate_proj(self, x):
        """Returns an array containing basis functions of the projection step
        evaluated for xi in x.

        Returns:
            2D array: array of basis functions evaluations.
                        Size : [len(x), p_order]
        """
        results = np.tile(x, (self.p_order, 1)).T +\
            np.tile(np.arange(self.p_order), (len(x), 1))
        return results

    def evaluate_regul(self, x):
        """Returns an array containing basis functions of the regularization step
        evaluated for xi in x.

        Returns:
            2D array: array of basis functions evaluations.
                        Size : [len(x), k_order]
        """
        results = np.tile(x, (self.k_order, 1)).T +\
            np.tile(np.arange(self.k_order), (len(x), 1))
        return results

    def evaluate_all(self, x):
        """Returns an array containing basis functions evaluated for
        xi in x.

        Returns:
            2D array: array of basis functions evaluations.
                        Size : [len(x), full_size]
        """
        results = np.tile(x, (self.full_size, 1)).T +\
            np.tile(np.arange(self.full_size), (len(x), 1))
        return results


class ShiftedLegendrePolys(Basis):
    """Basis functions class for Legendre Polynomials (L2 normalized).
    """
    def __init__(self, proj_order, regul_order, time_step):
        """Initialize the class with the basis
        size.

        Args:
            proj_order (int): order of the projection step
            regul_order (int): order of the regularization step
            time_step (float): timestep
        """
        super().__init__(proj_order, regul_order, time_step)
        # Construct matrices used to evaluate legendre polynomials
        # from arrays of powers of x
        self.A_proj, self.A_regul = self._build_A()
        # Construct transition matrix for regularization step
        self.M = self._build_M()
        # And multiply it by A_regul so that the regularization
        # basis can be simply evaluated from a vector of powers of x.
        if self.k_order == 0:
            self.MA_regul = np.zeros(0)
        else:
            self.MA_regul = self.M @ self.A_regul

        # Matrix to evaluate all basis functions at once
        self.A_full = np.zeros((self.full_size, self.full_size))
        self.A_full[0:self.p_order, 0:self.p_order] = self.A_proj
        self.A_full[self.p_order:, :] = self.MA_regul

        # Derivatives of the projection basis functions
        # (used during regularization)
        self.proj_diff_0, self.proj_diff_1 = self._build_proj_diff()

    def _build_A(self):
        A_proj = np.zeros((self.p_order, self.p_order))
        A_regul = np.zeros((self.k_order, self.full_size))

        for order in np.arange(self.p_order):
            A_proj[order] = np.sqrt(2*order+1) *\
                                   np.pad(np.flip(spe.sh_legendre(order)),
                                          [0, self.p_order - (order+1)])

        for i in range(self.k_order):
            poly_order = self.p_order + i
            A_regul[i] = np.pad(np.flip(spe.sh_legendre(poly_order)),
                                [0, self.full_size - (poly_order+1)])

        return A_proj, A_regul

    def _build_M(self):
        # To build the regularization basis, we need
        # k_order polynomials
        poly_coeffs = [np.pad(
                        np.flip(spe.sh_legendre(i)),
                        [0, self.full_size-(i+1)])
                       for i in range(self.p_order, self.full_size)]
        # Matrix of derivative values
        B_phi = np.zeros((self.k_order, self.k_order))
        # For each function of the regularization basis
        for i in range(self.k_order):
            # For each derivation order
            for j in range(int(self.k_order/2)):
                # Derivative of order j is equal to 0
                # if the order of the polynomial is less
                # than j
                if self.full_size >= j:
                    # Derivative at x=0
                    B_phi[i, j] = spe.factorial(j) * poly_coeffs[i][j] *\
                                  np.power(1/self.time_step, j)
                    # Derivative at x=1
                    factor = spe.factorial(np.arange(j, self.full_size)) / \
                        spe.factorial(np.arange(self.full_size-j))
                    B_phi[i, j+int(self.k_order/2)] = \
                        np.dot(factor, poly_coeffs[i][j:]) *\
                        np.power(1/self.time_step, j)

        # We can now compute the inverse of B_phi which gives us
        # a change of base
        M = np.linalg.pinv(B_phi.T).T
        return M

    def evaluate_proj(self, x):
        """Evaluates all basis functions of the projection
        step at points given in x.

        Args:
            x (array): evaluation points

        Returns:
            array: basis function evaluations. Size: [self.proj_order, len(x)]
        """
        # Powers of x needed
        x_pows = x ** np.outer(np.arange(self.p_order), np.ones(len(x)))
        return self.A_proj @ x_pows

    def evaluate_regul(self, x):
        """Evaluates all basis functions of the regularization
        step at points given in x.

        Args:
            x (array): evaluation points

        Returns:
            array: basis function evaluations. Size: [self.k_order, len(x)]
        """
        # Powers of x needed
        x_pows = x ** np.outer(np.arange(self.full_size), np.ones(len(x)))
        return self.MA_regul @ x_pows

    def evaluate_all(self, x):
        """Evaluates all basis functions
        at points given in x.

        Args:
            x (array): evaluation points

        Returns:
            array: basis function evaluations. Size: [self.full_size, len(x)]
        """
        # Powers of x needed
        x_pows = x ** np.outer(np.arange(self.full_size), np.ones(len(x)))
        return self.A_full @ x_pows

    def _build_proj_diff(self):
        """Computes derivatives of basis functions
        of the projection step at tau =0 and tau = 1 up
        to order self.k_order/2

        Returns:
            array: derivatives at tau = 0.
                   Size: [self.k_order / 2, self.p_order]
            array: derivatives at tau = 1.
                   Size: [self.k_order / 2, self.p_order]
        """
        diff_order = int(self.k_order / 2)
        # To build the regularization basis, we need
        # k_order polynomials
        poly_coeffs = [np.pad(
                        np.flip(spe.sh_legendre(i)) * np.sqrt(2*i+1),
                        [0, self.p_order-(i+1)])
                       for i in range(self.p_order)]
        # Matrix of derivative values
        diff_0 = np.zeros((diff_order, self.p_order))
        diff_1 = np.zeros((diff_order, self.p_order))
        # For each function of the projection basis
        for i in range(self.p_order):
            # For each derivation order
            for j in range(diff_order):
                # If the derivation order is more than the
                # maximum order of the porjection basis,
                # derivatives are equal to 0.
                if j <= self.p_order-1:
                    # Derivative at x=0
                    diff_0[j, i] = spe.factorial(j) * poly_coeffs[i][j] *\
                                  np.power(1/self.time_step, j)
                    # Derivative at x=1
                    factor = spe.factorial(np.arange(j, self.p_order)) / \
                        spe.factorial(np.arange(self.p_order-j))
                    diff_1[j, i] = \
                        np.dot(factor, poly_coeffs[i][j:]) *\
                        np.power(1/self.time_step, j)
        return diff_0, diff_1

    def get_proj_diff_0(self, dx, order):
        """Given the projection coefficients
        dx, returns the derivative of the sum of
        projected flows for the projection step at
        tau=0.
        n is the number of state for which the evaluation
        is needed.

        Args:
            dx (array): array of projection coefficients.
                        Size: [n, p_order]
            order (int): derivation order (0 means no derivation)

        Return:
            array: array of derivatives of the flow at t=0
                   Size: [n]
        """
        return np.dot(dx, self.proj_diff_0[order])

    def get_proj_diff_1(self, dx, order):
        """Given the projection coefficients
        dx, returns the derivative of the sum of
        projected flows for the projection step at
        tau=0.
        n is the number of state for which the evaluation
        is needed.

        Args:
            dx (array): array of projection coefficients.
                        Size: [n, proj_order]
            order (int): derivation order (0 means no derivation)

        Return:
            array: array of derivatives of the flow at t=0
                   Size: [n]
        """
        return np.dot(dx, self.proj_diff_1[order])
