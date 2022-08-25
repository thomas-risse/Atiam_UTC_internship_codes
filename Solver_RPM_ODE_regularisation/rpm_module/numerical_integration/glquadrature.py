"""Module to compute numerical integration
using gauss-legendre quadrature
"""
import numpy as np
import scipy.special as spe


class GLQuad():
    """General class to handle shifted orthonormal
    Legendre polynomials
    """
    def __init__(self, quad_order):
        """Initialization of the class

        Args:
            order (int): maximum polynomial order to compute
                        (used for pre-computation of weights)
        """
        self.quad_order = quad_order
        # Pre compute Roots and weights for the quadrature
        self.quad_roots, self.quad_weights = self._compute_roots_weights()

    def _compute_roots_weights(self):
        """Compute and store roots and weights of the shifted Legendre
        polynomial of order quad_order. They can later
        be used to perform Gaussian-Legendre quadrature
        over [0, 1]
        """
        original_roots, weights = spe.roots_legendre(self.quad_order)
        shifted_roots = original_roots / 2 + 0.5
        weights = weights / 2
        return shifted_roots, weights

    def quadrature(self, f):
        """Integrates function f between 0 and 1
        using Gauss-Legendre quadrature

        Args:
            f (function): function to integrate, must be able to
                        evaluate an array of value
        """
        # Evaluating function at the quadrature points
        f_eval = f(self.quad_roots)
        # Weighted sum
        return np.dot(f_eval, self.quad_weights)

    def quadrature_array(self, f):
        """Integrates function f between 0 and 1
        using Gauss-Legendre quadrature. f must be an array containing
        values of the function f estimated at points self.quad_roots.

        Args:
            f (2D array): array of values of the function to
                integrate evaluated at the quadrature points
                (can be obtained using self.quad_rules).
                Size : [n, quad_order] where n corresponds to different arrays
                to evaluate.
        """
        # Return weighted sum
        return np.dot(f, self.quad_weights)

    def integrate(self, f, xmin, xmax):
        """Integrates function f between a and b using
        Gauss-Legendre quadrature of order quad_order

        Args:
            f (function): function to integrate
            xmin (float): lower bound
            xmax (float): upper bound

        Returns:
            float: result
        """
        def fshifted(x):
            return f((xmax-xmin)*x + xmin)
        return (xmax-xmin) * self.quadrature(fshifted)


    def quadrature_convergence(self, f, solution, max_order):
        """Computes approximation error relative
        to the analytic integral solution for orders up to
        self.quad_order using Gauss-Leendre quadrature. Integration
        is performed between 0 and 1.
    
        Args:
            f (function): function to integrate
            solution (float): analytic solution
            max_order (int): maximum order of the converence analysis.
    
        Returns:
            array: approximation error for each order
        """
        approx_results = np.zeros(self.quad_order)
        for i in range(max_order):
            order = i + 1
            integrator = GLQuad(order)
            approx_results[i] = integrator.quadrature(f)
        approx_error = np.abs((solution - approx_results) / solution)
        return approx_error
