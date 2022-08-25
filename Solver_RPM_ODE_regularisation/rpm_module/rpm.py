import numpy as np
import rpm_module.basis_functions as bf
import rpm_module.numerical_integration as quad
import rpm_module.gradient_projection as proj
import rpm_module.hamiltonian as Ham


class RPMSolverPHS:
    """Blablabla
    """
    def __init__(self, phs_struct,
                 projection_order, regul_order, time_step, quad_order=10,
                 epsilon=10**(-10), max_iter=100):
        """Initialize the class

        Args:
            phs_struct (dict): dictionnary containing:
                    "S": interconnexion matrix
                    "States": array of sympy states variables
                    "H": hamiltonian sympy expression as a function of only
                         the state variables
                    "Constraints": number of constraints of the system

            projection_order (int): number of functions in the projection basis
            regul_order (int): regularization order, so that solutions are
                               smooths in H^(regul_order)
            time_step (float): step size
            quad_order (int, optional): quadrature order. Defaults to 10.
            epsilon (float, optional): absolute converence tolerance.
                                        Default to 10^(-10)
            max_iter (int, optional): maximum number of Newton Raphson
                iterations
        """
        """Solver parameters
        """
        # Time step
        self.time_step = time_step

        # Tolerance fo Newton-Raphson convergence
        self.epsilon = epsilon
        # Maximum Newton-Raphson iterations before
        # throwing an error
        self.max_iter = max_iter

        # Quadrature order for integrations
        self.quad_order = quad_order
        # Initialization of the quadrature class
        self.quad = quad.GLQuad(self.quad_order)

        """PHS system description
        """
        # Interconnection matrix
        self.S = phs_struct["S"]

        # Sympy Hamiltonian and states
        self.H_sp = phs_struct["H"]
        self.states = phs_struct["States"]

        # Number of state variables (before projection)
        self.n_state = len(self.states)

        # Number of constraints
        self.n_constraints = phs_struct["Constraints"]
        # Full size unprojected
        self.full_size_unproj = self.n_constraints + self.n_state

        # Hamiltonian evaluation from numpy array of states
        self.H = Ham.build_H_np(self.H_sp, self.states)

        """Projection parameters
        """
        # Projection and regularization order
        self.p_order = projection_order
        self.k_order = regul_order * 2  # Size of the regularization basis

        # Projection basis functions
        # Note : self.basis is a class instance which gives access to
        # evaluate_proj evaluate_regul and evaluate_all to evaluate
        # values of basis functions at points given in an arument x.
        self.basis = bf.ShiftedLegendrePolys(self.p_order, regul_order,
                                             self.time_step)

        # System size during projection step
        self.H_proj_size = self.p_order * self.n_state
        self.cons_proj_size = self.p_order * self.n_constraints

        # System full size
        self.full_size = (self.p_order + self.k_order) * self.full_size_unproj

        """             Pre-computations        """

        """Hamiltonian derivatives
        """
        # Needed derivatives of the hamiltonian.
        # To perform newton raphson on the projection step,
        # we need the first 2 derivatives. Then, if the regularization
        # order is more than 1 (k_order>2), we need more derivatives.
        self.H_diffs = Ham.compute_diffs(self.H_sp,
                                         self.states,
                                         max(2, int(self.k_order/2)+1))
        # For ease, we make copies of first and second derivatives
        # Gradient of the hamiltonian
        self.gradients = self.H_diffs[0]
        # Hessian of the hamiltonian
        self.hessian = self.H_diffs[1]

        # Given that the quadrature order wont change, get quadrature points
        self.quad_points = self.quad.quad_roots

        """Extended interconnexion matrix
        """
        # Build matrix S for the projected system
        self.S_proj = np.kron(self.S, np.eye(self.p_order))

        """Pre-computation of integrals and values of
        basis functions at points needed for quadrature
        """
        # Pre compute values f basis functions at quadrature points
        self.basis_quad_points = self._compute_basis_quad_points()
        # Pre compute integrals of basis functions between 0 and quadrature
        # points
        self.basis_int = self._compute_basis_integrals()
        # Pre computes matrix of the outer product of basis_quad_points
        # with basis_int to simplify hessian evaluation
        self.hessian_basis_term = proj.hessian_basis_term(self)

    def _compute_basis_quad_points(self):
        """Evaluates basis functions at quadrature points to ease
        computations later and reduce calls to functions

        Returns:
            2D array: array containin values of the basis functions
                        evaluated at quadrature points.
                        Size: [self.p_order, self.quad_order]
        """
        return self.basis.evaluate_proj(self.quad_points)

    def _compute_basis_integrals(self):
        """Pre compute integration of basis functions
        between 0 and the quadrature points to ease computation
        of intermediate points in the frames
        """
        integrals = np.zeros((self.p_order, self.quad_order))
        for j, point in enumerate(self.quad_points):
            integrals[:, j] = \
                self.quad.integrate(
                    self.basis.evaluate_proj,
                    0, point
                    )
        return integrals

    def _solve_newton_raphson(self, x0, dx_init, f_and_jac):
        """Solves the system f(x0, dx) = 0.

        Args:
            x0 (array): state at the begining of the frame.
                        Size: [self.n_state]
            dx_init (2D array): projected flows values for initialization
                            Size: [self.full_size_unproj, self.p_order]
            f_and_jac (function): function taking self, x0 and dx as arguments
                                  and returning [f, jac] where f is the vector
                                  f(x0, dx) and jac is the jacobian of the
                                  problem.


        Raises:
            Exception: If alorithm does not converge.

        Returns:
            2D array: Projected state flows solution.
                Size[self.full_size_unproj, self.p_order]
            int: number of iteration to converge
        """
        # Initialize iteration counter
        iter = 0
        # Initialize solution value with dxInit
        dx = dx_init
        while True:
            error, jac = f_and_jac(self, x0, dx)
            # Break if tolerance is reached
            if (np.max(np.abs(error)) < self.epsilon):
                break
            iter += 1
            # Compute correction
            delta = -np.linalg.pinv(jac) @ error
            # New estimates of dx
            dx = dx + delta.reshape(self.full_size_unproj, self.p_order)
            # Raise error if the algorithm does not converge
            # in less than self.max_iter iterations.
            if iter > self.max_iter:
                raise Exception("Newton Raphson solver did not converge")
        return dx, iter

    def simulate(self, init, duration):
        """Simulates the system for the given duration
        with initialization given in init.

        Args:
            init (array): initial state values
            duration (float): simulation duration

        Returns:
            arrays: states, projection coefficients of P for state variables,
                    projection coefficients of P for Larange multipliers,
                    projection coefficients of R
        """

        Nframes = int(duration / self.time_step)

        # Array to store states at each step
        x_frames = np.zeros((Nframes, self.n_state))
        x_frames[1] = init
        # Array to store projected states flows at each step
        dx_proj = np.zeros((Nframes, self.n_state, self.p_order))
        # Array to store regularization coeffs at each step
        dx_regul = np.zeros((Nframes, self.n_state, self.k_order))

        # Array to store lagrande multipliers projections at each step
        l_mults = np.zeros((Nframes, self.n_constraints, self.p_order))

        # Array to store number of NR iterations at each step
        iters = np.zeros(Nframes)

        for step in range(1, Nframes-1):
            # Find dx
            x0 = x_frames[step]
            # Use last projected flows as estimate for first newton raphson
            # iteration
            f_guess = np.concatenate((dx_proj[step-1], l_mults[step-1]),
                                     axis=0)
            # Solve system for the frame
            proj_coeffs, iters[step] =\
                self._solve_newton_raphson(x0, f_guess, proj.f_and_jac_quad)
            # Split solution in different parts
            dx_proj[step] = proj_coeffs[0:self.n_state]
            l_mults[step] = proj_coeffs[self.n_state:]
            # Compute state at end of frame
            x_frames[step+1] = x0 + self.time_step * dx_proj[step, :, 0]
            x1 = x_frames[step+1]

            # Regularization
            dx_regul[step] = proj.regularize(self, x0, x1, proj_coeffs)

        print(f"Mean number of NR iterations : {np.mean(iters[1:-1])}")
        print(f"Max number of NR iterations : {np.max(iters[1:-1])},\
              step index : {np.argmax(iters[1:-1])}")
        return x_frames[1:], dx_proj[1:], l_mults[1:], dx_regul[1:-1]


class linear_order1_solver:
    def __init__(self, phs_struct, time_step):
        self.struct = phs_struct
        self.S = self.struct["S"]
        self.L = self.struct["L"]
        self.n_state = 2
        self.time_step = time_step

        self.step_mat = self.build_stepping_matrix()

    def H(self, x):
        return np.diag(0.5 * x @ self.L @ x.T)

    def build_stepping_matrix(self):
        delta = np.eye(self.n_state) - self.time_step*self.S @ self.L / 2
        delta_inv = np.linalg.pinv(delta)
        A = self.S @ self.L
        return delta_inv @ A

    def simulate(self, init, duration):
        """Simulates the system for the given duration
        with initialization given in init.

        Args:
            init (array): initial state values
            duration (float): simulation duration

        Returns:
            arrays: states, projection coefficients of P for state variables,
                    projection coefficients of P for Larange multipliers,
                    projection coefficients of R
        """

        Nframes = int(duration / self.time_step)

        # Array to store states at each step
        x_frames = np.zeros((Nframes, self.n_state))
        x_frames[1] = init
        # Array to store projected state flow at each step
        dx_proj = np.zeros((Nframes, self.n_state))


        for step in range(1, Nframes-1):
            # Find dx
            x0 = x_frames[step]
            # Split projection coefficients
            dx_proj[step] = self.step_mat @ x0
            # Compute state at end of frame
            x_frames[step+1] = x0 + self.time_step * dx_proj[step, :]

        return x_frames[1:], dx_proj[1:]