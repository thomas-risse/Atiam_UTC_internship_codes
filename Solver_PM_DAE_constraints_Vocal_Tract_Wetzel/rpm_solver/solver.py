import numpy as np
import rpm_solver.quadrature as quad
import rpm_solver.basis as bf
import pickle


class RPMSolverPHS:
    """Solveur for RPM with constraints without regularization
    """

    def __init__(
        self,
        phs_struct,
        projection_order,
        regul_order,
        time_step,
        quad_order=10,
        epsilon=10 ** (-10),
        max_iter=100,
    ):
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
                               smooths in H^(regul_order). This parameter is for now ignored...
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
        self.sr = 1 / self.time_step

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
        self.struct = phs_struct

        """Projection parameters
        """
        # Projection and regularization order
        self.p_order = projection_order
        self.k_order = regul_order * 2  # Size of the regularization basis

        # Projection basis functions
        # Note : self.basis is a class instance which gives access to
        # evaluate_proj evaluate_regul and evaluate_all to evaluate
        # values of basis functions at points given in an arument x.
        self.basis = bf.ShiftedLegendrePolys(self)

        # Projection of the structure
        self.struct.build_p_struct(self.p_order)

        """             Pre-computations        """
        """Pre-computation of integrals and values of
        basis functions at points needed for quadrature
        """
        # Given that the quadrature order wont change, get quadrature points
        self.quad_points = self.quad.quad_roots
        # Pre compute values f basis functions at quadrature points
        self.basis.compute_basis_quad_points(self)
        # Pre compute integrals of basis functions between 0 and quadrature
        # points
        self.basis.compute_basis_integrals(self)
        # Pre computes matrix of the outer product of basis_quad_points
        # with basis_int to simplify hessian evaluation
        self.basis.compute_hessian_basis_term()

        """   Pre-allocation of jacobian matrix  """
        # Full matrix structure
        # [jac_xx, jac_xw, jac_xl, jax_xy],
        # [jac_wx, jac_ww, jac_wl, jax_wy],
        # [jac_lx, jac_lw, jac_ll, jax_ly],
        # [jac_yx, jac_yw, jac_yl, jac_yy]

        # Initialization of the matrix
        self.jac = np.zeros(
            (self.struct.proj_full_size, self.struct.proj_full_size)
        )

        # Pre-entering constant terms
        jac_xl = -self.struct.proj_Sxl @ np.eye(self.struct.proj_n_cons)

        jac_yl = -self.struct.proj_Syl @ np.eye(self.struct.proj_n_cons)

        jac_yy = np.eye(self.struct.proj_n_io)

        self.jac[
            : self.struct.proj_i_1, self.struct.proj_i_2 : self.struct.proj_i_3
        ] = jac_xl

        self.jac[
            self.struct.proj_i_3 :, self.struct.proj_i_2 : self.struct.proj_i_3
        ] = jac_yl

        self.jac[self.struct.proj_i_3 :, self.struct.proj_i_3 :] = jac_yy

    def simulate(self, init, duration, u_proj=None, save_results=None):
        """Simulates the system for the given duration
        with initialization given in init.

        Args:
            init (array): initial state values.
                Size:[self.struct.n_state]
            u_proj (array): values of projected inputs.
                Size:[self.Nframes, self.struct.n_io, self.p_order]
            duration (float): simulation duration
            save_results (string): Optionnal filename to save the results
                (numpy savez)

        Returns:
            arrays: states, projection coefficients of P for state variables,
                    projection coefficients of P for Larange multipliers,
                    projection coefficients of R
        """
        """Pre-allocation of the memory
        """
        # Number of simulation frames
        Nframes = int(duration / self.time_step)

        #self.jac_store = np.zeros(
        #    (
        #        Nframes * 100,
        #        self.struct.proj_full_size,
        #        self.struct.proj_full_size,
        #    )
        #)
        self.i_jac = 0

        if type(u_proj) != np.ndarray:
            u_proj = np.zeros((Nframes, self.struct.n_io, self.p_order))
        # Array to store states at each step
        x_frames = np.zeros((Nframes, self.struct.n_state))
        x_frames[0] = init

        # Arrays to store projected flows at each step

        # State flows
        dx_proj = np.zeros((Nframes, self.struct.n_state, self.p_order))
        # Flows of dissipative elements
        w_proj = np.zeros((Nframes, self.struct.n_diss, self.p_order))
        # Lagrange multipliers are on the other side of the equation...
        # Outputs
        y_proj = np.zeros((Nframes, self.struct.n_io, self.p_order))

        # Array to store lagrange multipliers projections at each step
        lambda_proj = np.zeros((Nframes, self.struct.n_cons, self.p_order))

        # Array to store number of NR iterations at each step
        iters = np.zeros(Nframes) + np.nan

        """Simulation loop
        """
        # First guess for newton-raphson
        dx_guess = dx_proj[0]
        w_guess = w_proj[0]
        y_guess = y_proj[0]
        lambda_guess = lambda_proj[0]
        for step in range(0, Nframes):
            # State at the begininin of the frame
            x0 = x_frames[step]
            try:
                (
                    dx_proj[step],
                    w_proj[step],
                    y_proj[step],
                    lambda_proj[step],
                    iters[step],
                ) = self._solve_newton_raphson(
                    x0, dx_guess, w_guess, y_guess, lambda_guess, u_proj[step]
                )
            except:
                print(f"Interruption at t = {self.time_step*step}")
                break
            # Compute state at end of frame
            if step != Nframes - 1:
                x_frames[step + 1] = x0 + self.time_step * dx_proj[step, :, 0]
            # Guess of the flows for the next frame
            dx_guess = dx_proj[step]
            w_guess = w_proj[step]
            y_guess = y_proj[step]
            lambda_guess = lambda_proj[step]

        if save_results is not None:
            self._save_results(
                save_results,
                x_frames,
                dx_proj,
                w_proj,
                u_proj,
                y_proj,
                lambda_proj,
                init,
                iters,
            )

        print(f"Mean number of NR iterations : {np.nanmean(iters)}")
        print(
            f"Max number of NR iterations : {np.nanmax(iters)},\
              step index : {np.nanargmax(iters[1:-1])}"
        )
        return x_frames, dx_proj, w_proj, y_proj, lambda_proj, iters

    def _solve_newton_raphson(
        self, x0, dx_guess, w_guess, y_guess, lambda_guess, u_proj
    ):
        """Finds the projection coefficients that satisfy the equations
        of the projected system for one step using newton raphson
        algorithm.

        Args:
            x0 (array): state at begining of frame.
                Size: [self.struct.n_state]
            dx_guess (array): guess for state projection coefficients.
                Size: [self.struct.n_state, self.p_order]
            w_guess (array): guess for dissipation projection coefficients.
                Size: [self.struct.n_diss, self.p_order]
            y_guess (array):guess for output projection coefficients.
                Size: [self.struct.n_io, self.p_order]
            lambda_guess (array):guess for lagrange multipliers
                projection coefficients.
                Size: [self.struct.n_cons, self.p_order]
            u_proj (array): input projection coefficients.
                Size: [self.struct.n_io, self.p_order]
        """
        # Initialize iteration counter
        iter = 0
        # Initialize solution values with guess
        dx = dx_guess
        w = w_guess
        y = y_guess
        l_mults = lambda_guess

        delta = np.zeros((self.struct.full_size, 1))

        while True:
            # We first compute the values of the states x,
            # dissipation parameters and dissipation
            # parameters w at quadrature points
            x_quad_points = self.basis.state_synthesis_quad_points(x0, dx)
            w_quad_points = self.basis.synthesis_quad_points(w)

            # Then, we compute the values of gradH and z(w,x) at
            # quadrature points
            gradH_quad_points = self.struct.grad_H(x_quad_points)
            zw_quad_points = self.struct.zw(w_quad_points, x_quad_points)

            # Finally, we compute projected gradient and projected z(w)
            # coefficients. Size: [self.struct.n_state, self.p_order] and
            # [self.struct.n_diss, self.p_order]
            proj_gradH = gradH_quad_points.T @ self.basis.w_basis_quad_points
            # print(np.amax(proj_gradH)/np.amin(proj_gradH))
            proj_zw = zw_quad_points.T @ self.basis.w_basis_quad_points
            # We can now compute the errors for the different parts
            # of the system
            error_dx = dx - (
                self.struct.Sxx @ proj_gradH
                + self.struct.Sxw @ proj_zw
                + self.struct.Sxl @ l_mults
                + self.struct.Sxu @ u_proj
            )
            error_w = w - (
                self.struct.Swx @ proj_gradH
                + self.struct.Sww @ proj_zw
                + self.struct.Swu @ u_proj
            )
            error_l_mults = -(
                self.struct.Slx @ proj_gradH + self.struct.Slu @ u_proj
            )
            error_out = y - (
                self.struct.Syx @ proj_gradH
                + self.struct.Syw @ proj_zw
                + self.struct.Syl @ l_mults
                + self.struct.Syu @ u_proj
            )

            # Complete vector of error
            error = np.concatenate(
                (
                    error_dx.flatten(),
                    error_w.flatten(),
                    error_l_mults.flatten(),
                    error_out.flatten(),
                )
            )

            # Relative error
            # rel_error = np.concatenate((error_dx.flatten()/(self.epsilon +dx.flatten()),
            #                            error_w.flatten()/(self.epsilon +w.flatten()),
            #                            error_l_mults.flatten()/((self.epsilon +l_mults.flatten())),
            #                            error_out.flatten()/(self.epsilon +y.flatten())))

            # Compute max error (absolute)
            max_error = np.amax(np.abs(error))

            # Break if tolerance is reached
            if max_error < self.epsilon:
                break
            error = error * (np.abs(error) > self.epsilon * 0.1)
            # Else, increment iteration counter
            iter += 1

            # And compute the jacobian matrix of the system

            # Values of hessian(H(x)) and grad(z(w, x)) at quadrature points
            # Size: [self.quad_order, self.struct.n_state, self.struct.n_state]
            hess_H_quad_points = self.struct.hess_H(x_quad_points)
            # Size: [self.quad_order, self.struct.n_diss, self.struct.n_diss]
            grad_zw_w_quad_points = self.struct.grad_zw_w(
                w_quad_points, x_quad_points
            )
            # Size: [self.quad_order, self.struct.n_diss, self.struct.n_state]
            grad_zw_x_quad_points = self.struct.grad_zw_x(
                w_quad_points, x_quad_points
            )
            # Values of the hessian of the projection of grad(H) with
            # respect to x
            # and of hessians of the projection of z(w,x) with
            # respect to x and w
            hess_proj_gradH = self.time_step * np.einsum(
                "aij, akl -> ikjl",
                hess_H_quad_points,
                self.basis.w_hess_integral_term,
            ).reshape(self.struct.proj_n_state, self.struct.proj_n_state)

            hess_proj_zw_x = self.time_step * np.einsum(
                "aij, akl -> ikjl",
                grad_zw_x_quad_points,
                self.basis.w_hess_integral_term,
            ).reshape(self.struct.proj_n_diss, self.struct.proj_n_state)

            hess_proj_zw_w = np.einsum(
                "aij, akl -> ikjl",
                grad_zw_w_quad_points,
                self.basis.w_hess_basis_term,
            ).reshape(self.struct.proj_n_diss, self.struct.proj_n_diss)

            # Update of the jacobian matrix
            # xx
            self.jac[: self.struct.proj_i_1, : self.struct.proj_i_1] = (
                np.eye(self.struct.proj_n_state)
                - self.struct.proj_Sxx @ hess_proj_gradH
                - self.struct.proj_Sxw @ hess_proj_zw_x
            )
            # xw
            self.jac[
                : self.struct.proj_i_1,
                self.struct.proj_i_1 : self.struct.proj_i_2,
            ] = (-self.struct.proj_Sxw @ hess_proj_zw_w)

            # wx
            self.jac[
                self.struct.proj_i_1 : self.struct.proj_i_2,
                : self.struct.proj_i_1,
            ] = (
                -self.struct.proj_Swx @ hess_proj_gradH
                - self.struct.proj_Sww @ hess_proj_zw_x
            )
            # ww
            self.jac[
                self.struct.proj_i_1 : self.struct.proj_i_2,
                self.struct.proj_i_1 : self.struct.proj_i_2,
            ] = (
                np.eye(self.struct.proj_n_diss)
                - self.struct.proj_Sww @ hess_proj_zw_w
            )

            # lx
            self.jac[
                self.struct.proj_i_2 : self.struct.proj_i_3,
                : self.struct.proj_i_1,
            ] = (-self.struct.proj_Slx @ hess_proj_gradH)

            # yx
            self.jac[self.struct.proj_i_3 :, : self.struct.proj_i_1] = (
                -self.struct.proj_Syx @ hess_proj_gradH
                - self.struct.proj_Syw @ hess_proj_zw_x
            )
            # yw
            self.jac[
                self.struct.proj_i_3 :,
                self.struct.proj_i_1 : self.struct.proj_i_2,
            ] = (-self.struct.proj_Syw @ hess_proj_zw_w)

            #self.jac_store[self.i_jac] = self.jac
            self.i_jac += 1
            # print(np.count_nonzero(self.jac)/self.struct.full_size**2)
            delta = np.linalg.solve(-self.jac, error).reshape(
                self.struct.full_size, self.p_order
            )
            # print(delta)
            # New estimates of unknowns
            dx = dx + delta[: self.struct.n_state]

            w = (
                w
                + delta[
                    self.struct.n_state : self.struct.n_state
                    + self.struct.n_diss
                ]
            )
            # display(sp.Matrix(self.jac))
            y = (
                y
                + delta[
                    self.struct.n_state
                    + self.struct.n_diss
                    + self.struct.n_cons :
                ]
            )

            l_mults = (
                l_mults
                + delta[
                    self.struct.n_state
                    + self.struct.n_diss : self.struct.n_state
                    + self.struct.n_diss
                    + self.struct.n_cons
                ]
            )

            # Raise error if the algorithm does not converge
            # in less than self.max_iter iterations.
            if iter > self.max_iter:
                raise Exception("Newton Raphson solver did not converge")

        return dx, w, y, l_mults, iter

    def power_bal(self, xframes, dx_proj, w_proj, u_proj, y_proj):
        # External power
        Pext = np.sum(np.sum(u_proj * y_proj, axis=1), axis=1).flatten()

        Pstored = np.zeros_like(Pext)
        Pdiss = np.zeros_like(Pext)

        proj_gradH = np.zeros((len(xframes), self.struct.n_state, self.p_order))
        for i in range(len(xframes)):
            # We first compute the values of the states x,
            # dissipation parameters and dissipation
            # parameters w at quadrature points
            x_quad_points = self.basis.state_synthesis_quad_points(
                xframes[i], dx_proj[i]
            )
            w_quad_points = self.basis.synthesis_quad_points(w_proj[i])

            # Then, we compute the values of gradH and z(w,x) at
            # quadrature points
            gradH_quad_points = self.struct.grad_H(x_quad_points)
            zw_quad_points = self.struct.zw(w_quad_points, x_quad_points)

            # Finally, we compute projected gradient and projected z(w)
            # coefficients. Size: [self.struct.n_state, self.p_order] and
            # [self.struct.n_diss, self.p_order]
            proj_gradH[i] = gradH_quad_points.T @ self.basis.w_basis_quad_points
            proj_zw = zw_quad_points.T @ self.basis.w_basis_quad_points

            Pstored[i] = np.sum(dx_proj[i] * proj_gradH[i])
            Pdiss[i] = np.sum(w_proj[i] * proj_zw)

        return Pstored, Pdiss, Pext, proj_gradH

    def _save_results(
        self,
        filename,
        x_frames,
        dx_proj,
        w_proj,
        u_proj,
        y_proj,
        lambda_proj,
        init,
        iters,
    ):
        """FUnction used to save the results of a simulation in a file in
        pickle format.

        Args:
            filename (string): name of the file.
            x_frames (array): state at begining of each frame.
            dx_proj (array): flow projection coefficients
            w_proj (array): dissipation variables projection coefficients.
            u_proj (array): inputs projection coefficients
            y_proj (array): outputs projection coefficients
            lambda_proj (array): constraints projection coefficients
            init (array): initial conditions
            iters (array): NR solver iterations
        """
        duration = len(x_frames) * self.time_step
        efforts_states = self.struct.grad_H(x_frames)

        Pstored, Pdiss, Pext, proj_gradH = self.power_bal(
            x_frames, dx_proj, w_proj, u_proj, y_proj
        )
        Ptot = Pstored + Pdiss + Pext
        t = np.linspace(0, len(x_frames) / self.sr, len(x_frames))

        simulation_parameters = {
            "p_order": self.p_order,
            "fs": self.sr,
            "duration": duration,
        }
        model_parameters = self.struct.get_parameters()
        full_parameters = dict(simulation_parameters, **model_parameters)
        to_write = {
            "Parameters": full_parameters,
            "Inputs": u_proj,
            "Outputs": y_proj,
            "States": x_frames,
            "States flows": dx_proj,
            "States Efforts": efforts_states,
            "Dissipations flows": w_proj,
            "Constraints": lambda_proj,
            "Init": init,
            "Pstored": Pstored,
            "Pdiss": Pdiss,
            "Pext": Pext,
            "Ptot": Ptot,
            "Time": t,
            "Projected gradH": proj_gradH,
        }
        with open(filename, "wb") as f:
            pickle.dump(to_write, f)
