import numpy as np
import sympy as sp

class QuadraticPhsSolverIndependantMatrices:
    """Class used to simulate a system put under a PHS form with a quadratic hamiltonian 
    with matrices independants of the state and without constraints
    """
    def __init__(self, J, R, L):
        """Initializes the simulator by computing all matrices necessaries for later.
        s_els is the number of storing elements of the system. io is the number of 
        input/output pairs. The method used to solve the system is the one presented 
        in the booklet §4.2, p.16.

        Args:
            J (array) : square matrix J of the system,
                        as presented in the booklet §2.2 (p.6)
            R (array) : square matrix R of size (s_els+io),
                        as presented in the booklet §2.2 (p.6)
            L (array) : square matrix of size s_els such that
                        H(x) = 0.5 x^T L x
        """        
        
        #Setting J, R and L as attributes of the class
        self.J = J
        self.R = R
        self.L = L

        #Deducing s_els and io from matrices sizes
        self.s_els = len(self.L) #number of storing elements
        self.full_size = len(self.J) #total system size
        self.io = self.full_size - self.s_els #number of input output pair

        #Computation of matrix M
        self.M = self.J - self.R

        """Division of matrix M into multiple parts used for computation
                M = [ Mxx  Mxu ]
                    [ Myx  Myu ]
        Note : Reshapes are used to ensure that arrays are 2 dimensionnal even if
        io or s_els is equal to 1.
        """
        self.Mxx = self.M[0:self.s_els, 0:self.s_els].reshape(self.s_els, self.s_els)
        self.Mxu = self.M[0:self.s_els, self.s_els:].reshape(self.s_els,self.io)
        self.Myx = self.M[self.s_els:, 0:self.s_els].reshape(self.io,self.s_els)
        self.Myu = self.M[self.s_els:, self.s_els:].reshape(self.io,self.io)
        
        self.A = self.Mxx @ self.L
        self.B = self.Mxu.view()

        #Utility reshapings used for vectorial computations of energies, etc... 
        #outside of the loop
        self.Myx = self.Myx.reshape(1, self.io, self.s_els)
        self.Myu = self.Myu.reshape(1, self.io, self.io)
        self.L = self.L.reshape(1, self.s_els, self.s_els) 


    def simulate(self, u_in, duration, sr, init_state = None):
        """Runs a simulation of the system for a given time using the 
        input u_in given as a function of time and a sampling rate

        Args:
            u_in (function): function returning the input values given a time array
                             The output of the function must be a 3D array of size (steps, io, 1)
            duration (float): duration of the simulation
            sr (float): sampling rate
            init_state (array): value of the state at t=0

        Returns:
            results (dict): dictionnary containing simulation results :

        """

        """ Creation of time array and control values array """

        #Sampling period
        dt = 1/sr 
        #Array of time for the simulation
        t = np.linspace(0, duration, int(duration * sr)) 
        #Number of simulation steps
        steps = len(t) 
        #Discretization of the input function
        ui = u_in(t) 


        """ Creation of arrays to store states and flows of storing elements """

        #Note : x and fx are 3 dimensionnal arrays so that for each step (first dimension),
        #   we have a column vector in the form of a 2D arrays that we can later transpose,
        #   which is not possible with 1D arrays

        #Setting 0 as initial conditions if not given
        if init_state.any() == None:
            init_state = np.zeros((self.s_els))

        #Array containing states of storing elements (column)
        x = np.zeros((steps, self.s_els, 1))
        #Initial state of storing elements 
        x[0,:,0] = init_state 
        #Array containing flows dx(step) / dt        (column)
        fx = np.zeros_like(x) 


        """ Computes matrix delta and its inverse """

        #delta is computed as stated in the booklet, p.16
        delta = np.identity(self.s_els) - dt/2 * self.A
        delta_inv = np.linalg.inv(delta)


        """ System dynamics computation """

        # As a quadratic hamiltonian is considered,
        # we use the method presented in the booklet §4.2 (p.16)
         
        for step in range(steps):
            #Computation of the flow for storing elements
            fx[step] = delta_inv @ (self.A @ x[step] + self.B @ ui[step]) 
            #Computation of the new state of storing elements(only if not last iteration)
            if step != steps-1:
                x[step+1] = x[step] + fx[step] * dt


        """ Reorganization and computing additional quantities outside of the loop """
        #Stored energy and derivative (effort associated to state variables x)
        H = (0.5 * x.reshape(steps, 1,self.s_els) @ self.L @ x).reshape(steps)
        dH = (self.L @ (x + 0.5*fx*dt)).reshape(steps, self.s_els)

        #Computation of the output
        y = ((self.Myx @ dH.reshape(steps, self.s_els, 1)) + self.Myu @ ui).reshape(steps,self.io) 

        #Flows as a 2D array (step, s_els)
        fx = fx.squeeze()

        #Efforts 
        ed = np.concatenate((dH, ui.reshape((steps, self.io))), axis = 1)

        #Powers
        Pstored = (dH.reshape(steps, 1, self.s_els) @ fx.reshape(steps, self.s_els, 1)).squeeze()

        Pext = (ui.reshape(steps, 1, self.io) @ y.reshape(steps, self.io, 1)).squeeze()

        Pdiss = (ed.reshape(steps, 1, self.full_size) \
                    @ self.R.reshape(1, self.full_size, self.full_size) \
                    @ ed.reshape(steps, self.full_size, 1)).squeeze()

        Ptot = Pstored + Pdiss + Pext


        """ Creation of a dictionnary containing all simulation data """

        results = {"State" : x.squeeze(), "Flows" : fx, "Pdiss" : Pdiss,
                    "Pstored" : Pstored, "Pext" : Pext, "Ptot" : Ptot,
                    "Estored" : H, "Efforts" : dH, "Input" : ui, "Output" : y, "Time" : t }

        return results


class QuadraticPhsSolver:
    """Class used to simulate a system put under a PHS form with a quadratic hamiltonian
    and without constraints
    """
    def __init__(self, Jfunc, Rfunc, L, s_els, io):
        """Initializes the simulator by computing all matrices necessaries for later.
        s_els is the number of storing elements of the system. io is the number of 
        input/output pairs. The method used to solve the system is the one presented 
        in the booklet §4.2, p.16.

        Args:
            Jfunc (function) : function that returns matrix Jq as defined in
                                4.3 (p.17) given the state q
            Jfunc (function) : function that returns matrix Rq as defined in
                                4.3 (p.17) given the state q
            L (array) : square matrix of size s_els such that
                        H(x) = 0.5 x^T L x
            s_els(int) : number of enery storing elements
            io (int) : number of input output pairs
        """  
        self.Jfunc = Jfunc
        self.Rfunc = Rfunc
        self.L = L

        self.s_els = s_els
        self.io = io
        self.full_size = self.io + self.s_els

    def Mfunc(self, x):
        """Returns matrix M = J-R given a state x

        Args:
            x (2D array): vector of state, must be of size s_els * 1

        Returns:
            M (2D array): matrix M
        """
        R = self.Rfunc(x)
        M = self.Jfunc(x) - R
        return M, R 

    def A(self, M):
        """Returns matrix A = Mxx L

        Args:
            M (2D array): M = J-R at current timestep

        Returns:
            A (2D array): Mxx L
        """
        return (M[0:self.s_els,0:self.s_els] @ self.L).reshape(self.s_els, self.s_els)
    
    def deltainv(self, A, dt):
        """Returns the inverse of matrix delta such that delta = I - (dt/2) A

        Args:
            A (2D array): matrix A
            dt (float): timestep

        Returns:
            deltainv (2D array): inverse of matrix delta
        """
        return np.linalg.inv(np.identity(self.s_els) - dt/2 * A).reshape(self.s_els, self.s_els)

    def B(self, M):
        """returns matrix B = Mxu

        Args:
            M (2D array): matrix M

        Returns:
            B (2D array): matrix B
        """
        return M[self.s_els:, 0:self.s_els].reshape(self.s_els,1)

    def dx(self, deltainv, A, B, x, u, dt):
        """returns state increment

        Args:
            deltainv (2D array): inverse of matrix delta
            A (2D array): matrix A
            B (2D array): matrix B
            x (2D array): state x
            u (2D array): control u
            dt (float): timestep

        Returns:
            dx (2D array): state increment
        """
        return dt * (deltainv @ (A @ x + B @ u))

    def simulate(self, u_in, duration, sr, init_state = None):
        """Runs a simulation of the system for a given time using the 
        input u_in given as a function of time and a sampling rate

        Args:
            u_in (function): function returning the input values given a time array
                             The output of the function must be a 3D array of size (steps, io, 1)
            duration (float): duration of the simulation
            sr (float): sampling rate
            init_state (array): value of the state at t=0

        Returns:
            results (dict): dictionnary containing simulation results :

        """
        """ Creation of time array and control values array """

        #Sampling period
        dt = 1/sr 
        #Array of time for the simulation
        t = np.linspace(0, duration, int(duration * sr)) 
        #Number of simulation steps
        steps = len(t) 
        #Discretization of the input function
        ui = u_in(t) 


        """ Creation of arrays to store states and flows of storing elements """

        #Note : x and fx are 3 dimensionnal arrays so that for each step (first dimension),
        #   we have a column vector in the form of a 2D arrays that we can later transpose,
        #   which is not possible with 1D arrays

        #Setting 0 as initial conditions if not given
        if init_state.any() == None:
            init_state = np.zeros((self.s_els))

        #Array containing states of storing elements (column)
        x = np.zeros((steps, self.s_els, 1))
        #Initial state of storing elements 
        x[0,:,0] = init_state 
        #Array containing flows dx(step) / dt        (column)
        fx = np.zeros_like(x) 
        #Array containing matrices M
        M = np.zeros((steps, self.full_size, self.full_size))
        #Array containing matrices R
        R = np.zeros_like(M)

        """ System dynamics computation """

        # As a quadratic hamiltonian is considered,
        # we use the method presented in the booklet §4.2 (p.16)
         
        for step in range(steps):
            #Computation of matrixes deltainv, A and B using current state
            M[step], R[step] = self.Mfunc(x[step])
            A = self.A(M[step])
            B = self.B(M[step])
            delta_inv = self.deltainv(A, dt)
            #Computation of the flow for storing elements
            fx[step] = delta_inv @ (A @ x[step] + B @ ui[step]) 
            #Computation of the new state of storing elements(only if not last iteration)
            if step != steps-1:
                x[step+1] = x[step] + fx[step] * dt


        """ Reorganization and computing additional quantities"""

        Myx = M[:, self.s_els:, 0:self.s_els]
        Myu = M[:, self.s_els:, self.s_els:]

        #Stored energy and derivative (effort associated to state variables x)
        H = (0.5 * x.reshape(steps, 1,self.s_els) @ self.L @ x).reshape(steps)
        dH = (self.L @ (x+0.5*fx*dt)).reshape(steps, self.s_els)

        #Computation of output
        y = ((Myx @ dH.reshape(steps, self.s_els, 1)) + Myu @ ui).reshape(steps,self.io) 

        #Flows as a 2D array (step, s_els)
        fx = fx.squeeze()

        #Efforts 
        ed = np.concatenate((dH, ui.reshape((steps, self.io))), axis = 1)

        #Powers
        Pstored = (dH.reshape(steps, 1, self.s_els) @ fx.reshape(steps, self.s_els, 1)).squeeze()

        Pext = (ui.reshape(steps, 1, self.io) @ y.reshape(steps, self.io, 1)).squeeze()

        Pdiss = (ed.reshape(steps, 1, self.full_size) \
                    @ R.reshape(steps, self.full_size, self.full_size) \
                    @ ed.reshape(steps, self.full_size, 1)).squeeze()

        Ptot = Pstored + Pdiss + Pext

        results = {"State" : x.squeeze(), "Flows" : fx, "Pdiss" : Pdiss,
                "Pstored" : Pstored, "Pext" : Pext, "Ptot" : Ptot,
                "Estored" : H, "Efforts" : dH, "Input" : ui, "Output" : y, "Time" : t }

        return results


class PHSQuadratiser:
    """This class allows to quadratise a given PHS using the method presented on the booklet 
    $4.4 (p.19)
    For the method implemented here to work, the hamiltonian must satisfy certain hyptotheses:
    (H1) : Separated variables
    (H2) : C1 regularity
    (H3) : locally quadractic at 0
    (H4) : strict quasi-convexity
    """
    def __init__(self, J, R, state, H):
        """Initialize the class and build matrices of functions M_q, J_q and R_q for the 
        quadratised system

        Args:
            J (2D array): matrix J of the system
            R (2D array): matrix R of the system
            state (array): numpy array of state variables (sympy variables)
            H (symbolic expression): sympy expression of the hamiltonian            
        """
        self.J = J
        self.R = R
        self.M = self.J-self.R

        #System size
        self.s_els = len(state)
        self.full_size = len(J)
        self.io = self.full_size-self.s_els

        #Symbolic expressions of state and hamiltonian with differenciation
        #between positive and negative values of the state in order to remove  
        #the sign operator from the change of state
        self.state = state
        self.stateNeg, self.statePos = self.separateStates()
        self.H = H

        self.HNeg = self.H
        self.HPos = self.H
        for i in range(self.s_els):
            self.HNeg = self.HNeg.subs(self.state[i], self.stateNeg[i])
            self.HPos = self.HPos.subs(self.state[i], self.statePos[i])

        #System size
        self.s_els = len(self.state)
        self.full_size = len(J)
        self.io = self.full_size-self.s_els

        #Separate hamiltonian
        self.HSepNeg, self.HSepPos = self.separateH()

        #Symbolic expression of the change of state
        self.QNeg, self.QPos = self.computeQ()

        #Symbolic expression of the jacobian matrix Jqx
        self.JqxNeg, self.JqxPos = self.computeJqx()

        #Symbolic expression of Jqx with added identity matrix 
        self.JqxTotNeg, self.JqxTotPos = self.computeJqxTot()

        #Compute M_q J_q and R_q !as functions of the original state!
        self.JxNeg, self.RxNeg, self.MxNeg, self.JxPos, self.RxPos, self.MxPos = self.computeJRMx()

    def separateStates(self):
        """This method creates two states variable arrays from the original one :
        one with positive values, the other with neative values

        Returns:
            array, array: state variable arrays for negative and positive values
                of the state
        """
        stateNeg = np.zeros_like(self.state, dtype = object)
        statePos = np.zeros_like(self.state, dtype = object)
        for i, state in enumerate(self.state):
            stateNeg[i] = sp.symbols(f'{state}_-', negative = True)
            statePos[i] = sp.symbols(f'{state}_+', positive = True)
        return stateNeg, statePos
    
    def separateH(self):
        """For the method implemented in this class to work, the hamiltonian
        must be separable. This function allows to separate the hamiltonian into
        s_els univariate functions of the state variables 

        Returns:
            array, array: arrays of separated hamiltonian functions for negative
                and positive values of the state
        """
        HSepNeg = np.zeros((self.s_els), dtype=object)
        HSepPos = np.zeros((self.s_els), dtype=object)
        for i in range(self.s_els):
            HSepNeg[i] = self.HNeg.as_independent(self.stateNeg[i])[1]
            HSepPos[i] = self.HPos.as_independent(self.statePos[i])[1]
        return HSepNeg, HSepPos

    def computeQ(self):
        """Returns an array of functions that represent the change of state
        Q applied to quadratise the system. Under Hypothesis (H1-4):
         Q(x_n) = sign(x_n)*sqrt(2*H_n(x_n))

        Returns:
            array, array: arrays of variable change functions Q for positive 
                and negative values of the state
        """
        QNeg = np.zeros((self.s_els), dtype=object)
        QPos = np.zeros((self.s_els), dtype=object)
        for i in range(self.s_els):
            QNeg[i] = -sp.sqrt(2*self.HSepNeg[i]).simplify()
            QPos[i] = sp.sqrt(2*self.HSepPos[i]).simplify()
        return QNeg, QPos

    def computeJqx(self):
        """Returns matrices Jqx for positive and negative values of the state

        Returns:
            Matrix, Matrix: Jqx for negative and positive values of the state
        """
        diagNeg = np.zeros((self.s_els), dtype=object)
        diagPos = np.zeros((self.s_els), dtype=object)
        for i in range(self.s_els):
            diagNeg[i] = sp.diff(self.QNeg[i], self.stateNeg[i]).simplify()
            diagPos[i] = sp.diff(self.QPos[i], self.statePos[i]).simplify()
        return sp.diag(*diagNeg), sp.diag(*diagPos)

    def computeJqxTot(self):
        """Extend matrix Jqx with identity matrix of size self.io

        Returns:
            Matrix, Matrix: Matrix Jqx with added identity matrix for negative and positive
            values of the state
        """
        diagJqxNeg = self.JqxNeg.diagonal()
        diagJqxPos = self.JqxPos.diagonal()
        diagIdentity = sp.ones(self.io,1)
        return sp.diag(*diagJqxNeg, *diagIdentity), sp.diag(*diagJqxPos, *diagIdentity)
    
    def computeJRMx(self):
        """Computes matrices Jq, Rq and Mq of the system as function of the original state.
        Before using them, the inverse transform X : q -> x must be applied to the state. 
        To differentiate the matrices from the one applied to the transformed state, they 
        are called Jx, Rq and Mx

        Returns:
            _type_: _description_
        """
        JxNeg = self.JqxTotNeg * self.J * self.JqxTotNeg.transpose()
        JxNeg.simplify()
        RxNeg = self.JqxTotNeg * self.R * self.JqxTotNeg.transpose()
        RxNeg.simplify()
        MxNeg = JxNeg-RxNeg
        MxNeg.simplify()
        JxPos = self.JqxTotPos * self.J * self.JqxTotPos.transpose()
        JxPos.simplify()
        RxPos = self.JqxTotPos * self.R * self.JqxTotPos.transpose()
        RxPos.simplify()
        MxPos = JxPos-RxPos
        MxPos.simplify()
        return JxNeg, RxNeg, MxNeg, JxPos, RxPos, MxPos
            
    def computeJRMq(self, XNeg, XPos):
        """Computes matrices Jq Rq and Mq using the funtion X : q -> x given

        Args:
            XNeg (array): array of sympy expressions of X(q) with q the transformed state 
                for negative values of X(q)
            XPos (array): array of sympy expressions of X(q) with q the transformed state 
                for positive values of X(q)

        """
        for i in range(self.s_els):
            self.JqNeg = self.JxNeg.subs(self.stateNeg[i], XNeg[i])
            self.JqPos = self.JxPos.subs(self.statePos[i], XPos[i])
            self.RqNeg = self.RxNeg.subs(self.stateNeg[i], XNeg[i])
            self.RqPos = self.RxPos.subs(self.statePos[i], XPos[i])
            self.MqNeg = self.MxNeg.subs(self.stateNeg[i], XNeg[i])
            self.MqPos = self.MxPos.subs(self.statePos[i], XPos[i])
            self.JqNeg.simplify()
            self.RqNeg.simplify()
            self.MqNeg.simplify()
            self.JqPos.simplify()
            self.RqPos.simplify()
            self.MqPos.simplify()
