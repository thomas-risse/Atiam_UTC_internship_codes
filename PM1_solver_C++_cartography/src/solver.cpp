#include "solver.h"

PM1Solver::PM1Solver(PHSModel& structure, int sr, double epsilon,
                     int maxIters) {
  this->structure = &structure;
  this->sr = sr;
  this->timeStep = 1 / static_cast<double>(sr);
  this->epsilon = epsilon;
  this->maxIters = maxIters;

  n_state = this->structure->get_n_state();
  n_diss = this->structure->get_n_diss();
  n_cons = this->structure->get_n_cons();
  n_io = this->structure->get_n_io();
  idx1 = n_state;
  idx2 = idx1 + n_diss;
  idx3 = idx2 + n_cons;
  full_size = this->structure->get_full_size();

  currentStates = VectorXd::Zero(n_state);
  currentFlowsEstimates = VectorXd::Zero(full_size);
  currentEffortsEstimates = VectorXd::Zero(full_size);
  currentCorrection = VectorXd::Zero(full_size);
  currentError = VectorXd::Zero(full_size);

  for (int i = 0; i < Quadrature4::order; i++) {
    statesQuadPoints[i] = VectorXd::Zero(n_state);
  }

  initJac();
  hessHProj = MatrixXd::Zero(n_state, n_state);
  gradZwProj = MatrixXd::Zero(n_diss, n_diss);
  gradZxProj = MatrixXd::Zero(n_diss, n_state);

  hessH = MatrixXd::Zero(n_state, n_state);
  gradZw = MatrixXd::Zero(n_diss, n_diss);
  gradZx = MatrixXd::Zero(n_diss, n_state);

  gradH = VectorXd::Zero(n_state);
  zw = VectorXd::Zero(n_diss);
}

void PM1Solver::initJac() {
  currentJac = MatrixXd::Zero(full_size, full_size);
  jacLeft = MatrixXd::Zero(full_size, full_size);
  jacRight = MatrixXd::Zero(full_size, full_size);
  jacLeft.block(0, 0, idx2, idx2) = MatrixXd::Identity(idx2, idx2);
  jacLeft.block(idx3, idx3, n_io, n_io) = MatrixXd::Identity(n_io, n_io);
  jacRight.block(idx2, idx2, n_cons, n_cons) =
      MatrixXd::Identity(n_cons, n_cons);
}

int PM1Solver::simulate(const VectorXd& init, const MatrixXd& input) {
  // Make sure that we start from fresh variables
  currentStates = VectorXd::Zero(n_state);
  currentFlowsEstimates = VectorXd::Zero(full_size);
  currentEffortsEstimates = VectorXd::Zero(full_size);
  currentCorrection = VectorXd::Zero(full_size);
  currentError = VectorXd::Zero(full_size);

  // std::cout << structure->getS() << std::endl;

  int steps = input.cols();
  // std::cout << "Simulation steps = " << steps << std::endl << std::endl;
  states = MatrixXd::Zero(n_state, steps + 1);
  flows = MatrixXd::Zero(full_size, steps);
  efforts = MatrixXd::Zero(full_size, steps);
  states.col(0) = init;
  currentStates = init;
  time = VectorXd::LinSpaced(steps, 0, timeStep * (steps - 1));

  for (int i = 0; i < steps; i++) {
    // Set input for current frame in effort vector
    currentEffortsEstimates(Eigen::seq(idx3, full_size - 1)) = input.col(i);
    // Solve for the step using Newton Raphson
    try {
      solveStep();
    } catch (ConvergenceError converror) {
      std::cout << converror.what() << " step " << i << std::endl;
      return 0;
    }

    // Compute and store states at end of frame
    currentStates += timeStep * getStates(currentFlowsEstimates);

    // Store results of the step
    states.col(i + 1) = currentStates;
    flows.col(i) = currentFlowsEstimates;
    efforts.col(i) = currentEffortsEstimates;
  };
  return 1;
}

void PM1Solver::solveStep() {
  int iter = 0;
  while (1) {
    for (int i = 0; i < Quadrature4::order; i++) {
      statesQuadPoints[i] = stateSynthesis(Quadrature4::quadPoints[i]);
    }
    computeEfforts();

    computeError();
    // Break if tolerance is reached
    if (currentError.cwiseAbs().maxCoeff() < epsilon) {
      break;
    }
    if (iter > maxIters) {
      throw ConvergenceError();
    }
    computeJac();
    solveLinear();
    applyCorrection();
    iter++;
  }
}

// Synthesis function
Eigen::VectorXd PM1Solver::stateSynthesis(double& tau) {
  return currentStates + timeStep * tau * getStates(currentFlowsEstimates);
}

// Intermediate compute functions
void PM1Solver::computeError() {
  currentError =
      currentFlowsEstimates - structure->getS() * currentEffortsEstimates;
  // std::cout << currentError << std::endl << std::endl;
}

void PM1Solver::computeEfforts() {
  // Computes efforts associated with current estimates of states
  // and dissipations. Efforts associated with inputs and lagrange multipliers
  // are considered already in the vector.
  currentEffortsEstimates(Eigen::seq(0, idx1 - 1)) =
      Quadrature4::quadWeights(0) *
      structure->gradH(statesQuadPoints[0], gradH);
  currentEffortsEstimates(Eigen::seq(idx1, idx2 - 1)) =
      Quadrature4::quadWeights(0) *
      structure->zDiss(currentFlowsEstimates(Eigen::seq(idx1, idx2 - 1)),
                       statesQuadPoints[0], zw);
  // accès à l'attribut statique 'order' de la classe Quadrature4.
  for (int i = 1; i < Quadrature4::order; i++) {
    // gradH Projection
    currentEffortsEstimates(Eigen::seq(0, idx1 - 1)) =
        currentEffortsEstimates(Eigen::seq(0, idx1 - 1)) +
        Quadrature4::quadWeights(i) *
            structure->gradH(statesQuadPoints[i], gradH);
    // Z(w) projection
    currentEffortsEstimates(Eigen::seq(idx1, idx2 - 1)) =
        currentEffortsEstimates(Eigen::seq(idx1, idx2 - 1)) +
        Quadrature4::quadWeights(i) *
            structure->zDiss(currentFlowsEstimates(Eigen::seq(idx1, idx2 - 1)),
                             statesQuadPoints[i], zw);
  }
}

void PM1Solver::computeJac() {
  // Projected hessian and gradients
  hessHProj = timeStep * Quadrature4::quadPoints[0] *
              Quadrature4::quadWeights[0] *
              structure->hessH(statesQuadPoints[0], hessH);

  gradZxProj = timeStep * Quadrature4::quadPoints[0] *
               Quadrature4::quadWeights[0] *
               structure->gradZDissStates(getDiss(currentFlowsEstimates),
                                          statesQuadPoints[0], gradZx);

  gradZwProj = Quadrature4::quadWeights[0] *
               structure->gradZDissWDiss(getDiss(currentFlowsEstimates),
                                         statesQuadPoints[0], gradZw);
  for (int i = 1; i < Quadrature4::order; i++) {
    hessHProj += timeStep * Quadrature4::quadPoints[i] *
                 Quadrature4::quadWeights[i] *
                 structure->hessH(statesQuadPoints[i], hessH);

    gradZxProj += timeStep * Quadrature4::quadPoints[i] *
                  Quadrature4::quadWeights[i] *
                  structure->gradZDissStates(getDiss(currentFlowsEstimates),
                                             statesQuadPoints[i], gradZx);

    gradZwProj += Quadrature4::quadWeights[i] *
                  structure->gradZDissWDiss(getDiss(currentFlowsEstimates),
                                            statesQuadPoints[i], gradZw);
  }
  jacRight.block(0, 0, n_state, n_state) = hessHProj;
  jacRight.block(n_state, 0, n_diss, n_state) = gradZxProj;
  jacRight.block(n_state, n_state, n_diss, n_diss) = gradZwProj;

  // Fill in jacobian matrix
  currentJac = jacLeft - structure->getS() * jacRight;
}

void PM1Solver::solveLinear() {
  currentCorrection = -currentJac.partialPivLu().solve(currentError);
}

void PM1Solver::applyCorrection() {
  currentFlowsEstimates.block(0, 0, idx2, 1) +=
      currentCorrection.block(0, 0, idx2, 1);
  currentEffortsEstimates.block(idx2, 0, n_cons, 1) +=
      currentCorrection.block(idx2, 0, n_cons, 1);
  currentFlowsEstimates.block(idx3, 0, n_io, 1) +=
      currentCorrection.block(idx3, 0, n_io, 1);
}

// linspace for generation of time vector

template <typename T>
std::vector<T> linspace(T a, T b, size_t N) {
  T h = (b - a) / static_cast<T>(N - 1);
  std::vector<T> xs(N);
  typename std::vector<T>::iterator x;
  T val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) *x = val;
  return xs;
}