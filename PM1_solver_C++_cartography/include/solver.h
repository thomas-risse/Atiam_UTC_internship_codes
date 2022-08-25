#ifndef SOLVER
#define SOLVER

#include <cmath>
#include <eigen-3.4.0/Eigen/Core>
#include <eigen-3.4.0/Eigen/Dense>
#include <iostream>
#include <vector>

#include "GaussQuadrature.h"
#include "PHSModel.h"

class PM1Solver {
  // Solver for RPM with p=1 and no regularization
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;

 private:
  // Structure
  PHSModel* structure;

  // Copy of the sizes for ease
  int n_state;
  int n_diss;
  int n_cons;
  int n_io;
  int idx1;
  int idx3;
  int idx2;
  int full_size;

  // Quadrature class
  // Pas d'objet Quadrature4, on appelle ses méthodes directement.

  // sr
  int sr;
  double timeStep;

  // Convergence tolerance
  double epsilon;

  // Max iter
  int maxIters;

  // Pre-allocation of intermediate vectors and matrices
  VectorXd currentStates;            // n_states
  VectorXd currentError;             // full_size
  VectorXd currentFlowsEstimates;    // full_size
  VectorXd currentEffortsEstimates;  // full_size
  VectorXd currentCorrection;        // full_size

  VectorXd statesQuadPoints[Quadrature4::order];  // n_states

  VectorXd gradH;
  VectorXd zw;

  MatrixXd hessH;
  MatrixXd gradZw;
  MatrixXd gradZx;

  MatrixXd hessHProj;
  MatrixXd gradZwProj;
  MatrixXd gradZxProj;

  MatrixXd currentJac;  // full_size, full_size
  MatrixXd jacLeft;
  MatrixXd jacRight;

  // Storage for solutions
  MatrixXd states;
  MatrixXd flows;
  MatrixXd efforts;
  VectorXd time;

  // Functions used in simulation
  void solveStep();

  VectorXd stateSynthesis(double& tau);

  void computeError();

  void solveLinear();

  void computeEfforts();

  void computeJac();

  void applyCorrection();

  // Intialization of jacobian matrix
  void initJac();

 public:
  // Constructor
  PM1Solver(PHSModel& structure, int sr, double epsilon, int maxIters);

  // Simulation
  int simulate(const VectorXd& init, const MatrixXd& input);

  inline const VectorXd getStates(const VectorXd& inputVec) {
    return inputVec(Eigen::seq(0, idx1 - 1));
  };
  inline const VectorXd getDiss(const VectorXd& inputVec) {
    return inputVec(Eigen::seq(idx1, idx2 - 1));
  };
  inline const VectorXd getCons(const VectorXd& inputVec) {
    return inputVec(Eigen::seq(idx2, idx3 - 1));
  };
  inline const VectorXd getIo(const VectorXd& inputVec) {
    return inputVec(Eigen::seq(idx3, full_size - 1));
  };

  void setStates(VectorXd& inputVec, const VectorXd& states) {
    inputVec(Eigen::seq(0, idx1 - 1)).noalias() = states;
  };
  void setDiss(VectorXd& inputVec, const VectorXd& diss) {
    inputVec(Eigen::seq(idx1, idx2 - 1)).noalias() = diss;
  };
  void setCons(VectorXd& inputVec, const VectorXd& cons) {
    inputVec(Eigen::seq(idx2, idx3 - 1)).noalias() = cons;
  };
  void setIo(VectorXd& inputVec, const VectorXd& io) {
    inputVec(Eigen::seq(idx3, full_size - 1)).noalias() = io;
  };

  PHSModel* getStructure() { return structure; };

  std::map<std::string, double> getSolverParameters() {
    std::map<std::string, double> params = {{"epsilon", epsilon},
                                            {"maxIters", maxIters},
                                            {"fs", sr},
                                            {"timestep", timeStep}};
    return params;
  };

  inline MatrixXd getStatesResults() { return states; }
  inline MatrixXd getFlowsResults() { return flows; }
  inline MatrixXd getEffortsResults() { return efforts; }
  inline VectorXd getTimeVect() { return time; }
  inline int getNStates() { return n_state; }
  inline int getNdiss() { return n_diss; }
  inline int getNcons() { return n_cons; }
  inline int getNio() { return n_io; }
};

class ConvergenceError : public std::exception {
 public:
  std::string what() { return "Newton Raphson solver did not converge"; }
};

template <typename T>
std::vector<T> linspace(T a, T b, size_t N);

#endif