#include "PHSModel.h"

double PHSModel::H(const VectorXd& states) {
  double out = 0;
  return this->H(states, out);
}

Eigen::VectorXd PHSModel::gradH(const VectorXd& states) {
  VectorXd out = VectorXd::Zero(get_n_state());
  return this->gradH(states, out);
}

Eigen::MatrixXd PHSModel::hessH(const VectorXd& states) {
  MatrixXd out = MatrixXd::Zero(get_n_state(), get_n_state());
  return this->hessH(states, out);
}

Eigen::VectorXd PHSModel::zDiss(const VectorXd& wDiss, const VectorXd& states) {
  VectorXd out = VectorXd::Zero(get_n_diss());
  return this->zDiss(wDiss, states, out);
}

Eigen::MatrixXd PHSModel::gradZDissStates(const VectorXd& wDiss,
                                          const VectorXd& states) {
  MatrixXd out = MatrixXd::Zero(get_n_diss(), get_n_state());
  return this->gradZDissStates(wDiss, states, out);
}

Eigen::MatrixXd PHSModel::gradZDissWDiss(const VectorXd& wDiss,
                                         const VectorXd& states) {
  MatrixXd out = MatrixXd::Zero(get_n_diss(), get_n_diss());
  return this->gradZDissWDiss(wDiss, states, out);
}

// Linear RLC
// Hamiltonian related functions
LinearRLC::LinearRLC(const float& C0, const float& L0, const float& R0) {
  this->S << 0, -1, 1, 0, 1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0;
  this->C0 = C0;
  this->L0 = L0;
  this->R0 = R0;
};

double LinearRLC::H(const VectorXd& states, double& out) {
  out = pow(states(0), 2) / (2 * this->C0) + pow(states(1), 2) / (2 * this->L0);
  return out;
}

Eigen::VectorXd LinearRLC::gradH(const VectorXd& states, VectorXd& out) {
  out(0) = states(0) / this->C0;
  out(1) = states(1) / this->L0;
  return out;
};
Eigen::MatrixXd LinearRLC::hessH(const VectorXd& states, MatrixXd& out) {
  out(0, 0) = 1 / this->C0;
  out(1, 1) = 1 / this->L0;
  return out;
};

// Dissipation related function
Eigen::VectorXd LinearRLC::zDiss(const VectorXd& wDiss, const VectorXd& states,
                                 VectorXd& out) {
  out(0) = wDiss(0) / this->R0;
  return out;
};
Eigen::MatrixXd LinearRLC::gradZDissStates(const VectorXd& wDiss,
                                           const VectorXd& states,
                                           MatrixXd& out) {
  out = MatrixXd::Zero(this->n_diss, this->n_state);
  return out;
};
Eigen::MatrixXd LinearRLC::gradZDissWDiss(const VectorXd& wDiss,
                                          const VectorXd& states,
                                          MatrixXd& out) {
  out(0, 0) = 1 / this->R0;
  return out;
};

/* Non linear RLC*/
NonLinearRLC::NonLinearRLC(const float& C0, const float& E0, const float& phi0,
                           const float& R0)
    : LinearRLC::LinearRLC(C0, L0, R0) {
  // this->S << 0, -1, 1, 0, 1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0;
  // this->C0 = C0;
  this->E0 = E0;
  this->phi0 = E0;
  // this->R0 = R0;
};

double NonLinearRLC::H(const VectorXd& states, double& out) {
  out = pow(states(0), 2) / (2 * this->C0) +
        this->E0 * log(cosh(states(1) / this->phi0));
  return out;
}

Eigen::VectorXd NonLinearRLC::gradH(const VectorXd& states, VectorXd& out) {
  out(0) = states(0) / this->C0;
  out(1) = this->E0 / this->phi0 * tanh(states(1) / this->phi0);
  return out;
};

Eigen::MatrixXd NonLinearRLC::hessH(const VectorXd& states, MatrixXd& out) {
  out(0, 0) = 1 / this->C0;
  out(1, 1) = this->E0 / pow(this->phi0, 2) *
              (1 - pow(tanh(states(1) / this->phi0), 2));
  return out;
};
