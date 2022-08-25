#include "VocGSI.h"

// Vocal Apparatus
// Hamiltonian related functions
VocalApparatusGSI1::VocalApparatusGSI1(std::map<std::string, double> Params) {
  // Parameters
  // Glottal flow
  this->rho0 = Params["rho0"];  // kg.m-3
  this->L0 = Params["L0"];      // m
  this->l0 = Params["l0"];      // m
  this->h0 = Params["h0"];      // m
  this->hr = Params["hr"];      // m

  this->rholL2 = 2 * rho0 * l0 * L0;     // kg.m-1 2*rho0*l0*L0
  this->rholL2h3 = rholL2 * pow(h0, 3);  // kg.m^2 2*rho0*L0*l0*h0**3

  // Vocal folds
  this->k0 = Params["k0"];          // N.m-1
  this->k1 = Params["k1"];          // N.m-1
  this->kappa0 = Params["kappa0"];  // N.m-1
  this->kappa1 = Params["kappa1"];  // N.m-1

  this->m = Params["m"];          // kg
  this->r = Params["r"];          // kg.s-1
  this->S_sub = Params["S_sub"];  // m^2
  this->S_sup = Params["S_sup"];  // m^2

  // Vocal tract impedance
  this->a0 = Params["a0"];  //
  this->q0 = Params["q0"];  // ohm
  this->w0 = Params["w0"];  // rad.s-1

  S = Eigen::MatrixXd::Zero(full_size, full_size);
  S(0, 1) = -1;
  S(0, 2) = 1;
  S(0, 10) = -S_sup;
  S(0, 12) = -1;
  S(0, 18) = -S_sub;

  S(2, 8) = -1;
  S(2, 15) = 1;

  S(3, 4) = -1;
  S(3, 5) = 1;
  S(3, 10) = -S_sup;
  S(3, 13) = -1;
  S(3, 18) = -S_sub;

  S(5, 8) = -1;
  S(5, 15) = -1;

  S(6, 10) = -L0 * h0;
  S(6, 14) = -L0 * h0;
  S(6, 18) = L0 * h0;

  S(7, 16) = 1;

  S(8, 9) = -2;
  S(8, 10) = 2 * l0 * L0;
  S(8, 14) = 2 * l0 * L0;
  S(8, 18) = 2 * l0 * L0;

  S(10, 11) = -1;
  S(10, 17) = -1;

  S = (S - S.transpose()).eval();
};

double VocalApparatusGSI1::H(const VectorXd& states, double& out) {
  // First vocal fold
  out = 0.5 * (states(0) * states(0) / m + states(1) * states(1) * k0 +
               states(2) * states(2) * kappa0);
  // Second vocal fold
  out += 0.5 * (states(3) * states(3) / m + states(4) * states(4) * k1 +
                states(5) * states(5) * kappa1);
  // Glottal flow
  double m = rho0 * (hr + states(9));
  double m33 =
      m / 12 * (1 + 4 * l0 * l0 / ((hr + states(9)) * (hr + states(9))));
  out += pow(((hr + states(9)) / h0), 2) *
             (states(6) * states(6) + states(7) * states(7)) / (2 * rho0 * h0) +
         states(8) * states(8) / (8 * m33);
  // Vocal tract
  out += 0.5 * (states(10) * states(10) * a0 +
                states(11) * states(11) * w0 * w0 / a0);
  return out;
}

Eigen::VectorXd VocalApparatusGSI1::gradH(const VectorXd& states,
                                          VectorXd& out) {
  // First vocal fold
  out(0) = states(0) / m;
  out(1) = states(1) * k0;
  out(2) = states(2) * kappa0;
  // Second vocal fold
  out(3) = states(3) / m;
  out(4) = states(4) * k1;
  out(5) = states(5) * kappa1;

  // Glottal flow
  // double pix = states(6);
  // double piy = states(7);
  // double piexp = states(8);
  // double h = states(9);
  double htot = hr + states(9);
  double htot2 = htot * htot;
  double term1 = 4 * l0 * l0 / htot2 + 1;
  out(Eigen::seq(6, 7)) = states(Eigen::seq(6, 7)) * htot2 / rholL2h3;
  out(8) = 3 * states(8) / (rholL2 * htot * term1);
  out(9) = (states(6) * states(6) + states(7) * states(7)) * 2 * htot /
               (2 * rholL2h3) +
           3 * states(8) * states(8) *
               (2 * l0 / (L0 * rho0 * pow(htot2, 2) * term1 * term1) -
                1 / (2 * rholL2 * htot2 * term1));

  // Vocal tract
  out(10) = states(10) * a0;
  out(11) = states(11) * w0 * w0 / a0;
  return out;
};
Eigen::MatrixXd VocalApparatusGSI1::hessH(const VectorXd& states,
                                          MatrixXd& out) {
  // As both vocal folds and the vocal tract are linear in this model,
  // we could initialize the matrix accordingly...
  // First vocal fold
  out(0, 0) = 1 / m;
  out(1, 1) = k0;
  out(2, 2) = kappa0;
  // Second vocal fold
  out(3, 3) = 1 / m;
  out(4, 4) = k1;
  out(5, 5) = kappa1;

  // Glottal flow
  // double pix = states(6);
  // double piy = states(7);
  // double piexp = states(8);
  // double h = states(9);
  double htot = states(9) + hr;
  double term1 = 4 * l0 * l0 / (htot * htot) + 1;

  out(6, 6) = htot * htot / rholL2h3;
  out(6, 9) = 2 * htot * states(6) / rholL2h3;

  out(7, 7) = out(6, 6);
  out(7, 9) = 2 * htot * states(7) / rholL2h3;

  out(8, 8) = 3 / (rholL2 * htot * term1);
  out(8, 9) =
      states(8) * (12 * l0 / (L0 * rho0 * pow(htot, 4) * term1 * term1) -
                   3 / (rholL2 * htot * htot * term1));

  out(9, 6) = out(6, 9);
  out(9, 7) = out(7, 9);
  out(9, 8) = out(8, 9);
  out(9, 9) =
      (states(6) * states(6) + states(7) * states(7)) /
          (rholL2 * h0 * h0 * h0) +
      states(8) * states(8) *
          (96 * pow(l0, 3) / (rho0 * L0 * pow(htot, 7) * pow(term1, 3)) -
           30 * l0 / (L0 * rho0 * pow(htot, 5) * term1 * term1) +
           3 / (rholL2 * pow(htot, 3) * term1));

  // Vocal tract
  out(10, 10) = a0;
  out(11, 11) = w0 * w0 / a0;
  return out;
};

// Dissipation related function
Eigen::VectorXd VocalApparatusGSI1::zDiss(const VectorXd& wDiss,
                                          const VectorXd& states,
                                          VectorXd& out) {
  // Vocal folds
  out(0) = wDiss(0) * r;
  out(1) = wDiss(1) * r;

  // Glottal flow
  double htot = states(9) + hr;
  out(2) = (wDiss(2) > 0) ? 0.5 * rho0 * pow((wDiss(2) / (L0 * htot)), 2) : 0;
  out(3) = wDiss(4) * h0 / htot;
  out(4) = -wDiss(3) * h0 / htot;

  // Vocal tract
  out(5) = wDiss(5) * q0 * w0 / a0;
  return out;
};
Eigen::MatrixXd VocalApparatusGSI1::gradZDissStates(const VectorXd& wDiss,
                                                    const VectorXd& states,
                                                    MatrixXd& out) {
  // Vocal folds : zeros

  // Glottal flow
  double htot = hr + states(9);
  out(2, 9) = (wDiss(2) > 0)
                  ? -rho0 * (wDiss(2) * wDiss(2) / (L0 * L0 * pow(htot, 3)))
                  : 0;
  out(3, 9) = h0 * wDiss(4) / (htot * htot);
  out(4, 9) = -h0 * wDiss(3) / (htot * htot);

  // Vocal tract : zeros
  return out;
};
Eigen::MatrixXd VocalApparatusGSI1::gradZDissWDiss(const VectorXd& wDiss,
                                                   const VectorXd& states,
                                                   MatrixXd& out) {
  // Vocal folds
  out(0, 0) = r;
  out(1, 1) = r;

  // Glottal flow
  double htot = hr + states(9);
  out(2, 2) = (wDiss(2) > 0) ? wDiss(2) * rho0 * pow(1 / (L0 * htot), 2) : 0;
  out(3, 4) = h0 / htot;
  out(4, 3) = -h0 / htot;

  // Vocal tract
  out(5, 5) = q0 * w0 / a0;
  return out;
};

std::map<std::string, double> VocalApparatusGSI1::getParams() {
  std::map<std::string, double> Params = {
      {"rho0", rho0},     {"L0", L0}, {"l0", l0}, {"h0", h0},
      {"hr", hr},         {"k0", k0}, {"k1", k1}, {"kappa0", kappa0},
      {"kappa1", kappa1}, {"m", m},   {"r", r},   {"S_sub", S_sub},
      {"S_sup", S_sup},   {"a0", a0}, {"q0", q0}, {"w0", w0}};
  return Params;
}

// Vocal apparatus 2 : velocity as in the article
VocalApparatusGSI2::VocalApparatusGSI2(std::map<std::string, double> Params) {
  // Parameters
  // Glottal flow
  this->rho0 = Params["rho0"];  // kg.m-3
  this->L0 = Params["L0"];      // m
  this->l0 = Params["l0"];      // m
  this->h0 = Params["h0"];      // m
  this->hr = Params["hr"];      // m

  this->mu0 = 2 * rho0 * l0 * L0;     // kg.m 2*rho0*l0*L0
  this->rholL2h3 = mu0 * pow(h0, 3);  // kg.m^4 2*rho0*L0*l0*h0**3

  // Vocal folds
  this->k0 = Params["k0"];          // N.m-1
  this->k1 = Params["k1"];          // N.m-1
  this->kappa0 = Params["kappa0"];  // N.m-1
  this->kappa1 = Params["kappa1"];  // N.m-1

  this->m = Params["m"];          // kg
  this->r = Params["r"];          // kg.s-1
  this->S_sub = Params["S_sub"];  // m^2
  this->S_sup = Params["S_sup"];  // m^2

  // Vocal tract impedance
  this->a0 = Params["a0"];  //
  this->q0 = Params["q0"];  // ohm
  this->w0 = Params["w0"];  // rad.s-1

  S = Eigen::MatrixXd::Zero(full_size, full_size);

  S(0, 1) = -1;
  S(0, 2) = 1;
  S(0, 10) = -S_sup;
  S(0, 18) = -S_sub;
  S(2, 14) = -1;
  S(2, 16) = -0.5;

  S(3, 4) = -1;
  S(3, 5) = 1;
  S(3, 10) = -S_sup;
  S(3, 18) = -S_sub;

  S(5, 14) = 1;
  S(5, 16) = -0.5;

  S(6, 10) = -L0 / mu0;
  S(6, 12) = -L0 / mu0;
  S(6, 18) = L0 / mu0;

  S(7, 13) = 1;

  S(8, 15) = 1;
  S(9, 16) = 1;

  S(10, 11) = -1;
  S(10, 16) = -L0 * l0;
  S(10, 17) = -1;

  S(12, 16) = -L0 * l0;

  S(16, 18) = L0 * l0;

  S = (S - S.transpose()).eval();
  S(0, 0) = -r;
  S(3, 3) = -r;
};

double VocalApparatusGSI2::H(const VectorXd& states, double& out) {
  // First vocal fold
  out = 0.5 * (states(0) * states(0) / m + states(1) * states(1) * k0 +
               states(2) * states(2) * kappa0);
  // Second vocal fold
  out += 0.5 * (states(3) * states(3) / m + states(4) * states(4) * k1 +
                states(5) * states(5) * kappa1);
  // Glottal flow
  double mGlot = mu0 * (states(9) + hr);
  double m33 =
      mGlot / 12 * (1 + 4 * l0 * l0 / ((hr + states(9)) * (hr + states(9))));
  out += mGlot / 2 * (states(6) * states(6) + states(7) * states(7)) +
         m33 / 2 * states(8) * states(8);
  // Vocal tract
  out += 0.5 * (states(10) * states(10) * a0 +
                states(11) * states(11) * w0 * w0 / a0);
  return out;
}

Eigen::VectorXd VocalApparatusGSI2::gradH(const VectorXd& states,
                                          VectorXd& out) {
  // First vocal fold
  out(0) = states(0) / m;
  out(1) = states(1) * k0;
  out(2) = states(2) * kappa0;
  // Second vocal fold
  out(3) = states(3) / m;
  out(4) = states(4) * k1;
  out(5) = states(5) * kappa1;

  // Glottal flow
  // double pix = states(6);
  // double piy = states(7);
  // double piexp = states(8);
  // double h = states(9);
  double htot = hr + states(9);
  double mGlot = mu0 * htot;
  double m33 = mGlot / 12 * (1 + 4 * l0 * l0 / (htot * htot));

  out(Eigen::seq(6, 7)) = states(Eigen::seq(6, 7)) * mGlot;
  out(8) = m33 * states(8);
  out(9) = mu0 / (2) *
           (states(6) * states(6) + states(7) * states(7) +
            1 / 12 * states(8) * states(8) * (1 - 4 * l0 * l0 / (htot * htot)));

  // Vocal tract
  out(10) = states(10) * a0;
  out(11) = states(11) * w0 * w0 / a0;
  return out;
};
Eigen::MatrixXd VocalApparatusGSI2::hessH(const VectorXd& states,
                                          MatrixXd& out) {
  // As both vocal folds and the vocal tract are linear in this model,
  // we could initialize the matrix accordingly...
  // First vocal fold
  out(0, 0) = 1 / m;
  out(1, 1) = k0;
  out(2, 2) = kappa0;
  // Second vocal fold
  out(3, 3) = 1 / m;
  out(4, 4) = k1;
  out(5, 5) = kappa1;

  // Glottal flow
  // double pix = states(6);
  // double piy = states(7);
  // double piexp = states(8);
  // double h = states(9);

  double htot = states(9) + hr;
  double mGlot = mu0 * (htot);
  double m33 = mGlot / 12 * (1 + 4 * l0 * l0 / ((htot) * (htot)));

  out(6, 6) = mu0 * htot;
  out(6, 9) = mu0 * states(6);

  out(7, 7) = out(6, 6);
  out(7, 9) = mu0 * states(7);

  out(8, 8) = mu0 * htot * (4 * l0 * l0 / (htot * htot) + 1) / 12;
  out(8, 9) = mu0 / 12 * states(8) * (1 - 4 * l0 * l0 / (htot * htot));

  out(9, 6) = out(6, 9);
  out(9, 7) = out(7, 9);
  out(9, 8) = out(8, 9);
  out(9, 9) = mu0 * l0 * l0 * states(8) * states(8) / (3 * htot * htot * htot);

  // Vocal tract
  out(10, 10) = a0;
  out(11, 11) = w0 * w0 / a0;
  return out;
};

// Dissipation related function
Eigen::VectorXd VocalApparatusGSI2::zDiss(const VectorXd& wDiss,
                                          const VectorXd& states,
                                          VectorXd& out) {
  // Glottal flow
  out(0) = (wDiss(2) > 0)
               ? 0.5 * rho0 * pow((wDiss(2) / (L0 * (hr + states(9)))), 2)
               : 0;

  double mGlot = mu0 * (states(9) + hr);
  double m33 =
      mGlot / 12 * (1 + 4 * l0 * l0 / ((states(9) + hr) * (hr + states(9))));
  out(1) = wDiss(2) * 1 / mGlot;
  out(2) = -wDiss(1) * 1 / mGlot;
  out(3) = wDiss(4) * 1 / m33;
  out(4) = -wDiss(3) * 1 / m33;

  // Vocal tract
  out(5) = wDiss(5) * q0 * w0 / a0;
  return out;
};
Eigen::MatrixXd VocalApparatusGSI2::gradZDissStates(const VectorXd& wDiss,
                                                    const VectorXd& states,
                                                    MatrixXd& out) {
  // Vocal folds : zeros

  // Glottal flow
  double htot = hr + states(9);
  out(0, 9) = (wDiss(2) > 0)
                  ? -rho0 * (wDiss(2) * wDiss(2) / (L0 * L0 * pow(htot, 3)))
                  : 0;
  double mGlot = mu0 * htot;
  double m33 = mGlot / 12 * (1 + 4 * l0 * l0 / (htot * htot));

  double dm33 = mu0 * (1 / 12 - l0 * l0 / (3 * htot * htot));

  out(1, 9) = -wDiss(2) * mu0 / (mGlot * mGlot);
  out(2, 9) = wDiss(1) * mu0 / (mGlot * mGlot);

  double dinvm33 = -dm33 / (m33 * m33);
  out(3, 9) = wDiss(4) * dinvm33;
  out(4, 9) = -wDiss(3) * dinvm33;

  // Vocal tract : zeros
  return out;
};
Eigen::MatrixXd VocalApparatusGSI2::gradZDissWDiss(const VectorXd& wDiss,
                                                   const VectorXd& states,
                                                   MatrixXd& out) {
  // Glottal flow
  double htot = hr + states(9);
  out(0, 0) = (wDiss(2) > 0) ? wDiss(2) * rho0 * pow(1 / (L0 * htot), 2) : 0;
  double mGlot = mu0 * htot;
  double m33 = mGlot / 12 * (1 + 4 * l0 * l0 / (htot * htot));

  out(1, 2) = 1 / mGlot;
  out(2, 1) = -1 / mGlot;
  out(3, 4) = 1 / m33;
  out(4, 3) = -1 / m33;

  // Vocal tract
  out(5, 5) = q0 * w0 / a0;
  return out;
};

std::map<std::string, double> VocalApparatusGSI2::getParams() {
  std::map<std::string, double> Params = {
      {"rho0", rho0},     {"L0", L0}, {"l0", l0}, {"h0", h0},
      {"hr", hr},         {"k0", k0}, {"k1", k1}, {"kappa0", kappa0},
      {"kappa1", kappa1}, {"m", m},   {"r", r},   {"S_sub", S_sub},
      {"S_sup", S_sup},   {"a0", a0}, {"q0", q0}, {"w0", w0}};
  return Params;
}