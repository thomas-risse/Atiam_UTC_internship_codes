#ifndef VOCGSI_h
#define VOCGSI_h

#include "PHSModel.h"

class VocalApparatusGSI1 : public PHSModel {
 private:
  // System sizes
  int n_state = 12;
  int n_cons = 0;
  int n_diss = 6;
  int n_io = 1;
  int full_size = 19;
  int idx1 = 12;
  int idx2 = 18;
  int idx3 = 18;

  // Parameters
  // Glottal flow
  double rho0 = 1.3;  // kg.m-3
  double L0;          // m
  double l0;          // m
  double h0;          // m
  double hr;          // m

  double rholL2;    // kg.m-1 2*rho0*l0*L0
  double rholL2h3;  // kg.m^2 2*rho0*L0*l0*h0**3

  // Vocal folds
  double k0;      // N.m-1
  double k1;      // N.m-1
  double kappa0;  // N.m-1
  double kappa1;  // N.m-1

  double m;      // kg
  double r;      // kg.s-1
  double S_sub;  // m^2
  double S_sup;  // m^2

  // Vocal tract impedance
  double a0;  //
  double q0;  // ohm
  double w0;  // rad.s-1

  // System interconnexion matrix
  Eigen::MatrixXd S;

 public:
  // Getters
  virtual int get_n_state() { return n_state; }
  virtual int get_n_diss() { return n_diss; }
  virtual int get_n_cons() { return n_cons; }
  virtual int get_n_io() { return n_io; }
  virtual int get_full_size() { return full_size; }
  virtual int get_idx1() { return idx1; }
  virtual int get_idx2() { return idx2; }
  virtual int get_idx3() { return idx3; }
  virtual std::map<std::string, double> getParams();

  virtual MatrixXd getS() { return S; }

  // Setters
  void set_a0(double& a0) { this->a0 = a0; }
  void set_r(double& r) { this->r = r; }
  void set_hr(double& hr) { this->hr = hr; }

  // Constructor
  VocalApparatusGSI1(std::map<std::string, double> Params);

  // Hamiltonian related functions
  virtual double H(const VectorXd& states, double& out);
  virtual VectorXd gradH(const VectorXd& states, VectorXd& out);
  virtual MatrixXd hessH(const VectorXd& states, MatrixXd& out);

  // Dissipation related function
  virtual VectorXd zDiss(const VectorXd& wDiss, const VectorXd& state,
                         VectorXd& out);
  virtual MatrixXd gradZDissStates(const VectorXd& wDiss,
                                   const VectorXd& states, MatrixXd& out);
  virtual MatrixXd gradZDissWDiss(const VectorXd& wDiss, const VectorXd& state,
                                  MatrixXd& out);
};

class VocalApparatusGSI2 : public PHSModel {
 private:
  // System sizes
  int n_state = 12;
  int n_cons = 0;
  int n_diss = 6;
  int n_io = 1;
  int full_size = 19;
  int idx1 = 12;
  int idx2 = 18;
  int idx3 = 18;

  // Parameters
  // Glottal flow
  double rho0 = 1.3;  // kg.m-3
  double L0;          // m
  double l0;          // m
  double h0;          // m
  double hr;          // m

  double mu0;       // kg.m 2*rho0*l0*L0
  double rholL2h3;  // kg.m^4 2*rho0*L0*l0*h0**3

  // Vocal folds
  double k0;      // N.m-1
  double k1;      // N.m-1
  double kappa0;  // N.m-1
  double kappa1;  // N.m-1

  double m;      // kg
  double r;      // kg.s-1
  double S_sub;  // m^2
  double S_sup;  // m^2

  // Vocal tract impedance
  double a0;  //
  double q0;  // ohm
  double w0;  // rad.s-1

  // System interconnexion matrix
  Eigen::MatrixXd S;

 public:
  // Getters
  virtual int get_n_state() { return n_state; }
  virtual int get_n_diss() { return n_diss; }
  virtual int get_n_cons() { return n_cons; }
  virtual int get_n_io() { return n_io; }
  virtual int get_full_size() { return full_size; }
  virtual int get_idx1() { return idx1; }
  virtual int get_idx2() { return idx2; }
  virtual int get_idx3() { return idx3; }
  virtual std::map<std::string, double> getParams();

  virtual MatrixXd getS() { return S; }

  // Setters
  void set_a0(double& a0) { this->a0 = a0; }
  void set_r(double& r) {
    this->r = r;
    S(0, 0) = -r;
    S(3, 3) = -r;
  }
  void set_hr(double& hr) { this->hr = hr; }

  // Constructor
  VocalApparatusGSI2(std::map<std::string, double> Params);

  // Hamiltonian related functions
  virtual double H(const VectorXd& states, double& out);
  virtual VectorXd gradH(const VectorXd& states, VectorXd& out);
  virtual MatrixXd hessH(const VectorXd& states, MatrixXd& out);

  // Dissipation related function
  virtual VectorXd zDiss(const VectorXd& wDiss, const VectorXd& state,
                         VectorXd& out);
  virtual MatrixXd gradZDissStates(const VectorXd& wDiss,
                                   const VectorXd& states, MatrixXd& out);
  virtual MatrixXd gradZDissWDiss(const VectorXd& wDiss, const VectorXd& state,
                                  MatrixXd& out);
};

#endif