#ifndef PHS_MODEL_h
#define PHS_MODEL_h

#include <eigen-3.4.0/Eigen/Dense>
#include <iostream>
#include <list>
#include <map>
#include <string>

// #pragma once

class PHSModel {
  // General class for PHS models
 private:
  // System sizes
  int n_state = 0;
  int n_cons = 0;
  int n_diss = 0;
  int n_io = 0;
  int full_size = 0;
  int idx1 = n_state;
  int idx2 = n_diss + idx1;
  int idx3 = idx2 + n_cons;

  // System interconnexion matrix
  Eigen::MatrixXd S;

 public:
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;
  // Sub matrices
  inline MatrixXd Sxx() {
    return getS().block(0, 0, get_n_state(), get_n_state());
  };
  inline MatrixXd Sxw() {
    return getS().block(0, get_n_state(), get_n_state(), get_n_diss());
  };
  inline MatrixXd Sxl() {
    return getS().block(0, get_idx2(), get_n_state(), get_n_cons());
  };
  inline MatrixXd Sxu() {
    return getS().block(0, get_idx3(), get_n_state(), get_n_io());
  };

  inline MatrixXd Swx() {
    return getS().block(get_n_state(), 0, get_n_diss(), get_n_state());
  };
  inline MatrixXd Sww() {
    return getS().block(get_n_state(), get_n_state(), get_n_diss(),
                        get_n_diss());
  };
  inline MatrixXd Swl() {
    return getS().block(get_n_state(), get_idx2(), get_n_diss(), get_n_cons());
  };
  inline MatrixXd Swu() {
    return getS().block(get_n_state(), get_idx3(), get_n_diss(), get_n_io());
  };

  inline MatrixXd Slx() {
    return getS().block(get_idx2(), 0, get_n_cons(), get_n_state());
  };
  inline MatrixXd Slw() {
    return getS().block(get_idx2(), get_n_state(), get_n_cons(), get_n_diss());
  };
  inline MatrixXd Sll() {
    return getS().block(get_idx2(), get_idx2(), get_n_cons(), get_n_cons());
  };
  inline MatrixXd Slu() {
    return getS().block(get_idx2(), get_idx3(), get_n_cons(), get_n_io());
  };

  inline MatrixXd Syx() {
    return getS().block(get_idx3(), 0, get_n_io(), get_n_state());
  };
  inline MatrixXd Syw() {
    return getS().block(get_idx3(), get_n_state(), get_n_io(), get_n_diss());
  };
  inline MatrixXd Syl() {
    return getS().block(get_idx3(), get_idx2(), get_n_io(), get_n_cons());
  };
  inline MatrixXd Syu() {
    return getS().block(get_idx3(), get_idx3(), get_n_io(), get_n_io());
  };

  // Hamiltonian related functions
  virtual double H(const VectorXd& states, double& out) = 0;
  double H(const VectorXd& states);
  virtual VectorXd gradH(const VectorXd& states, VectorXd& out) = 0;
  Eigen::VectorXd gradH(const VectorXd& states);
  virtual MatrixXd hessH(const VectorXd& states, MatrixXd& out) = 0;
  MatrixXd hessH(const VectorXd& states);

  // Dissipation related function
  virtual VectorXd zDiss(const VectorXd& wDiss, const VectorXd& states,
                         VectorXd& out) = 0;
  VectorXd zDiss(const VectorXd& wDiss, const VectorXd& states);
  virtual MatrixXd gradZDissStates(const VectorXd& wDiss,
                                   const VectorXd& states, MatrixXd& out) = 0;
  MatrixXd gradZDissStates(const VectorXd& wDiss, const VectorXd& states);
  virtual MatrixXd gradZDissWDiss(const VectorXd& wDiss, const VectorXd& states,
                                  MatrixXd& out) = 0;
  MatrixXd gradZDissWDiss(const VectorXd& wDiss, const VectorXd& states);

  // Getters
  virtual int get_n_state() { return n_state; }
  virtual int get_n_diss() { return n_diss; }
  virtual int get_n_cons() { return n_cons; }
  virtual int get_n_io() { return n_io; }
  virtual int get_full_size() { return full_size; }
  virtual int get_idx1() { return idx1; }
  virtual int get_idx2() { return idx2; }
  virtual int get_idx3() { return idx3; }

  virtual std::map<std::string, double> getParams() = 0;

  virtual MatrixXd getS() { return S; }
};

class LinearRLC : public PHSModel {
 public:
  static const int n_state = 2;
  static const int n_cons = 0;
  static const int n_diss = 1;
  static const int n_io = 1;
  static const int full_size = n_state + n_cons + n_diss + n_io;
  static const int idx1 = n_state;
  static const int idx2 = idx1 + n_diss;
  static const int idx3 = idx2 + n_cons;

  // Getters
  int get_n_state() { return n_state; }
  int get_n_diss() { return n_diss; }
  int get_n_cons() { return n_cons; }
  int get_n_io() { return n_io; }
  int get_full_size() { return full_size; }
  int get_idx1() { return idx1; }
  int get_idx2() { return idx2; }
  int get_idx3() { return idx3; }
  MatrixXd getS() { return S; }

  // Parameters
  double C0;
  double R0;
  double L0;

  // System interconnexion matrix
  Eigen::Matrix<double, 4, 4> S;

  // Constructor
  LinearRLC(const float& C0, const float& L0, const float& R0);

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

class NonLinearRLC : public LinearRLC {
 private:
  float E0;
  float phi0;

 public:
  NonLinearRLC(const float& C0, const float& E0, const float& phi0,
               const float& R0);
  // Hamiltonian related functions
  virtual double H(const VectorXd& states, double& out);
  virtual VectorXd gradH(const VectorXd& states, VectorXd& out);
  virtual MatrixXd hessH(const VectorXd& states, MatrixXd& out);
};

#endif  // PHS_MODEL_h