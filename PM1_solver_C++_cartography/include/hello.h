#include <cmath>
#include <eigen-3.4.0/Eigen/Dense>
#ifndef HELLO_H
#define HELLO_H

// #pragma once

void printHello();

inline double test_func(double& x) { return pow(x, 3); }

inline Eigen::MatrixXd test_func2(double& x) {
  Eigen::MatrixXd m(2, 2);
  m(0, 0) = 3 * x;
  m(1, 0) = 2.5 * x * x;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  return m;
}

#endif  // HELLO_H