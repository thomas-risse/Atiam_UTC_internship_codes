#ifndef GAUSS_QUAD
#define GAUSS_QUAD

#include <cmath>
#include <eigen-3.4.0/Eigen/Dense>
#include <functional>
// #pragma once

class Quadrature4 {
  // Gauss legendre quadrature of order 4
  // on interval [0, 1]
 public:
  // Constructeur '= delete' : aucun objet ne peut être instancié.
  // car la classe est statique.
  Quadrature4() = delete;
  // System sizes
  static const int order = 4;

  static Eigen::VectorXd quadWeights;
  static Eigen::VectorXd quadPoints;

  // Polulating weights and points
  static Eigen::VectorXd populateQuadWeights();
  static Eigen::VectorXd populateQuadPoints();

  // public:
  // Constructor

  // Template function to integrate
  // fuctions which returns floats, vectors or matrices
  template <typename T>
  static T integrate(std::function<T(double&)> integrand) {
    T integral = quadWeights[0] * integrand(quadPoints[0]);
    for (int i = 1; i < order; i++) {
      integral += quadWeights[i] * integrand(quadPoints[i]);
    }
    return integral;
  };
};

#endif  // GAUSS_QUAD