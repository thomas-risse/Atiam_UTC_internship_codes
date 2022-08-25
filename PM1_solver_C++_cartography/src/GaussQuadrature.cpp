#include "GaussQuadrature.h"

Eigen::VectorXd Quadrature4::quadPoints = Quadrature4::populateQuadPoints();
Eigen::VectorXd Quadrature4::quadWeights = Quadrature4::populateQuadWeights();

Eigen::VectorXd Quadrature4::populateQuadPoints() {
  Eigen::VectorXd points = Eigen::VectorXd::Zero(4);
  double x1 = (15 + 2 * std::sqrt(30)) / 35;
  double x2 = (15 - 2 * std::sqrt(30)) / 35;
  points(0) = (-std::sqrt(x1) / 2) + 0.5;
  points(1) = (-std::sqrt(x2) / 2) + 0.5;
  points(2) = (std::sqrt(x2) / 2) + 0.5;
  points(3) = (std::sqrt(x1) / 2) + 0.5;
  return points;
}

Eigen::VectorXd Quadrature4::populateQuadWeights() {
  Eigen::VectorXd weights = Eigen::VectorXd::Zero(4);
  double x1 = (18 - std::sqrt(30)) / 36;
  double x2 = (18 + std::sqrt(30)) / 36;
  weights(0) = x1 / 2;
  weights(1) = x2 / 2;
  weights(2) = x2 / 2;
  weights(3) = x1 / 2;
  return weights;
}
