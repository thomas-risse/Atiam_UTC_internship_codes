#include <chrono>
#include <cmath>
#include <eigen-3.4.0/Eigen/Dense>
#include <iostream>
#include <vector>

#include "GaussQuadrature.h"
#include "PHSModel.h"
#include "hello.h"
#include "helloConfig.h"
#include "matplotlibcpp.h"
#include "solver.h"

namespace plt = matplotlibcpp;

using Eigen::MatrixXd;

int main(int argc, char** argv) {
  // Number of allowed threads
  Eigen::setNbThreads(12);
  // Parameters
  float C0 = 1;
  float R0 = 5;
  float phi0 = 0.1;
  float L0 = 1;
  float E0 = pow(phi0, 2) / L0;

  NonLinearRLC rlc = NonLinearRLC(C0, E0, phi0, R0);
  int sr = 10000;
  double epsilon = 1e-16;
  int maxIters = 10;
  // std::cout << Quadrature4::quadWeights << std::endl;

  std::cout << "Constructing solver" << std::endl << std::endl;
  PM1Solver solver = PM1Solver(rlc, sr, epsilon, maxIters);

  Eigen::VectorXd init(2);
  init << 0, 1;

  float duration = 100;
  Eigen::VectorXd time =
      Eigen::ArrayXd::LinSpaced(static_cast<int>(duration * sr), 0, duration);

  Eigen::MatrixXd input =
      Eigen::MatrixXd::Zero(rlc.get_n_io(), static_cast<int>(duration * sr));
  // for (int elem = 0; elem < time.size(); elem++) {
  //  input(0, elem) = std::sin(2 * M_PI * 0.5 * time(elem));
  //}

  std::vector<double> timeVec(time.data(), time.data() + time.size());
  std::vector<double> inputVec(input.data(), input.data() + input.size());
  plt::plot(timeVec, inputVec);
  plt::show();

  std::cout << "Beginning simulation" << std::endl << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  solver.simulate(init, input);
  auto stop = std::chrono::high_resolution_clock::now();
  auto simu_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "End of simulation, time elapsed : " << simu_time.count() << "ms"
            << std::endl
            << std::endl;

  Eigen::MatrixXd states = solver.getStatesResults();
  Eigen::MatrixXd state0 = states.row(0);
  std::vector<double> v0(state0.data(), state0.data() + state0.size());
  Eigen::MatrixXd state1 = states.row(1);
  std::vector<double> v1(state1.data(), state1.data() + state1.size());
  // std::cout << linspace(static_cast<float>(0), duration, v1.size()).size()
  //          << std::endl;
  plt::plot(v0);
  plt::plot(v1);
  plt::show();

  printHello();
  return 0;
}