#include <math.h>

#include <chrono>
#include <cmath>
#include <eigen-3.4.0/Eigen/Dense>
#include <iostream>
#include <vector>

#include "GaussQuadrature.h"
#include "PHSModel.h"
#include "VocGSI.h"
#include "hello.h"
#include "helloConfig.h"
#include "map"
#include "matplotlibcpp.h"
#include "solver.h"
namespace plt = matplotlibcpp;

using Eigen::MatrixXd;

int main(int argc, char** argv) {
  // Number of allowed threads
  Eigen::setNbThreads(12);
  // Parameters
  std::map<std::string, double> Params = {
      {"rho0", 1.3},    {"L0", 0.011}, {"l0", 2e-3},  {"h0", 1e-3},
      {"hr", 1e-3},     {"k0", 100},   {"k1", 100},   {"kappa0", 100},
      {"kappa1", 100},  {"m", 2e-4},   {"r", 1.5e-2}, {"S_sub", 11e-5},
      {"S_sup", 11e-7}, {"a0", 1e6},   {"q0", 0.4},   {"w0", 2 * M_PI * 640}};

  VocalApparatusGSI2 Voc = VocalApparatusGSI2(Params);
  int sr = 40000;
  double epsilon = 1e-10;
  int maxIters = 10;
  int n_state = Voc.get_n_state();
  int n_diss = Voc.get_n_diss();
  Eigen::MatrixXd gradZw = Eigen::MatrixXd::Zero(n_diss, n_diss);
  Eigen::MatrixXd gradZx = Eigen::MatrixXd::Zero(n_diss, n_state);
  // std::cout << Voc.gradZDissWDiss(Eigen::VectorXd::Ones(n_diss) * 0.01,
  //                                 Eigen::VectorXd::Ones(n_state) * 0.01,
  //                                 gradZw)
  //           << std::endl;
  // std::cout << Voc.gradZDissStates(Eigen::VectorXd::Ones(n_diss) * 0.01,
  //                                  Eigen::VectorXd::Ones(n_state) * 0.01,
  //                                  gradZx)
  //           << std::endl;
  std::cout << Voc.getS() << std::endl << std::endl;

  std::cout << "Constructing solver" << std::endl << std::endl;
  PM1Solver solver = PM1Solver(Voc, sr, epsilon, maxIters);

  Eigen::VectorXd init = Eigen::MatrixXd::Zero(Voc.get_n_state(), 1);

  float duration = 0.01;
  Eigen::MatrixXd input =
      Eigen::MatrixXd::Zero(Voc.get_n_io(), static_cast<int>(duration * sr));

  double P0 = 200;
  double trise = 5e-3;
  double tdelay = 5e-3;
  double timeStep = 1 / static_cast<double>(sr);
  for (int i = 0; i < static_cast<int>(duration * sr);
       i++) {  //< static_cast<int>(duration * sr)
    if (i * timeStep > tdelay) {
      if (i * timeStep > tdelay + trise) {
        input(i) = P0;
      } else {
        input(i) = P0 * (i * timeStep - tdelay) / trise;
      }
    }
  }
  std::vector<double> inputVec(input.data(), input.data() + input.size());
  // plt::plot(inputVec);
  // plt::show();

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
  Eigen::MatrixXd stateP = states.row(1);
  std::vector<double> statePvec(stateP.data(), stateP.data() + stateP.size());
  Eigen::MatrixXd state1 = states.row(9);
  std::vector<double> v1(state1.data(), state1.data() + state1.size());
  // std::cout << linspace(static_cast<float>(0), duration, v1.size()).size()
  //          << std::endl;
  // plt::plot(statePvec);
  // plt::show();
  for (int i = 0; i < 12; i++) {
    plt::subplot2grid(3, 4, (i - i % 4) / 4, i % 4);
    Eigen::MatrixXd statei = states.row(i);
    std::vector<double> stateiVec(statei.data(), statei.data() + statei.size());
    plt::plot(stateiVec);
  }
  plt::tight_layout();
  plt::show();
  return 0;
}
