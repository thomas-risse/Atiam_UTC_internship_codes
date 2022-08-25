#include <math.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <eigen-3.4.0/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

#include "GaussQuadrature.h"
#include "HDFql.hpp"
#include "PHSModel.h"
#include "VocGSI.h"
#include "h5readwrite.h"
#include "hello.h"
#include "helloConfig.h"
#include "map"
#include "matplotlibcpp.h"
#include "solver.h"
namespace plt = matplotlibcpp;

using Eigen::MatrixXd;

int main(int argc, char** argv) {
  std::string Filename;
  if (argc > 1) {
    Filename = argv[1];
  } else {
    Filename = "../Cartography/simulations_parameters/test.hdf5";
  }

  std::cout << "Reading parameters file" << Filename << std::endl << std::endl;
  SimuParams params = SimuParams(Filename);

  std::cout << "Initilialization of the model" << std::endl << std::endl;
  VocalApparatusGSI1 Voc = VocalApparatusGSI1(params.getFixedParams());

  std::cout << "Creating output file" << std::endl << std::endl;
  std::ifstream src(Filename, std::ios::binary);

  std::string outputFilename;
  if (argc > 1) {
    outputFilename = argv[2];
  } else {
    outputFilename = "simulationResults.hdf5";
  }
  std::ofstream dst(outputFilename, std::ios::binary);

  // Copying parameter file so that results file contains simulations
  // parameters
  dst << src.rdbuf();
  dst.close();

  // Variable parameters
  double P0;
  double hr;
  double a0;

  // Solver parameters
  double fs = params.getSolverParams()["fs"];
  double Te = params.getSolverParams()["Te"];
  double epsilon = params.getSolverParams()["epsilon"];

  PM1Solver solver = PM1Solver(Voc, fs, epsilon, 10);

  // Input matrix
  double duration = params.getInputParams()["tmax"];
  std::cout << "Simulations duration = " << duration << "s" << std::endl
            << std::endl;
  Eigen::MatrixXd input =
      Eigen::MatrixXd::Zero(Voc.get_n_io(), static_cast<int>(duration * fs));

  Eigen::VectorXd init = Eigen::MatrixXd::Zero(Voc.get_n_state(), 1);

  double trise = params.getInputParams()["trise"];
  double tdelay = params.getInputParams()["tdelay"];

  std::string groupName = "simulation";

  int success = 0;

  for (int i = 0; i < params.getNPoints(); i++) {
    std::cout << "Simulation " << i << std::endl;
    hr = params.getVariableParams()["hr"][i];
    P0 = params.getVariableParams()["P0"][i];
    a0 = params.getVariableParams()["a0"][i];

    // Setting variable params
    Voc.set_hr(hr);
    Voc.set_a0(a0);

    // std::cout << "a0 = " << (*solver.getStructure()).getParams()["a0"]
    //          << std::endl;
    // std::cout << "P0 = " << (*solver.getStructure()).getParams()["P0"]
    //          << std::endl;
    // std::cout << "r = " << (*solver.getStructure()).getParams()["r"]
    //          << std::endl
    //          << std::endl;

    // Setting input
    input =
        Eigen::MatrixXd::Zero(Voc.get_n_io(), static_cast<int>(duration * fs));
    for (int i = 0; i < static_cast<int>(duration * fs); i++) {
      if (i * Te > tdelay) {
        if (i * Te > tdelay + trise) {
          input(i) = P0;
        } else {
          input(i) = P0 * (i * Te - tdelay) / trise;
        }
      }
    }

    success = solver.simulate(init, input);
    // Write results to file
    groupName = "simulation" + std::to_string(i);
    writeResults(solver, success, groupName, outputFilename);
  }
  return 0;
}
