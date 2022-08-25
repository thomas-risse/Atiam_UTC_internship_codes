#ifndef READ_PARAMS
#define READ_PARAMS

#include <iostream>
#include <list>
#include <map>
#include <string>

#include "HDFql.hpp"
#include "solver.h"

// #pragma once
class SimuParams {
  // Storing of simulation parameters for realization
  // of cartographies
 private:
  std::map<std::string, double> modelFixedParameters;
  std::map<std::string, std::vector<double>> modelVariableParameters;
  std::map<std::string, double> inputParameters;

  std::map<std::string, double> solverParameters;

  int NPoints;

 public:
  SimuParams(std::string Filename);
  void readFromFile(std::string Filename);

  std::map<std::string, double> getFixedParams() {
    return modelFixedParameters;
  };
  std::map<std::string, std::vector<double>> getVariableParams() {
    return modelVariableParameters;
  };
  std::map<std::string, double> getInputParams() { return inputParameters; };
  std::map<std::string, double> getSolverParams() { return solverParameters; };

  int getNPoints() { return NPoints; };
};

void writeResults(PM1Solver &solver, int success, std::string groupName,
                  std::string Filename);

#endif  // READ_PARAMS