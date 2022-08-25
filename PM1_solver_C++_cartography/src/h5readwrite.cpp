#include "h5readwrite.h"

SimuParams::SimuParams(std::string Filename) {
  modelFixedParameters = {
      {"rho0", 1.3},      {"L0", 0.011}, {"l0", 2e-3},     {"h0", 1e-4},
      {"hr", 1e-4},       {"k0", 100},   {"k1", 97},       {"kappa0", 3 * 100},
      {"kappa1", 3 * 97}, {"m", 2e-4},   {"S_sub", 11e-5}, {"S_sup", 11e-7},
      {"q0", 0.4},        {"w0", 640},   {"r", {}}};  //,        {"a0", {}}};
  inputParameters = {{"trise", 0}, {"tdelay", 0}, {"tmax", 0}};
  modelVariableParameters = {{"P0", {}}, {"hr", {}}, {"a0", {}}};
  solverParameters = {{"fs", 0}, {"Te", 0}, {"epsilon", 0}};

  SimuParams::readFromFile(Filename);
}

void SimuParams::readFromFile(std::string Filename) {
  std::string script = "USE FILE " + Filename;
  HDFql::execute(script.c_str());

  // show (i.e. get) HDF5 file currently in use and populate HDFql default
  // cursor with it
  HDFql::execute("SHOW USE FILE");

  // display HDF5 file currently in use
  HDFql::cursorFirst();
  std::cout << "Fichier utilisé : " << HDFql::cursorGetChar() << std::endl;

  // loop along fixed parameters
  std::cout << std::endl << "Récupération des paramètres fixés" << std::endl;
  for (const auto &pair : modelFixedParameters) {
    std::string key = pair.first;
    std::cout << "Clé : " << key << "    ";
    script = "SELECT FROM model/fixed_parameters/" + key;
    HDFql::execute(script.c_str());
    HDFql::cursorFirst();
    modelFixedParameters[key] = *HDFql::cursorGetDouble();
    std::cout << "Valeur : " << modelFixedParameters[key] << std::endl;
  }
  std::cout << std::endl << std::endl;

  // loop along input parameters
  std::cout << std::endl << "Récupération des paramètres d'entrée" << std::endl;
  for (const auto &pair : inputParameters) {
    std::string key = pair.first;
    std::cout << "Clé : " << key << "    ";
    script = "SELECT FROM model/fixed_parameters/" + key;
    HDFql::execute(script.c_str());
    HDFql::cursorFirst();
    inputParameters[key] = *HDFql::cursorGetDouble();
    HDFql::cursorFirst();
    std::cout << "Valeur : " << inputParameters[key] << std::endl;
  }
  std::cout << std::endl << std::endl;

  // loop along solver parameters
  std::cout << std::endl
            << "Récupération des paramètres de solveur" << std::endl;
  for (const auto &pair : solverParameters) {
    std::string key = pair.first;
    std::cout << "Clé : " << key << "    ";
    script = "SELECT FROM solver_parameters/" + key;
    HDFql::execute(script.c_str());
    HDFql::cursorFirst();
    solverParameters[key] = *HDFql::cursorGetDouble();
    HDFql::cursorFirst();
    std::cout << "Valeur : " << solverParameters[key] << std::endl;
  }
  std::cout << std::endl << std::endl;

  // loop along variable parameters
  std::cout << std::endl
            << "Récupération des paramètres variables" << std::endl;
  for (const auto &pair : modelVariableParameters) {
    std::string key = pair.first;
    std::cout << "Clé : " << key << std::endl;
    std::cout << "Valeurs : " << std::endl;
    script = "SELECT FROM model/variable_parameters/" + key;
    HDFql::execute(script.c_str());

    HDFql::cursorFirst();
    modelVariableParameters[key].push_back(*HDFql::cursorGetDouble());
    while (HDFql::cursorNext() == HDFql::Success) {
      modelVariableParameters[key].push_back(*HDFql::cursorGetDouble());
      std::cout << *HDFql::cursorGetDouble() << "  ";
    }
    modelFixedParameters[key] = modelVariableParameters[key][0];
    HDFql::cursorFirst();
    std::cout << std::endl << std::endl;
    NPoints = modelVariableParameters[key].size();
  }
  std::cout << std::endl << std::endl;

  script = "CLOSE FILE " + Filename;
  HDFql::execute(script.c_str());
}

void writeResults(PM1Solver &solver, int success, std::string groupName,
                  std::string Filename) {
  // Opening file
  std::string script = "USE FILE " + Filename;
  HDFql::execute(script.c_str());

  // Creation of a group containing simulation results
  script = "CREATE GROUP " + groupName;
  HDFql::execute(script.c_str());

  // Getting results in vector
  std::vector<double> effortsResults;
  effortsResults.resize(solver.getEffortsResults().size());
  Eigen::VectorXd::Map(&effortsResults[0], solver.getEffortsResults().size()) =
      solver.getEffortsResults();

  std::vector<double> flowsResults;
  flowsResults.resize(solver.getFlowsResults().size());
  Eigen::VectorXd::Map(&flowsResults[0], solver.getFlowsResults().size()) =
      solver.getFlowsResults();

  std::vector<double> statesResults;
  statesResults.resize(solver.getStatesResults().size());
  Eigen::VectorXd::Map(&statesResults[0], solver.getStatesResults().size()) =
      solver.getStatesResults();

  std::vector<double> time;
  time.resize(solver.getTimeVect().size());
  Eigen::VectorXd::Map(&time[0], solver.getTimeVect().size()) =
      solver.getTimeVect();

  // Creating datasets
  // create dataset 'efforts'
  script = "CREATE DATASET " + groupName + "/efforts" + " as double(" +
           std::to_string(effortsResults.size()) + ")";
  HDFql::execute(script.c_str());

  script = "INSERT INTO " + groupName + "/efforts" + " VALUES FROM MEMORY " +
           std::to_string(HDFql::variableTransientRegister(effortsResults));
  HDFql::execute(script.c_str());

  // create dataset 'flows'
  script = "CREATE DATASET " + groupName + "/flows" + " as double(" +
           std::to_string(flowsResults.size()) + ")";
  HDFql::execute(script.c_str());

  script = "INSERT INTO " + groupName + "/flows" + " VALUES FROM MEMORY " +
           std::to_string(HDFql::variableTransientRegister(flowsResults));
  HDFql::execute(script.c_str());

  // create dataset 'states'
  script = "CREATE DATASET " + groupName + "/states" + " as double(" +
           std::to_string(statesResults.size()) + ")";
  HDFql::execute(script.c_str());

  script = "INSERT INTO " + groupName + "/states" + " VALUES FROM MEMORY " +
           std::to_string(HDFql::variableTransientRegister(statesResults));
  HDFql::execute(script.c_str());

  // create dataset 'time'
  script = "CREATE DATASET " + groupName + "/time" + " as double(" +
           std::to_string(time.size()) + ")";
  HDFql::execute(script.c_str());

  script = "INSERT INTO " + groupName + "/time" + " VALUES FROM MEMORY " +
           std::to_string(HDFql::variableTransientRegister(time));
  HDFql::execute(script.c_str());

  // create attribute success
  script = "CREATE ATTRIBUTE " + groupName + "/success" + " as int";
  HDFql::execute(script.c_str());

  script = "INSERT INTO " + groupName + "/success" + " VALUES FROM MEMORY " +
           std::to_string(HDFql::variableTransientRegister(&success));
  HDFql::execute(script.c_str());
}