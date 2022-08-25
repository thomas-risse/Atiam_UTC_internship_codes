cmake_minimum_required(VERSION 3.14.0)
set(CMAKE_CXX_STANDARD 11)

# Set the project name and version
project(Cartography VERSION 1.0)

set(HELLO_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(HELLO_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

# specify the C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(HELLO_FILES

    # ${HELLO_SOURCE_DIR}/hello.cpp
    ${HELLO_SOURCE_DIR}/Cartography.cpp

    # ${HELLO_SOURCE_DIR}/RLC.cpp
    # ${HELLO_SOURCE_DIR}/VocalApparatus.cpp
    ${HELLO_SOURCE_DIR}/GaussQuadrature.cpp
    ${HELLO_SOURCE_DIR}/PHSModel.cpp
    ${HELLO_SOURCE_DIR}/solver.cpp
    ${HELLO_SOURCE_DIR}/VocGSI.cpp
    ${HELLO_SOURCE_DIR}/h5readwrite.cpp

    #
    ${HELLO_INCLUDE_DIR}/hello.h
    ${HELLO_INCLUDE_DIR}/PHSModel.h
    ${HELLO_INCLUDE_DIR}/GaussQuadrature.h
    ${HELLO_INCLUDE_DIR}/solver.h
    ${HELLO_INCLUDE_DIR}/matplotlibcpp.h
    ${HELLO_INCLUDE_DIR}/VocGSI.h
    ${HELLO_INCLUDE_DIR}/h5readwrite.h

    # ${HELLO_INCLUDE_DIR}/hdfql/HDFql.hpp
)

# Profiling
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
add_executable(Cartography ${HELLO_FILES})

# find_library(HDFQL libHDFql.so ${HELLO_INCLUDE_DIR}/hdfql)
# message(STATUS ${HDFQL})

# target_link_libraries(Cartography ${HDFQL})

# Compile options
# SET(CMAKE_BUILD_TYPE "Release")
message(STATUS "CMAKE build type : ${CMAKE_BUILD_TYPE}")
add_compile_options(-Wall -Werror
    -fopenmp
    $<$<CONFIG::RELEASE> - O3>
    $<$<CONFIG::DEBUG> - OOt>
    $<$<CONFIG::DEBUG> - ggdb3>)

# Header file to pass version number
configure_file(${HELLO_INCLUDE_DIR}/helloConfig.h.in
    ${HELLO_INCLUDE_DIR}/helloConfig.h)

target_include_directories(Cartography
    PUBLIC
    ${HELLO_INCLUDE_DIR}

    # ${EIGEN3_INCLUDE_DIR}
    ${HELLO_INCLUDE_DIR}/hdfql
)
message(STATUS "truc : $ENV{EIGEN3_INCLUDE_DIR}")

# Matplotlib cpp
# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

# find python libraries
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(PythonLibs 3.0 REQUIRED)
include_directories(${PYTHON3_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})

# populate matplotlib repository
include(FetchContent)
FetchContent_Declare(
    matplotlib
    GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
    GIT_TAG f23347fca25219d1c42cbb91608b5556814bf572
)
FetchContent_GetProperties(matplotlib)

if(NOT matplotlib_POPULATED)
    FetchContent_Populate(matplotlib)
endif()

include_directories(SYSTEM ${matplotlib_SOURCE_DIR})

# link python and numpy
target_link_libraries(Cartography
    PRIVATE
    ${PYTHON_LIBRARIES}
    Python3::NumPy

    "${HELLO_INCLUDE_DIR}/hdfql/libHDFql.so"
)