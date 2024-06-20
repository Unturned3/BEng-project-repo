
cmake_minimum_required(VERSION 3.21...3.28)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(Ceres-Test VERSION 0.1
                   DESCRIPTION "Bundle adjustment with Ceres-Solver"
                   LANGUAGES C CXX)

# Include custom modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(utils)

prevent_in_src_build()

# See: https://www.scivision.dev/cmake-ignore-anaconda-libs/
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_IGNORE_PREFIX_PATH $ENV{CONDA_PREFIX})
    list(APPEND CMAKE_IGNORE_PATH $ENV{CONDA_PREFIX}/bin)
endif()

add_subdirectory(thirdparty/cnpy)  # for reading/writing numpy arrays
add_subdirectory(thirdparty/fmt)
add_subdirectory(thirdparty/argparse)


find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)     # required by cnpy


### estimate executable ###

add_executable(estimate estimate-vanilla.cpp)

set_target_properties(estimate PROPERTIES CXX_EXTENSIONS OFF)

target_compile_features(estimate PRIVATE cxx_std_20)

target_compile_options(estimate PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion
    -Wno-unused-variable -Wno-unused-parameter
    -Wno-sign-conversion
    -O3 #-g3 -Werror
)

target_link_libraries(estimate PRIVATE
    ${HDF5_LIBRARIES}
    fmt::fmt
    cnpy
    ZLIB::ZLIB
    Eigen3::Eigen
    Ceres::ceres
    ${OpenCV_LIBS}
    argparse
)


### Interp Executable ###

add_executable(interp interp.cpp)

set_target_properties(interp PROPERTIES CXX_EXTENSIONS OFF)

target_compile_features(interp PRIVATE cxx_std_20)

target_compile_options(interp PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion
    -Wno-unused-variable -Wno-unused-parameter
    -Wno-sign-conversion
    -O3 #-g3 -Werror
)

target_link_libraries(interp PRIVATE
    ${HDF5_LIBRARIES}
    fmt::fmt
    cnpy
    ZLIB::ZLIB
    Eigen3::Eigen
    Ceres::ceres
    ${OpenCV_LIBS}
)