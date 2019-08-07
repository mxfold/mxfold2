cmake_minimum_required(VERSION 3.8)
project(dnnfold CXX)
add_executable(a.out src/fold.cpp src/fold.h src/parameter.cpp src/parameter.h src/fasta.cpp src/fasta.h src/argparse.hpp src/default_params.cpp src/default_params.h)
target_compile_features(a.out PUBLIC cxx_std_17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(VIENNARNA REQUIRED RNAlib2)

find_package(Torch REQUIRED)

include_directories(${VIENNARNA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(a.out ${VIENNARNA_LDFLAGS} ${TORCH_LIBRARIES})