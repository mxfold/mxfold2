cmake_minimum_required(VERSION 3.8)
project(dnnfold CXX)

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -D_LIBCPP_DEBUG=0")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -D_GLIBCXX_DEBUG")

add_executable(predict src/predict.cpp 
    src/fold.cpp src/fold.h src/parameter.cpp src/parameter.h src/fasta.cpp src/fasta.h src/argparse.hpp src/default_params.cpp src/default_params.h src/trimatrix.h)
target_compile_features(predict PUBLIC cxx_std_17)
target_link_libraries(predict ${TORCH_LIBRARIES})

add_executable(train src/train.cpp 
    src/fold.cpp src/fold.h src/parameter.cpp src/parameter.h src/fasta.cpp src/fasta.h src/argparse.hpp src/default_params.cpp src/default_params.h src/trimatrix.h src/bpseq.h src/bpseq.cpp)
target_compile_features(train PUBLIC cxx_std_17)
target_link_libraries(train ${TORCH_LIBRARIES})