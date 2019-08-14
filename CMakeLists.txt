cmake_minimum_required(VERSION 3.8)
project(dnnfold CXX)

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

add_executable(predict src/predict.cpp 
    src/fold.cpp src/fold.h src/parameter.cpp src/parameter.h src/fasta.cpp src/fasta.h src/argparse.hpp src/default_params.cpp src/default_params.h )
target_compile_features(predict PUBLIC cxx_std_17)
target_link_libraries(predict ${TORCH_LIBRARIES})

add_executable(train src/train.cpp 
    src/fold.cpp src/fold.h src/parameter.cpp src/parameter.h src/fasta.cpp src/fasta.h src/argparse.hpp src/default_params.cpp src/default_params.h src/bpseq.h src/bpseq.cpp)
target_compile_features(train PUBLIC cxx_std_17)
target_link_libraries(train ${TORCH_LIBRARIES})