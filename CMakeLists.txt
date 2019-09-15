cmake_minimum_required(VERSION 3.8)
project(dnnfold CXX)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(pybind11)
pybind11_add_module(interface
    src/interface.cpp 
    src/parameter.cpp src/parameter.h 
    src/default_params.cpp src/default_params.h
    src/fold.cpp src/fold.h src/trimatrix.h )
target_compile_features(interface PUBLIC cxx_std_17)
#target_link_libraries(dnnfold)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -D_LIBCPP_DEBUG=0")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -D_GLIBCXX_DEBUG")
