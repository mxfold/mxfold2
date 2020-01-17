cmake_minimum_required(VERSION 3.10)
project(dnnfold CXX)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(pybind11)
pybind11_add_module(interface
    src/interface.cpp 
    src/param/turner.cpp src/param/turner.h 
    src/param/positional.cpp src/param/positional.h 
    src/param/bpscore.cpp src/param/bpscore.h     
    src/param/mix.cpp src/param/mix.h
    src/param/util.h
    src/fold/fold.cpp src/fold/fold.h 
    src/fold/zuker.cpp src/fold/zuker.h 
    src/fold/nussinov.cpp src/fold/nussinov.h 
    src/fold/trimatrix.h )
target_compile_features(interface PUBLIC cxx_std_17)
#target_link_libraries(dnnfold)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -D_LIBCPP_DEBUG=0")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -D_GLIBCXX_DEBUG")
