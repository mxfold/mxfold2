cmake_minimum_required(VERSION 3.10)
project(mxfold2 CXX)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(pybind11)
pybind11_add_module(interface
    interface.cpp 
    param/turner.cpp param/turner.h 
    param/positional.cpp param/positional.h 
    param/bpscore.cpp param/bpscore.h     
    param/mix.cpp param/mix.h
    param/util.h
    fold/fold.cpp fold/fold.h 
    fold/zuker.cpp fold/zuker.h 
    fold/nussinov.cpp fold/nussinov.h 
    fold/trimatrix.h )
target_compile_features(interface PUBLIC cxx_std_17)
#target_link_libraries(mxfold2)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -D_LIBCPP_DEBUG=0")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -D_GLIBCXX_DEBUG")
