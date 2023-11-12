cmake_minimum_required(VERSION 3.10)
project(mxfold2 CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DUSE_OPENMP)
endif()

#add_subdirectory(pybind11)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(interface
    interface.cpp 
    param/turner.cpp param/turner.h 
    param/contrafold.cpp param/contrafold.h 
    param/positional.cpp param/positional.h 
    param/positional_bl.cpp param/positional_bl.h 
    param/positional_1d.cpp param/positional_1d.h 
    param/bpscore.cpp param/bpscore.h     
    param/mix.cpp param/mix.h
    param/util.h
    fold/fold.cpp fold/fold.h 
    fold/zuker.cpp fold/zuker.h 
    fold/nussinov.cpp fold/nussinov.h 
    fold/trimatrix.h
    fold/linfold.cpp fold/linfold.h 
    )
target_compile_features(interface PUBLIC cxx_std_17)
if(OPENMP_FOUND)
    target_link_libraries(interface PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS interface DESTINATION ${CMAKE_SOURCE_DIR}/..)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -D_LIBCPP_DEBUG=0")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -D_GLIBCXX_DEBUG")
