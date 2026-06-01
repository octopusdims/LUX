cmake_minimum_required(VERSION 3.24)
Project(LUX
    VERSION 0.1
    DESCRIPTION "LUX: A rendering infrastructure for my research."
    LANGUAGES CXX CUDA
)

# Set C++/CUDA standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native) 
endif()

# Global compile options
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(LUX_BUILD_TESTS "Build LUX tests" OFF)
option(LUX_BUILD_TOOLS "Build LUX tools" OFF)
option(LUX_ENABLE_OPENEXR "Enable OpenEXR environment image loading" ON)

if(LUX_ENABLE_OPENEXR)
    find_package(OpenEXR CONFIG QUIET)
endif()

# Add subdirectories
add_subdirectory(src)

if(LUX_BUILD_TESTS AND EXISTS "${PROJECT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

if(LUX_BUILD_TOOLS AND EXISTS "${PROJECT_SOURCE_DIR}/tools/CMakeLists.txt")
    add_subdirectory(tools)
endif()
