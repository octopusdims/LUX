cmake_minimum_required(VERSION 3.24)

option(LUX_ENABLE_CUDA "Enable CUDA GPU rendering support" ON)

if(LUX_ENABLE_CUDA)
    set(LUX_PROJECT_LANGUAGES CXX CUDA)
else()
    set(LUX_PROJECT_LANGUAGES CXX)
endif()

Project(LUX
    VERSION 0.1
    DESCRIPTION "LUX: A rendering infrastructure for my research."
    LANGUAGES ${LUX_PROJECT_LANGUAGES}
)

# Set C++/CUDA standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(LUX_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# CUDA architecture
if(LUX_ENABLE_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native) 
endif()

# Global compile options
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(LUX_BUILD_TESTS "Build LUX tests" OFF)
option(LUX_BUILD_TOOLS "Build LUX tools" OFF)
option(LUX_ENABLE_OPENEXR "Enable OpenEXR environment image loading" ON)

set(LUX_DEFAULT_SCENE_DEEP_CURRENT_CHECKS OFF)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    set(LUX_DEFAULT_SCENE_DEEP_CURRENT_CHECKS ON)
endif()
option(LUX_ENABLE_SCENE_DEEP_CURRENT_CHECKS
       "Enable deep validation for prepared scene-derived state at construction/upload boundaries"
       ${LUX_DEFAULT_SCENE_DEEP_CURRENT_CHECKS})

if(LUX_ENABLE_CUDA)
    add_compile_definitions(LUX_ENABLE_CUDA=1)
else()
    add_compile_definitions(LUX_ENABLE_CUDA=0)
endif()

if(LUX_ENABLE_SCENE_DEEP_CURRENT_CHECKS)
    add_compile_definitions(LUX_ENABLE_SCENE_DEEP_CURRENT_CHECKS=1)
else()
    add_compile_definitions(LUX_ENABLE_SCENE_DEEP_CURRENT_CHECKS=0)
endif()

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
