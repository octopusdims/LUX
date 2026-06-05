# integrator: rendering algorithms
#   integrator.h         — host abstract base class (pure C++)
#   wavefront_pt.cu      — baseline path tracing (CUDA kernels)
#   workqueue.cuh        — device-side work queue
#   pathstate.cuh        — SoA path state
# Submodules:
#   restir/              — ReSTIR PT
#   pathguide/           — path guiding
option(LUX_BUILD_RESTIR "Build ReSTIR PT extensions" OFF)
option(LUX_BUILD_PATHGUIDE "Build path guiding extensions" OFF)

if(LUX_BUILD_RESTIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/restir/CMakeLists.txt")
    add_subdirectory(restir)
endif()

if(LUX_BUILD_PATHGUIDE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pathguide/CMakeLists.txt")
    add_subdirectory(pathguide)
endif()

if(LUX_ENABLE_CUDA)
    add_library(lux_wavefront STATIC
        wavefront_kernels.cu
        wavefront_pt.cu
        wavefront_runtime.cu
        wavefront_scheduler.cu
    )
    target_include_directories(lux_wavefront PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(lux_wavefront PUBLIC
        lux_bvh
        lux_scene
        lux_film
        lux_camera
        lux_material
        lux_light
        lux_sampler
        lux_core
        lux_geometry
    )
endif()

add_library(lux_integrator STATIC
    cpu_path_tracer.cpp
    integrator_factory.cpp
)
target_include_directories(lux_integrator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lux_integrator PUBLIC
    lux_scene
    lux_bvh
    lux_sampler
    lux_camera
    lux_material
    lux_light
    lux_film
    lux_core
)

if(LUX_ENABLE_CUDA)
    target_link_libraries(lux_integrator PUBLIC lux_wavefront)
endif()

if(TARGET lux_restir)
    target_link_libraries(lux_integrator PUBLIC lux_restir)
endif()

if(TARGET lux_pathguide)
    target_link_libraries(lux_integrator PUBLIC lux_pathguide)
endif()
