# bvh: CPU traversal/build helpers are header-only; GPU builders are optional.
if(LUX_ENABLE_CUDA)
    add_library(lux_bvh STATIC
        gpu_bvh.cu
        bvh_overlap.cu
    )
    target_include_directories(lux_bvh PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(lux_bvh PUBLIC lux_core lux_geometry)
else()
    add_library(lux_bvh INTERFACE)
    target_include_directories(lux_bvh INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(lux_bvh INTERFACE lux_core lux_geometry)
endif()
