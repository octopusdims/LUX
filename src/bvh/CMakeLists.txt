# bvh: LBVH build (host) + traversal (device)
add_library(lux_bvh STATIC
    gpu_bvh.cu
    bvh_overlap.cu
)
target_include_directories(lux_bvh PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lux_bvh PUBLIC lux_core lux_geometry)
