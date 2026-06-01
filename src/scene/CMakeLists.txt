# scene: scene graph, instance/mesh/material aggregation, accel structure hook
add_library(lux_scene STATIC
    prepared_scene.cu
)
target_include_directories(lux_scene PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lux_scene PUBLIC
    lux_geometry
    lux_material
    lux_bvh
    lux_light
)
