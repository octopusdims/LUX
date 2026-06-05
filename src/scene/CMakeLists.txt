# scene: scene graph, instance/mesh/material aggregation, accel structure hook
if(LUX_ENABLE_CUDA)
    set(LUX_SCENE_SOURCES prepared_scene.cu)
else()
    set(LUX_SCENE_SOURCES prepared_scene_cpu.cpp)
endif()

add_library(lux_scene STATIC ${LUX_SCENE_SOURCES})
target_include_directories(lux_scene PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lux_scene PUBLIC
    lux_geometry
    lux_material
    lux_bvh
    lux_light
)
