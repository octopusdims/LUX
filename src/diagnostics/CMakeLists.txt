add_library(lux_cpu_diagnostics STATIC
    cpu/cpu_diagnostics.cpp
    cpu/cpu_probe.cpp
    cpu/cpu_mesh.cpp
    cpu/cpu_aov.cpp
)

target_include_directories(lux_cpu_diagnostics PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(lux_cpu_diagnostics PUBLIC
    lux_core
    lux_scene
    lux_geometry
    lux_bvh
    lux_camera
    lux_film
    lux_material
    lux_light
)
