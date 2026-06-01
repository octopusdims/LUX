# light: area light, env map, light sampler
add_library(lux_light STATIC
    light_distribution.cpp
)

target_include_directories(lux_light PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(lux_light PUBLIC
    lux_core
    lux_geometry
    lux_material
    lux_sampler
)
