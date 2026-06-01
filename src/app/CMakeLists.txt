add_library(lux_app_runner STATIC
    render_runner.cpp
)

target_include_directories(lux_app_runner PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(lux_app_runner PRIVATE
    lux_integrator
    lux_scene
    lux_film
    lux_bvh
    lux_geometry
    lux_camera
    lux_material
    lux_light
    lux_sampler
    lux_core
    lux_cpu_diagnostics
    lux_util
)

add_executable(lux main.cpp)

target_link_libraries(lux PRIVATE
    lux_app_runner
)

# Copy assets to the executable output directory so relative paths resolve
# at runtime (e.g. "assets/models/stanford-bunny.obj").
add_custom_command(TARGET lux POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:lux>/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/assets
        ${CMAKE_BINARY_DIR}/assets
    COMMENT "Copying assets to output directory"
)
