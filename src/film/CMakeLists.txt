# film: framebuffer, EXR output, AOV (albedo/normal/depth)
add_library(lux_film INTERFACE)
target_include_directories(lux_film INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lux_film INTERFACE lux_core)
