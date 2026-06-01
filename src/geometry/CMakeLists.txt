# geometry: triangle, mesh, vertex/index buffers
add_library(lux_geometry INTERFACE)
target_include_directories(lux_geometry INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lux_geometry INTERFACE lux_core)
