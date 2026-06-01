# camera: pinhole, thin-lens
add_library(lux_camera INTERFACE)
target_include_directories(lux_camera INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lux_camera INTERFACE lux_core)
