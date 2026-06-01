# material: BSDF (Lambert, GGX), sampling, evaluation
add_library(lux_material INTERFACE)
target_include_directories(lux_material INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lux_material INTERFACE lux_core lux_sampler)
