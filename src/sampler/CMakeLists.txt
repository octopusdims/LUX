# sampler: PCG, Sobol, blue noise
add_library(lux_sampler INTERFACE)
target_include_directories(lux_sampler INTERFACE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(lux_sampler INTERFACE lux_core)
