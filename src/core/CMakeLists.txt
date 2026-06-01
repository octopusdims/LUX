# core: math (vec3, mat4), ray, aabb, rng, memory
# Mostly header-only (.cuh), host + device dual-use.
# Add .cu files here when device-side kernels are needed.
add_library(lux_core INTERFACE)
target_include_directories(lux_core INTERFACE ${PROJECT_SOURCE_DIR}/src)
