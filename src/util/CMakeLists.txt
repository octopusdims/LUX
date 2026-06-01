# util: logging, timers, assertions, obj loading
add_library(lux_util STATIC
    env_image_loader.cpp
    obj_loader.cpp
)
target_include_directories(lux_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lux_util PUBLIC
    lux_core
    lux_scene
    lux_geometry
    lux_material
)

if(TARGET OpenEXR::OpenEXR)
    target_link_libraries(lux_util PUBLIC OpenEXR::OpenEXR)
    target_compile_definitions(lux_util PUBLIC LUX_HAS_OPENEXR=1)
else()
    target_compile_definitions(lux_util PUBLIC LUX_HAS_OPENEXR=0)
endif()
