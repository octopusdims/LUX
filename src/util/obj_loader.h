#pragma once

#ifndef LUX_UTIL_OBJ_LOADER_H
#define LUX_UTIL_OBJ_LOADER_H

#include <string>
#include "scene/scene.h"

struct ObjLoadOptions {
    bool cap_boundary_holes = false;
    bool generate_missing_normals = false;
};

bool load_obj_into_scene(const std::string& obj_path, Scene& scene,
                         const ObjLoadOptions& options = {},
                         SceneMeshAppendRange* appended_range = nullptr);

#endif // LUX_UTIL_OBJ_LOADER_H
