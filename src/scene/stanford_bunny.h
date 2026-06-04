#pragma once

#ifndef LUX_SCENE_STANFORD_BUNNY_H
#define LUX_SCENE_STANFORD_BUNNY_H

#include <filesystem>
#include <stdexcept>

#include "core/types.h"
#include "geometry/triangle.h"
#include "material/material.h"
#include "scene.h"
#include "util/obj_loader.h"

LuxInline Scene make_stanford_bunny_scene(const std::filesystem::path& asset_root,
                                          ObjLoadOptions obj_options) {
    Scene scene;

    // Materials
    int floor_mat = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::lambert(vec3(0.5f, 0.5f, 0.5f)));

    int light_mat = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::emissive(vec3(15.0f, 14.0f, 12.0f)));

    // Floor: large grey quad at y = 0, normal pointing +y
    add_quad(scene, vec3(-1.0f, 0.0f, 1.0f), vec3(1.0f, 0.0f, 1.0f),
             vec3(1.0f, 0.0f, -1.0f), vec3(-1.0f, 0.0f, -1.0f), floor_mat);

    // Ceiling light: emissive quad at y = 0.5, normal pointing -y (toward bunny)
    add_quad(scene, vec3(-0.3f, 0.5f, -0.3f), vec3(0.3f, 0.5f, -0.3f),
             vec3(0.3f, 0.5f, 0.3f), vec3(-0.3f, 0.5f, 0.3f), light_mat);

    // Load Stanford Bunny from OBJ
    int first_bunny_triangle = scene_triangle_count(scene);
    int first_bunny_material = static_cast<int>(scene.materials.size());
    std::filesystem::path model_path = asset_root / "models" / "stanford-bunny.obj";
    SceneMeshAppendRange bunny_range;
    obj_options.placement.ground_to_y = true;
    obj_options.placement.target_y = Float(0);
    if (!load_obj_into_scene(model_path.string(), scene, obj_options, &bunny_range)) {
        throw std::runtime_error("Failed to load Stanford Bunny OBJ: " + model_path.string());
    }
    if (scene_triangle_count(scene) == first_bunny_triangle) {
        throw std::runtime_error("Stanford Bunny OBJ loaded no triangles: " + model_path.string());
    }

    for (int i = first_bunny_material; i < static_cast<int>(scene.materials.size()); ++i) {
        if (scene.materials[i].type == MaterialType::Lambert
            && !scene.materials[i].is_emissive()) {
            set_scene_material(scene, i, Material::thin_lambert(scene.materials[i].albedo));
        }
    }

    return scene;
}

LuxInline Scene make_stanford_bunny_scene(const std::filesystem::path& asset_root) {
    return make_stanford_bunny_scene(asset_root, ObjLoadOptions{});
}

LuxInline Scene make_stanford_bunny_scene(ObjLoadOptions obj_options = {}) {
    return make_stanford_bunny_scene(std::filesystem::path("assets"), obj_options);
}

#endif // LUX_SCENE_STANFORD_BUNNY_H
