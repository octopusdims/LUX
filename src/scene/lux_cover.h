#pragma once

#ifndef LUX_SCENE_LUX_COVER_H
#define LUX_SCENE_LUX_COVER_H

#include <filesystem>
#include <stdexcept>
#include <string>

#include "core/types.h"
#include "material/material.h"
#include "scene.h"
#include "util/obj_loader.h"

namespace lux_cover_detail {

LuxInline int require_material_id(const ObjLoadResult& result, const char* name) {
    auto found = result.material_name_to_id.find(name);
    if (found == result.material_name_to_id.end()) {
        throw std::runtime_error(std::string("LUX cover OBJ is missing material: ") + name);
    }
    return found->second;
}

} // namespace lux_cover_detail

LuxInline Scene make_lux_cover_scene(const std::filesystem::path& asset_root) {
    Scene scene;

    int floor_mat = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::lambert(vec3(0.42f, 0.43f, 0.42f)));
    int key_light_mat = static_cast<int>(scene.materials.size());
    scene.materials.push_back(
        Material::emissive(vec3(22.0f, 21.0f, 19.0f), vec3(0),
                           EmissionSidedness::DoubleSided));
    int l_mat = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::lambert(vec3(0.78f, 0.76f, 0.70f)));
    int u_mat = static_cast<int>(scene.materials.size());
    scene.materials.push_back(
        Material::dielectric(vec3(0.90f, 0.97f, 1.00f), Float(1.5)));
    int x_mat = static_cast<int>(scene.materials.size());
    scene.materials.push_back(
        Material::ggx(vec3(0.82f, 0.78f, 0.68f), Float(0.22), Float(1)));

    add_quad(scene, vec3(-2.1f, 0.0f, 0.18f), vec3(2.1f, 0.0f, 0.18f),
             vec3(2.1f, 0.0f, -2.2f), vec3(-2.1f, 0.0f, -2.2f), floor_mat);
    add_quad(scene, vec3(-2.1f, 0.0f, 0.18f), vec3(-2.1f, 1.8f, 0.18f),
             vec3(2.1f, 1.8f, 0.18f), vec3(2.1f, 0.0f, 0.18f), floor_mat);
    add_quad(scene, vec3(-1.25f, 1.85f, -0.55f), vec3(1.25f, 1.85f, -0.55f),
             vec3(1.25f, 1.85f, 0.80f), vec3(-1.25f, 1.85f, 0.80f), key_light_mat);

    std::filesystem::path model_path = asset_root / "models" / "LUX.obj";
    int first_logo_triangle = scene_triangle_count(scene);
    ObjLoadResult logo_result;
    ObjLoadOptions options;
    options.normal_mode = ObjNormalMode::GenerateMissingSmooth;
    options.boundary_topology = ObjBoundaryTopology::WeldedPositions;
    options.placement.scale_mode = ObjPlacementScaleMode::XExtent;
    options.placement.target_extent = Float(1.85);
    options.placement.center_xz = true;
    options.placement.ground_to_y = true;
    options.placement.target_y = Float(0);
    options.material_bindings["L_lambert"] = l_mat;
    options.material_bindings["U_glass"] = u_mat;
    options.material_bindings["X_metal"] = x_mat;
    if (!load_obj_into_scene_with_result(model_path.string(), scene, options, logo_result)) {
        throw std::runtime_error("Failed to load LUX cover OBJ: " + model_path.string());
    }
    if (scene_triangle_count(scene) == first_logo_triangle) {
        throw std::runtime_error("LUX cover OBJ loaded no triangles: " + model_path.string());
    }

    if (lux_cover_detail::require_material_id(logo_result, "L_lambert") != l_mat
        || lux_cover_detail::require_material_id(logo_result, "U_glass") != u_mat
            || lux_cover_detail::require_material_id(logo_result, "X_metal") != x_mat) {
        throw std::runtime_error("LUX cover OBJ material bindings did not round-trip");
    }

    return scene;
}

LuxInline Scene make_lux_cover_scene() {
    return make_lux_cover_scene(std::filesystem::path("assets"));
}

#endif // LUX_SCENE_LUX_COVER_H
