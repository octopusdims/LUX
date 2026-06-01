#pragma once

#ifndef LUX_SCENE_STANFORD_DRAGON_H
#define LUX_SCENE_STANFORD_DRAGON_H

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>

#include "core/types.h"
#include "geometry/triangle.h"
#include "material/material.h"
#include "scene.h"
#include "util/obj_loader.h"

LuxInline void normalize_scene_vertices(Scene& scene, const SceneMeshAppendRange& range,
                                        Float target_max_extent, Float floor_y) {
    if (range.vertex_count <= 0) return;
    if (range.mesh_id < 0 || range.mesh_id >= static_cast<int>(scene.mesh_assets.size())) {
        throw std::runtime_error("normalize_scene_vertices: mesh id out of range");
    }

    SceneMesh& mesh = scene.mesh_assets[range.mesh_id];
    vec3 bmin(std::numeric_limits<Float>::infinity());
    vec3 bmax(-std::numeric_limits<Float>::infinity());
    int end_vertex = range.first_vertex + range.vertex_count;
    for (int i = range.first_vertex; i < end_vertex; ++i) {
        const vec3& vertex = mesh.vertices[i];
        bmin.x = std::min(bmin.x, vertex.x);
        bmin.y = std::min(bmin.y, vertex.y);
        bmin.z = std::min(bmin.z, vertex.z);
        bmax.x = std::max(bmax.x, vertex.x);
        bmax.y = std::max(bmax.y, vertex.y);
        bmax.z = std::max(bmax.z, vertex.z);
    }

    vec3 extent = bmax - bmin;
    Float max_extent = std::max({extent.x, extent.y, extent.z});
    if (max_extent <= 0) return;

    Float scale = target_max_extent / max_extent;
    vec3 center(Float(0.5) * (bmin.x + bmax.x),
                bmin.y,
                Float(0.5) * (bmin.z + bmax.z));

    for (int i = range.first_vertex; i < end_vertex; ++i) {
        mesh.vertices[i] = (mesh.vertices[i] - center) * scale + vec3(0, floor_y, 0);
    }
    commit_scene_geometry_edit(scene);
}

LuxInline Scene make_stanford_dragon_scene(const std::filesystem::path& asset_root,
                                           ObjLoadOptions obj_options) {
    Scene scene;

    int floor_mat = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::lambert(vec3(0.5f, 0.5f, 0.5f)));

    int light_mat = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::emissive(vec3(15.0f, 14.0f, 12.0f)));

    add_quad(scene, vec3(-1.0f, 0.0f, 1.0f), vec3(1.0f, 0.0f, 1.0f),
             vec3(1.0f, 0.0f, -1.0f), vec3(-1.0f, 0.0f, -1.0f), floor_mat);

    add_quad(scene, vec3(-0.35f, 0.65f, -0.35f), vec3(0.35f, 0.65f, -0.35f),
             vec3(0.35f, 0.65f, 0.35f), vec3(-0.35f, 0.65f, 0.35f), light_mat);

    int first_dragon_triangle = scene_triangle_count(scene);
    std::filesystem::path model_path = asset_root / "models" / "dragon.obj";
    SceneMeshAppendRange dragon_range;
    if (!load_obj_into_scene(model_path.string(), scene, obj_options, &dragon_range)) {
        throw std::runtime_error("Failed to load dragon OBJ: " + model_path.string());
    }
    if (scene_triangle_count(scene) == first_dragon_triangle) {
        throw std::runtime_error("Dragon OBJ loaded no triangles: " + model_path.string());
    }

    for (int i = 2; i < static_cast<int>(scene.materials.size()); ++i) {
        if (scene.materials[i].type == MaterialType::Lambert
            && !scene.materials[i].is_emissive()) {
            set_scene_material(scene, i, Material::thin_lambert(scene.materials[i].albedo));
        }
    }
    normalize_scene_vertices(scene, dragon_range, Float(0.5), Float(0));
    return scene;
}

LuxInline Scene make_stanford_dragon_scene(const std::filesystem::path& asset_root) {
    return make_stanford_dragon_scene(asset_root, ObjLoadOptions{});
}

LuxInline Scene make_stanford_dragon_scene(ObjLoadOptions obj_options = {}) {
    return make_stanford_dragon_scene(std::filesystem::path("assets"), obj_options);
}

#endif // LUX_SCENE_STANFORD_DRAGON_H
