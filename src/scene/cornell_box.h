#pragma once

#ifndef LUX_SCENE_CORNELL_BOX_H
#define LUX_SCENE_CORNELL_BOX_H

#include "core/types.h"
#include "scene.h"

namespace cornell_box_detail {

LuxInline void add_box(Scene& scene, const vec3& p_min, const vec3& p_max, int material_id) {
    vec3 p000(p_min.x, p_min.y, p_min.z);
    vec3 p001(p_min.x, p_min.y, p_max.z);
    vec3 p010(p_min.x, p_max.y, p_min.z);
    vec3 p011(p_min.x, p_max.y, p_max.z);
    vec3 p100(p_max.x, p_min.y, p_min.z);
    vec3 p101(p_max.x, p_min.y, p_max.z);
    vec3 p110(p_max.x, p_max.y, p_min.z);
    vec3 p111(p_max.x, p_max.y, p_max.z);

    add_quad(scene, p000, p100, p110, p010, material_id);
    add_quad(scene, p101, p001, p011, p111, material_id);
    add_quad(scene, p001, p000, p010, p011, material_id);
    add_quad(scene, p100, p101, p111, p110, material_id);
    add_quad(scene, p010, p110, p111, p011, material_id);
    add_quad(scene, p001, p101, p100, p000, material_id);
}

} // namespace cornell_box_detail

LuxInline Scene make_cornell_box() {
    Scene scene;

    int white = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::lambert(vec3(0.78f, 0.78f, 0.74f)));
    int red = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::lambert(vec3(0.65f, 0.05f, 0.05f)));
    int green = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::lambert(vec3(0.12f, 0.45f, 0.15f)));
    int light = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::emissive(vec3(15.0f, 14.0f, 12.0f)));

    add_quad(scene, vec3(-1, 0, 2), vec3(1, 0, 2), vec3(1, 0, 0), vec3(-1, 0, 0), white);
    add_quad(scene, vec3(-1, 2, 0), vec3(1, 2, 0), vec3(1, 2, 2), vec3(-1, 2, 2), white);
    add_quad(scene, vec3(-1, 0, 0), vec3(1, 0, 0), vec3(1, 2, 0), vec3(-1, 2, 0), white);
    add_quad(scene, vec3(-1, 0, 2), vec3(-1, 0, 0), vec3(-1, 2, 0), vec3(-1, 2, 2), red);
    add_quad(scene, vec3(1, 0, 0), vec3(1, 0, 2), vec3(1, 2, 2), vec3(1, 2, 0), green);

    cornell_box_detail::add_box(scene, vec3(-0.65f, 0, 0.35f), vec3(-0.15f, 0.8f, 0.9f), white);
    cornell_box_detail::add_box(scene, vec3(0.25f, 0, 0.75f), vec3(0.75f, 1.2f, 1.35f), white);

    add_quad(scene, vec3(-0.35f, 1.99f, 0.75f), vec3(0.35f, 1.99f, 0.75f),
             vec3(0.35f, 1.99f, 1.35f), vec3(-0.35f, 1.99f, 1.35f), light);

    return scene;
}
#endif // LUX_SCENE_CORNELL_BOX_H
