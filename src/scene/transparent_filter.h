#pragma once

#ifndef LUX_SCENE_TRANSPARENT_FILTER_H
#define LUX_SCENE_TRANSPARENT_FILTER_H

#include "core/types.h"
#include "scene.h"

LuxInline Scene make_transparent_filter_scene() {
    Scene scene;

    int receiver = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::lambert(vec3(0.75f, 0.75f, 0.75f)));
    int area_light = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::emissive(vec3(8.0f, 8.0f, 8.0f)));
    int tinted_filter = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::tinted_filter(vec3(0.25f, 0.65f, 1.0f)));
    int clear_interface = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::interface());
    int card_light = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::emissive(vec3(3.0f, 2.2f, 1.2f)));

    add_quad(scene, vec3(-1.6f, 0.0f, 1.5f), vec3(1.6f, 0.0f, 1.5f),
             vec3(1.6f, 0.0f, -1.5f), vec3(-1.6f, 0.0f, -1.5f), receiver);

    add_quad(scene, vec3(-0.45f, 2.0f, -0.45f), vec3(0.45f, 2.0f, -0.45f),
             vec3(0.45f, 2.0f, 0.45f), vec3(-0.45f, 2.0f, 0.45f), area_light);

    add_quad(scene, vec3(-0.8f, 1.0f, -0.8f), vec3(0.8f, 1.0f, -0.8f),
             vec3(0.8f, 1.0f, 0.8f), vec3(-0.8f, 1.0f, 0.8f), tinted_filter);

    add_quad(scene, vec3(-0.35f, 0.35f, -1.25f), vec3(0.35f, 0.35f, -1.25f),
             vec3(0.35f, 1.05f, -1.25f), vec3(-0.35f, 1.05f, -1.25f), card_light);

    add_quad(scene, vec3(-0.55f, 0.2f, 0.75f), vec3(0.55f, 0.2f, 0.75f),
             vec3(0.55f, 1.2f, 0.75f), vec3(-0.55f, 1.2f, 0.75f), clear_interface);
    add_quad(scene, vec3(-0.55f, 0.2f, -0.2f), vec3(0.55f, 0.2f, -0.2f),
             vec3(0.55f, 1.2f, -0.2f), vec3(-0.55f, 1.2f, -0.2f), tinted_filter);

    return scene;
}

#endif // LUX_SCENE_TRANSPARENT_FILTER_H
