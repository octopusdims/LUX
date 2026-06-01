#pragma once

#ifndef LUX_SCENE_DIRECT_LIGHT_TEST_H
#define LUX_SCENE_DIRECT_LIGHT_TEST_H

#include "core/types.h"
#include "scene.h"

LuxInline Scene make_direct_light_test_scene() {
    Scene scene;

    int receiver = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::lambert(vec3(0.8f, 0.8f, 0.8f)));
    int light = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::emissive(vec3(1.0f, 1.0f, 1.0f)));

    add_quad(scene, vec3(-1.0f, 0.0f, 1.0f), vec3(1.0f, 0.0f, 1.0f),
             vec3(1.0f, 0.0f, -1.0f), vec3(-1.0f, 0.0f, -1.0f), receiver);

    add_quad(scene, vec3(-0.2f, 1.4f, -0.2f), vec3(0.2f, 1.4f, -0.2f),
             vec3(0.2f, 1.4f, 0.2f), vec3(-0.2f, 1.4f, 0.2f), light);

    return scene;
}

#endif // LUX_SCENE_DIRECT_LIGHT_TEST_H
