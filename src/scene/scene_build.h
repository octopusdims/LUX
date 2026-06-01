#pragma once

#ifndef LUX_SCENE_BUILD_H
#define LUX_SCENE_BUILD_H

#include "scene/scene_edit.h"
#include "scene/scene_validate.h"

LuxInline int add_point_light(Scene& scene, const vec3& position, const vec3& intensity) {
    Light light;
    light.kind = LightKind::Point;
    light.point = PointLight{position, intensity};
    scene.lights.push_back(light);
    mark_scene_modified(scene);
    return static_cast<int>(scene.lights.size()) - 1;
}

LuxInline int add_distant_light(Scene& scene, const vec3& direction_to_light,
                                const vec3& radiance) {
    if (length2(direction_to_light) == 0) {
        throw std::runtime_error("add_distant_light: direction must be non-zero");
    }
    Light light;
    light.kind = LightKind::Distant;
    light.distant = DistantLight{normalize(direction_to_light), radiance};
    scene.lights.push_back(light);
    mark_scene_modified(scene);
    return static_cast<int>(scene.lights.size()) - 1;
}

LuxInline int add_uniform_infinite_light(Scene& scene, const vec3& radiance) {
    Light light;
    light.kind = LightKind::UniformInfinite;
    light.uniform_infinite = UniformInfiniteLight{radiance};
    scene.lights.push_back(light);
    mark_scene_modified(scene);
    return static_cast<int>(scene.lights.size()) - 1;
}

LuxInline SceneImageLightAsset make_image_light_asset(
        const std::vector<vec3>& pixels, int width, int height, const vec3& scale) {
    if (width <= 0 || height <= 0
        || pixels.size() != static_cast<size_t>(width * height)) {
        throw std::runtime_error("make_image_light_asset: pixel count must match dimensions");
    }

    SceneImageLightAsset asset;
    asset.pixels = pixels;
    asset.width = width;
    asset.height = height;
    asset.cdf.reserve(pixels.size());
    Float sum = 0;
    for (int y = 0; y < height; ++y) {
        Float theta = kPi * ((Float(y) + Float(0.5)) / Float(height));
        Float sin_theta = fmaxf(Float(0), sinf(theta));
        for (int x = 0; x < width; ++x) {
            vec3 radiance = pixels[y * width + x] * scale;
            Float weight = fmaxf(Float(0), luminance(radiance)) * sin_theta;
            sum += weight;
            asset.cdf.push_back(sum);
        }
    }
    asset.total_weight = sum;
    return asset;
}

LuxInline int add_image_infinite_light(Scene& scene, const std::vector<vec3>& pixels,
                                       int width, int height,
                                       const vec3& scale = vec3(1)) {
    SceneImageLightAsset asset = make_image_light_asset(pixels, width, height, scale);
    scene.image_light_assets.push_back(std::move(asset));
    int image_id = static_cast<int>(scene.image_light_assets.size()) - 1;
    const SceneImageLightAsset& stored = scene.image_light_assets[image_id];

    Light light;
    light.kind = LightKind::ImageInfinite;
    light.image_infinite = ImageInfiniteLight{
        stored.pixels.data(),
        stored.cdf.data(),
        image_id,
        stored.width,
        stored.height,
        stored.total_weight,
        scale
    };
    scene.lights.push_back(light);
    mark_scene_modified(scene);
    return static_cast<int>(scene.lights.size()) - 1;
}

LuxInline void refresh_scene_light_asset_pointers(Scene& scene) {
    for (Light& light : scene.lights) {
        if (light.kind != LightKind::ImageInfinite) continue;
        int image_id = light.image_infinite.image_id;
        if (image_id < 0
            || image_id >= static_cast<int>(scene.image_light_assets.size())) {
            throw std::runtime_error("Scene validation failed: image infinite light asset id out of range");
        }
        const SceneImageLightAsset& asset = scene.image_light_assets[image_id];
        light.image_infinite.pixels = asset.pixels.data();
        light.image_infinite.cdf = asset.cdf.data();
        light.image_infinite.width = asset.width;
        light.image_infinite.height = asset.height;
        light.image_infinite.total_weight = asset.total_weight;
    }
}

LuxInline void finalize_scene(Scene& scene) {
    ensure_scene_instance_assets(scene);
    refresh_scene_geometry_cache(scene);
    refresh_scene_light_asset_pointers(scene);
    validate_scene_shape(scene);
    mark_scene_modified(scene);
}


#endif // LUX_SCENE_BUILD_H
