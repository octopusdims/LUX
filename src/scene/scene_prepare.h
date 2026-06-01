#pragma once

#ifndef LUX_SCENE_PREPARE_H
#define LUX_SCENE_PREPARE_H

#include <vector>
#include <utility>

#include "bvh/bvh_build_input.h"
#include "light/light_distribution.h"
#include "scene/scene.h"

LuxInline LightDistribution build_scene_light_distribution(const Scene& scene) {
    validate_scene_shape(scene);

    std::vector<AreaLightPrimitive> lights;
    LightDistribution generic_distribution;
    for (int triangle_id = 0; triangle_id < scene_triangle_count(scene); ++triangle_id) {
        PrimitiveRef primitive_ref = scene_primitive_ref(scene, triangle_id);
        SceneTriangle triangle = scene_triangle_view(scene, primitive_ref);
        int material_id = triangle.material_id;
        if (material_id < 0 || material_id >= static_cast<int>(scene.materials.size())) {
            continue;
        }
        const Material& material = scene.materials[material_id];
        if (!material.is_emissive()) {
            continue;
        }

        Float area = triangle_area(triangle.triangle);
        if (area <= 0) {
            continue;
        }

        lights.push_back(AreaLightPrimitive{primitive_ref, area});
        Light light;
        light.kind = LightKind::DiffuseArea;
        light.area = DiffuseAreaLight{
            -1,
            material.emission,
            material.emission_sidedness,
            material.emission_source,
            material.emission_image_id,
            material.emission_scale,
            area};
        light.area.primitive_ref = primitive_ref;
        append_light_distribution_entry(
            generic_distribution, light,
            area * fmaxf(Float(1e-6),
                         scene_material_emission_power_hint(scene, material)));
    }
    for (const Light& scene_light : scene.lights) {
        Light light = scene_light;
        Float power = Float(1);
        switch (light.kind) {
            case LightKind::Point:
                power = fmaxf(Float(1e-6), luminance(light.point.intensity));
                break;
            case LightKind::Distant:
                power = fmaxf(Float(1e-6), luminance(light.distant.radiance));
                break;
            case LightKind::UniformInfinite:
                power = fmaxf(Float(1e-6), luminance(light.uniform_infinite.radiance));
                break;
            case LightKind::ImageInfinite:
                power = fmaxf(Float(1e-6), light.image_infinite.total_weight);
                break;
            case LightKind::DiffuseArea:
                if (!primitive_ref_valid(light.area.primitive_ref)
                    && light.area.primitive_id >= 0
                    && light.area.primitive_id < scene_triangle_count(scene)) {
                    light.area.primitive_ref =
                        scene_primitive_ref(scene, light.area.primitive_id);
                }
                if (light.area.emission_source == EmissionSourceKind::Image) {
                    Material emission_material = Material::image_emissive(
                        light.area.emission_image_id, light.area.emission_scale,
                        vec3(0), light.area.sidedness);
                    power = fmaxf(
                        Float(1e-6),
                        light.area.area
                            * scene_material_emission_power_hint(scene, emission_material));
                } else {
                    power = fmaxf(Float(1e-6),
                                  light.area.area * luminance(light.area.emission));
                }
                break;
        }
        append_light_distribution_entry(generic_distribution, light, power);
    }
    LightDistribution distribution = build_area_light_distribution(lights);
    distribution.lights = std::move(generic_distribution.lights);
    distribution.light_power_cdf = std::move(generic_distribution.light_power_cdf);
    distribution.total_light_power = generic_distribution.total_light_power;
    return distribution;
}

LuxInline std::vector<BvhBuildPrimitive> collect_scene_mesh_bvh_build_primitives(
        const SceneMesh& mesh, int mesh_id = -1) {
    std::vector<BvhBuildPrimitive> primitives;
    primitives.reserve(scene_mesh_triangle_count(mesh));

    for (int triangle_id = 0; triangle_id < scene_mesh_triangle_count(mesh); ++triangle_id) {
        AABB bounds = triangle_bounds(scene_mesh_triangle_geometry(mesh, triangle_id));
        primitives.push_back(BvhBuildPrimitive{
            bounds,
            bounds.centroid(),
            triangle_id,
            -1,
            mesh_id,
            triangle_id,
            PrimitiveKind::Triangle
        });
    }

    return primitives;
}

LuxInline std::vector<BvhBuildPrimitive> collect_scene_tlas_build_primitives(
        const Scene& scene) {
    std::vector<BvhBuildPrimitive> primitives;
    primitives.reserve(scene.instances.size());

    for (int instance_id = 0; instance_id < static_cast<int>(scene.instances.size());
         ++instance_id) {
        const SceneInstance& instance = scene.instances[instance_id];
        if (instance.world_bounds.is_empty()) continue;
        primitives.push_back(BvhBuildPrimitive{
            instance.world_bounds,
            instance.world_bounds.centroid(),
            instance_id,
            instance_id,
            instance.mesh_id,
            -1,
            PrimitiveKind::None
        });
    }

    return primitives;
}

#endif // LUX_SCENE_PREPARE_H
