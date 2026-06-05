#pragma once

#ifndef LUX_SCENE_PREPARE_H
#define LUX_SCENE_PREPARE_H

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <utility>

#include "bvh/bvh_build_input.h"
#include "light/light_distribution.h"
#include "scene/scene.h"

#ifndef LUX_ENABLE_SCENE_DEEP_CURRENT_CHECKS
#define LUX_ENABLE_SCENE_DEEP_CURRENT_CHECKS 0
#endif

namespace scene_light_sampling_detail {

LuxInline bool close_float(Float a, Float b) {
    Float scale = fmaxf(Float(1), fmaxf(fabsf(a), fabsf(b)));
    return fabsf(a - b) <= Float(1e-5) * scale;
}

LuxInline bool same_vec3(const vec3& a, const vec3& b) {
    return close_float(a.x, b.x) && close_float(a.y, b.y) && close_float(a.z, b.z);
}

LuxInline bool same_aabb(const AABB& actual, const AABB& expected) {
    return same_vec3(actual.p_min, expected.p_min)
        && same_vec3(actual.p_max, expected.p_max);
}

LuxInline bool same_diffuse_area_light(const DiffuseAreaLight& actual,
                                       const DiffuseAreaLight& expected) {
    return actual.primitive_id == expected.primitive_id
        && same_primitive_ref(actual.primitive_ref, expected.primitive_ref)
        && same_vec3(actual.emission, expected.emission)
        && actual.sidedness == expected.sidedness
        && actual.emission_source == expected.emission_source
        && actual.emission_image_id == expected.emission_image_id
        && same_vec3(actual.emission_scale, expected.emission_scale)
        && close_float(actual.area, expected.area);
}

LuxInline bool same_point_light(const PointLight& actual, const PointLight& expected) {
    return same_vec3(actual.position, expected.position)
        && same_vec3(actual.intensity, expected.intensity);
}

LuxInline bool same_distant_light(const DistantLight& actual, const DistantLight& expected) {
    return same_vec3(actual.direction_to_light, expected.direction_to_light)
        && same_vec3(actual.radiance, expected.radiance);
}

LuxInline bool same_uniform_infinite_light(const UniformInfiniteLight& actual,
                                           const UniformInfiniteLight& expected) {
    return same_vec3(actual.radiance, expected.radiance);
}

LuxInline bool same_image_infinite_light(const ImageInfiniteLight& actual,
                                         const ImageInfiniteLight& expected) {
    return actual.pixels == expected.pixels
        && actual.cdf == expected.cdf
        && actual.image_id == expected.image_id
        && actual.width == expected.width
        && actual.height == expected.height
        && close_float(actual.total_weight, expected.total_weight)
        && same_vec3(actual.scale, expected.scale);
}

LuxInline bool same_light_payload(const Light& actual, const Light& expected) {
    if (actual.kind != expected.kind) return false;
    switch (expected.kind) {
        case LightKind::DiffuseArea:
            return same_diffuse_area_light(actual.area, expected.area);
        case LightKind::Point:
            return same_point_light(actual.point, expected.point);
        case LightKind::Distant:
            return same_distant_light(actual.distant, expected.distant);
        case LightKind::UniformInfinite:
            return same_uniform_infinite_light(
                actual.uniform_infinite, expected.uniform_infinite);
        case LightKind::ImageInfinite:
            return same_image_infinite_light(
                actual.image_infinite, expected.image_infinite);
    }
    return false;
}

LuxInline bool same_alias_table(const std::vector<AliasTableEntry>& actual,
                                const std::vector<AliasTableEntry>& expected) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!close_float(actual[i].probability, expected[i].probability)) return false;
        if (actual[i].alias != expected[i].alias) return false;
        if (!close_float(actual[i].pmf, expected[i].pmf)) return false;
    }
    return true;
}

LuxInline bool same_light_bvh_node(const LightBvhNode& actual,
                                   const LightBvhNode& expected) {
    return same_aabb(actual.bounds, expected.bounds)
        && actual.left == expected.left
        && actual.right == expected.right
        && actual.parent == expected.parent
        && actual.light_index == expected.light_index
        && close_float(actual.power, expected.power);
}

LuxInline bool same_discrete_sampling_table(const DiscreteSamplingTable& actual,
                                            const DiscreteSamplingTable& expected) {
    if (actual.cdf.size() != expected.cdf.size()) return false;
    if (!close_float(actual.total_weight, expected.total_weight)) return false;
    if (!same_alias_table(actual.alias_table, expected.alias_table)) return false;
    for (std::size_t i = 0; i < expected.cdf.size(); ++i) {
        if (!close_float(actual.cdf[i], expected.cdf[i])) return false;
    }
    return true;
}

LuxInline bool same_area_light_sampling_table(const AreaLightSamplingTable& actual,
                                              const AreaLightSamplingTable& expected) {
    if (actual.primitive_refs.size() != expected.primitive_refs.size()
        || actual.primitive_areas.size() != expected.primitive_areas.size()) {
        return false;
    }
    if (!same_discrete_sampling_table(
            actual.area_distribution, expected.area_distribution)) {
        return false;
    }
    for (std::size_t i = 0; i < expected.primitive_refs.size(); ++i) {
        if (!same_primitive_ref(actual.primitive_refs[i], expected.primitive_refs[i])) return false;
        if (!close_float(actual.primitive_areas[i], expected.primitive_areas[i])) return false;
    }
    return true;
}

LuxInline bool same_light_selection_table(const LightSelectionTable& actual,
                                          const LightSelectionTable& expected) {
    if (actual.lights.size() != expected.lights.size()) return false;
    if (!same_discrete_sampling_table(
            actual.power_distribution, expected.power_distribution)) {
        return false;
    }
    for (std::size_t i = 0; i < expected.lights.size(); ++i) {
        if (!same_light_payload(actual.lights[i], expected.lights[i])) return false;
    }
    return true;
}

LuxInline bool same_light_bvh_table(const LightBvhTable& actual,
                                    const LightBvhTable& expected) {
    if (actual.nodes.size() != expected.nodes.size()
        || actual.leaf_nodes_by_light_index.size()
            != expected.leaf_nodes_by_light_index.size()
        || actual.non_bvh_light_indices.size()
            != expected.non_bvh_light_indices.size()) {
        return false;
    }
    if (actual.root != expected.root) return false;
    if (!same_discrete_sampling_table(
            actual.non_bvh_light_distribution,
            expected.non_bvh_light_distribution)) {
        return false;
    }
    for (std::size_t i = 0; i < expected.nodes.size(); ++i) {
        if (!same_light_bvh_node(actual.nodes[i], expected.nodes[i])) return false;
    }
    for (std::size_t i = 0; i < expected.leaf_nodes_by_light_index.size(); ++i) {
        if (actual.leaf_nodes_by_light_index[i]
            != expected.leaf_nodes_by_light_index[i]) {
            return false;
        }
    }
    for (std::size_t i = 0; i < expected.non_bvh_light_indices.size(); ++i) {
        if (actual.non_bvh_light_indices[i]
            != expected.non_bvh_light_indices[i]) {
            return false;
        }
    }
    return true;
}

LuxInline bool same_surface_light_lookup_table(
        const SurfaceLightLookupTable& actual,
        const SurfaceLightLookupTable& expected) {
    if (actual.primitive_light_indices.size()
        != expected.primitive_light_indices.size()) {
        return false;
    }
    for (std::size_t i = 0; i < expected.primitive_light_indices.size(); ++i) {
        if (actual.primitive_light_indices[i]
            != expected.primitive_light_indices[i]) {
            return false;
        }
    }
    return true;
}

LuxInline bool same_prepared_light_sampling(const PreparedLightSampling& actual,
                                            const PreparedLightSampling& expected) {
    if (!actual.source_scene.matches(expected.source_scene)) {
        return false;
    }
    return same_area_light_sampling_table(actual.area_lights, expected.area_lights)
        && same_light_selection_table(actual.light_selection, expected.light_selection)
        && same_light_bvh_table(actual.light_bvh, expected.light_bvh)
        && same_surface_light_lookup_table(actual.surface_lookup, expected.surface_lookup);
}

LuxInline void mark_primitive_light_index(std::vector<int>& primitive_light_indices,
                                          int primitive_id,
                                          int light_index) {
    if (primitive_id < 0
        || primitive_id >= static_cast<int>(primitive_light_indices.size())
        || light_index < 0) {
        return;
    }
    int& stored = primitive_light_indices[primitive_id];
    stored = stored == -1 ? light_index : -2;
}

LuxInline int resolve_area_light_primitive_id(const Scene& scene,
                                              const DiffuseAreaLight& light,
                                              PrimitiveRef& primitive_ref) {
    if (!primitive_ref_valid(primitive_ref)
        && light.primitive_id >= 0
        && light.primitive_id < scene_triangle_count(scene)) {
        primitive_ref = scene_primitive_ref(scene, light.primitive_id);
    }
    if (!primitive_ref_valid(primitive_ref)
        || !scene_primitive_ref_exists(scene, primitive_ref)) {
        return -1;
    }
    return scene_primitive_index(scene, primitive_ref);
}

LuxInline bool light_bounds_for_bvh(const Scene& scene,
                                    const Light& light,
                                    AABB& bounds,
                                    int& primitive_id) {
    primitive_id = -1;
    switch (light.kind) {
        case LightKind::Point:
            bounds = AABB(light.point.position);
            return true;
        case LightKind::DiffuseArea: {
            PrimitiveRef primitive_ref = light.area.primitive_ref;
            primitive_id = resolve_area_light_primitive_id(scene, light.area, primitive_ref);
            if (primitive_id < 0) return false;
            bounds = triangle_bounds(scene_triangle_geometry(scene, primitive_ref));
            return !bounds.is_empty();
        }
        case LightKind::Distant:
        case LightKind::UniformInfinite:
        case LightKind::ImageInfinite:
            return false;
    }
    return false;
}

} // namespace scene_light_sampling_detail

LuxInline PreparedLightSampling prepare_scene_light_sampling(const Scene& scene) {
    validate_scene_shape(scene);

    std::vector<AreaLightPrimitive> area_light_primitives;
    LightSelectionTable light_selection;
    std::vector<LightBvhBuildItem> light_bvh_items;
    std::vector<int> primitive_light_indices(scene_triangle_count(scene), -1);
    auto append_generic_light = [&](const Light& light,
                                    Float power,
                                    bool has_bvh_bounds,
                                    const AABB& bvh_bounds,
                                    int primitive_id = -1) {
        std::size_t before = light_selection.lights.size();
        append_light_selection_entry(light_selection, light, power);
        if (light_selection.lights.size() == before) return;

        int light_index = static_cast<int>(light_selection.lights.size()) - 1;
        if (has_bvh_bounds) {
            light_bvh_items.push_back(LightBvhBuildItem{
                bvh_bounds,
                bvh_bounds.centroid(),
                light_index,
                power
            });
        }
        if (light.kind == LightKind::DiffuseArea) {
            scene_light_sampling_detail::mark_primitive_light_index(
                primitive_light_indices, primitive_id, light_index);
        }
    };

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

        area_light_primitives.push_back(AreaLightPrimitive{primitive_ref, area});
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
        append_generic_light(
            light,
            area * fmaxf(Float(1e-6),
                         scene_material_emission_power_hint(scene, material)),
            true,
            triangle_bounds(triangle.triangle),
            triangle_id);
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
        AABB light_bounds;
        int primitive_id = -1;
        bool has_bvh_bounds = scene_light_sampling_detail::light_bounds_for_bvh(
            scene, light, light_bounds, primitive_id);
        append_generic_light(light, power, has_bvh_bounds, light_bounds, primitive_id);
    }
    PreparedLightSampling sampling;
    sampling.area_lights = build_area_light_sampling_table(area_light_primitives);
    sampling.light_selection = std::move(light_selection);
    sampling.surface_lookup.primitive_light_indices = std::move(primitive_light_indices);
    sampling.source_scene = scene_stamp(scene);
    build_light_bvh(sampling.light_bvh, sampling.light_selection, light_bvh_items);
    finalize_prepared_light_sampling_tables(sampling);
    return sampling;
}

LuxInline void require_prepared_light_sampling_stamp_current(
        const Scene& scene,
        const PreparedLightSampling& sampling,
        const char* message) {
    if (!sampling.source_scene.matches(scene_stamp(scene))) {
        throw std::runtime_error(message);
    }
}

LuxInline void require_prepared_light_sampling_deep_current(
        const Scene& scene,
        const PreparedLightSampling& sampling,
        const char* message) {
    PreparedLightSampling expected = prepare_scene_light_sampling(scene);
    if (!scene_light_sampling_detail::same_prepared_light_sampling(
            sampling, expected)) {
        throw std::runtime_error(message);
    }
}

LuxInline void require_prepared_light_sampling_current(
        const Scene& scene,
        const PreparedLightSampling& sampling,
        const char* message) {
#if LUX_ENABLE_SCENE_DEEP_CURRENT_CHECKS
    require_prepared_light_sampling_deep_current(scene, sampling, message);
#else
    require_prepared_light_sampling_stamp_current(scene, sampling, message);
#endif
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
