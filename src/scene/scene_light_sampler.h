#pragma once

#ifndef LUX_SCENE_LIGHT_SAMPLER_H
#define LUX_SCENE_LIGHT_SAMPLER_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "light/light_distribution.h"
#include "light/light_sampler.h"
#include "scene/scene.h"
#include "scene/scene_prepare.h"

struct SceneLightSampler {
    const Scene* scene = nullptr;
    const PreparedLightSampling* light_sampling = nullptr;
    SceneStamp source_scene;
    LightSamplerKind sampler_kind = LightSamplerKind::Power;

    bool empty() const;
};

namespace scene_light_sampler_detail {

LuxInline void require_current(const SceneLightSampler& sampler) {
    if (sampler.scene == nullptr) return;
    if (!sampler.source_scene.matches(scene_stamp(*sampler.scene))) {
        throw std::runtime_error(
            "SceneLightSampler cannot be used after its Scene has been modified");
    }
}

} // namespace scene_light_sampler_detail

LuxInline bool SceneLightSampler::empty() const {
    scene_light_sampler_detail::require_current(*this);
    return light_sampling == nullptr || light_sampling->empty();
}

LuxInline SceneLightSampler make_scene_light_sampler(
        const Scene& scene,
        const PreparedLightSampling& light_sampling,
        LightSamplerKind sampler_kind = LightSamplerKind::Power) {
    require_prepared_light_sampling_current(
        scene, light_sampling,
        "SceneLightSampler requires PreparedLightSampling built from the current Scene");
    return SceneLightSampler{
        &scene, &light_sampling, scene_stamp(scene), sampler_kind};
}

LuxInline bool sample_area_light(const SceneLightSampler& sampler,
                                 Float select_u, const vec2& surface_u,
                                 LightSample& sample) {
    if (sampler.scene == nullptr || sampler.light_sampling == nullptr) return false;
    scene_light_sampler_detail::require_current(sampler);
    const Scene& scene = *sampler.scene;
    const AreaLightSamplingTable& area_lights = sampler.light_sampling->area_lights;
    const DiscreteSamplingTable& area_distribution = area_lights.area_distribution;
    if (area_lights.primitive_refs.empty() || area_distribution.total_weight <= 0) return false;

    int light_index = area_distribution.alias_table.empty()
        ? sample_cdf_index(
            area_distribution.cdf.data(),
            static_cast<int>(area_distribution.cdf.size()),
            area_distribution.total_weight, select_u)
        : sample_alias_table_index(
            area_distribution.alias_table.data(),
            static_cast<int>(area_distribution.alias_table.size()),
            select_u);
    if (light_index < 0) return false;

    PrimitiveRef light_ref = area_lights.primitive_refs[light_index];
    int light_triangle_id = scene_primitive_index(scene, light_ref);
    SceneTriangle light_triangle = scene_triangle_view(scene, light_ref);
    if (light_triangle.material_id < 0
        || light_triangle.material_id >= static_cast<int>(scene.materials.size())) {
        return false;
    }

    const Material& material = scene.materials[light_triangle.material_id];
    sample.triangle_id = light_triangle_id;
    sample.primitive_ref = light_ref;
    sample.position = sample_triangle_area(light_triangle.triangle, surface_u);
    sample.normal = triangle_normal(light_triangle.triangle);
    vec2 bary_uv = sample_triangle_barycentric_uv(surface_u);
    sample.uv = interpolate_triangle_uv(light_triangle, bary_uv.x, bary_uv.y);
    sample.emission = scene_material_emission_value(scene, material, sample.uv);
    sample.emission_sidedness = material.emission_sidedness;
    sample.area = area_lights.primitive_areas[light_index];
    sample.pdf_area = Float(1) / area_distribution.total_weight;
    return sample.area > 0 && max_component(sample.emission) > 0;
}

LuxInline Float light_pmf(const SceneLightSampler& sampler,
                          const LightSampleContext& ctx,
                          int light_index);

LuxInline SampledLight sample_light(const SceneLightSampler& sampler,
                                    const LightSampleContext& ctx,
                                    Float select_u) {
    SampledLight sampled;
    if (sampler.light_sampling == nullptr) return sampled;
    scene_light_sampler_detail::require_current(sampler);
    const LightSelectionTable& selection = sampler.light_sampling->light_selection;
    const DiscreteSamplingTable& power_distribution = selection.power_distribution;
    const LightBvhTable& bvh = sampler.light_sampling->light_bvh;
    if (selection.lights.empty() || power_distribution.total_weight <= 0) return sampled;

    int light_index = -1;
    Float selected_pmf = 0;
    if (sampler.sampler_kind == LightSamplerKind::Uniform) {
        light_index = static_cast<int>(
            clamp_cdf_sample(select_u) * Float(selection.lights.size()));
        if (light_index >= static_cast<int>(selection.lights.size())) {
            light_index = static_cast<int>(selection.lights.size()) - 1;
        }
    } else if (sampler.sampler_kind == LightSamplerKind::Bvh
               && bvh.valid()) {
        light_index = sample_light_bvh_mixture(
            bvh.nodes.data(),
            static_cast<int>(bvh.nodes.size()),
            bvh.root,
            bvh.non_bvh_light_indices.data(),
            bvh.non_bvh_light_distribution.cdf.data(),
            bvh.non_bvh_light_distribution.alias_table.empty()
                ? nullptr
                : bvh.non_bvh_light_distribution.alias_table.data(),
            static_cast<int>(bvh.non_bvh_light_indices.size()),
            bvh.non_bvh_light_distribution.total_weight,
            ctx,
            select_u,
            selected_pmf);
    } else {
        light_index = power_distribution.alias_table.empty()
            ? sample_cdf_index(
                power_distribution.cdf.data(),
                static_cast<int>(power_distribution.cdf.size()),
                power_distribution.total_weight, select_u)
            : sample_alias_table_index(
                power_distribution.alias_table.data(),
                static_cast<int>(power_distribution.alias_table.size()),
                select_u);
    }
    if (light_index < 0) return sampled;
    sampled.light_index = light_index;
    sampled.light = selection.lights[light_index];
    sampled.pmf = selected_pmf > 0 ? selected_pmf : light_pmf(sampler, ctx, light_index);
    sampled.valid = sampled.pmf > 0;
    return sampled;
}

LuxInline SampledLight sample_light(const SceneLightSampler& sampler, Float select_u) {
    return sample_light(sampler, LightSampleContext{}, select_u);
}

LuxInline Float light_pmf(const SceneLightSampler& sampler,
                          const LightSampleContext& ctx,
                          int light_index) {
    if (sampler.light_sampling == nullptr) return 0;
    scene_light_sampler_detail::require_current(sampler);
    const LightSelectionTable& selection = sampler.light_sampling->light_selection;
    const DiscreteSamplingTable& power_distribution = selection.power_distribution;
    const LightBvhTable& bvh = sampler.light_sampling->light_bvh;
    if (light_index < 0 || light_index >= static_cast<int>(selection.lights.size())
        || power_distribution.total_weight <= 0) {
        return 0;
    }
    if (sampler.sampler_kind == LightSamplerKind::Uniform) {
        return Float(1) / Float(selection.lights.size());
    }
    if (sampler.sampler_kind == LightSamplerKind::Bvh
        && bvh.valid()) {
        return light_bvh_mixture_pmf(
            bvh.nodes.data(),
            static_cast<int>(bvh.nodes.size()),
            bvh.root,
            bvh.leaf_nodes_by_light_index.data(),
            static_cast<int>(bvh.leaf_nodes_by_light_index.size()),
            bvh.non_bvh_light_indices.data(),
            bvh.non_bvh_light_distribution.cdf.data(),
            bvh.non_bvh_light_distribution.alias_table.empty()
                ? nullptr
                : bvh.non_bvh_light_distribution.alias_table.data(),
            static_cast<int>(bvh.non_bvh_light_indices.size()),
            bvh.non_bvh_light_distribution.total_weight,
            ctx,
            light_index);
    }
    if (!power_distribution.alias_table.empty()) {
        return alias_table_pmf(
            power_distribution.alias_table.data(),
            static_cast<int>(power_distribution.alias_table.size()),
            light_index);
    }
    return cdf_interval_pmf(
        power_distribution.cdf.data(),
        static_cast<int>(power_distribution.cdf.size()),
        power_distribution.total_weight, light_index);
}

LuxInline Float light_pmf(const SceneLightSampler& sampler, int light_index) {
    return light_pmf(sampler, LightSampleContext{}, light_index);
}

LuxInline vec3 diffuse_area_light_emission_value(const Scene& scene,
                                                 const DiffuseAreaLight& light,
                                                 const vec2& uv);

LuxInline LightLiSample sample_light_li(const SceneLightSampler& sampler,
                                        const LightSampleContext& ctx,
                                        Float light_select_u,
                                        Float light_component_u,
                                        const vec2& light_u) {
    SampledLight sampled = sample_light(sampler, ctx, light_select_u);
    LightLiSample li;
    if (!sampled.valid) return li;

    if (sampled.light.kind == LightKind::DiffuseArea) {
        if (sampler.scene == nullptr) return li;
        PrimitiveRef primitive_ref = sampled.light.area.primitive_ref;
        if (!primitive_ref_valid(primitive_ref)
            && sampled.light.area.primitive_id >= 0
            && sampled.light.area.primitive_id < scene_triangle_count(*sampler.scene)) {
            primitive_ref = scene_primitive_ref(*sampler.scene, sampled.light.area.primitive_id);
        }
        if (!scene_primitive_ref_exists(*sampler.scene, primitive_ref)) return li;
        int triangle_id = scene_primitive_index(*sampler.scene, primitive_ref);
        SceneTriangle light_triangle = scene_triangle_view(*sampler.scene, primitive_ref);
        LightSample area_sample;
        area_sample.triangle_id = triangle_id;
        area_sample.primitive_ref = primitive_ref;
        area_sample.position = sample_triangle_area(light_triangle.triangle, light_u);
        area_sample.normal = triangle_normal(light_triangle.triangle);
        vec2 bary_uv = sample_triangle_barycentric_uv(light_u);
        area_sample.uv = interpolate_triangle_uv(light_triangle, bary_uv.x, bary_uv.y);
        area_sample.emission = diffuse_area_light_emission_value(
            *sampler.scene, sampled.light.area, area_sample.uv);
        area_sample.emission_sidedness = sampled.light.area.sidedness;
        area_sample.area = sampled.light.area.area;
        area_sample.pdf_area = area_sample.area > 0 ? Float(1) / area_sample.area : 0;
        vec3 to_light = area_sample.position - ctx.position;
        Float distance2 = length2(to_light);
        if (distance2 <= kRayEpsilon * kRayEpsilon) return li;
        li.distance = sqrtf(distance2);
        li.wi = to_light / li.distance;
        li.position = area_sample.position;
        li.normal = area_sample.normal;
        li.radiance = eval_emission(
            area_sample.emission, area_sample.emission_sidedness, area_sample.normal, -li.wi);
        Float solid_pdf = pdf_solid_angle(area_sample, ctx.position);
        li.pdf = sampled.pmf * solid_pdf;
        li.pdf_measure = PdfMeasure::SolidAngle;
        li.light_index = sampled.light_index;
        li.primitive_id = area_sample.triangle_id;
        li.valid = max_component(li.radiance) > 0 && li.pdf > 0;
        return li;
    }

    li = sample_light_li(sampled.light, ctx, light_component_u, light_u);
    li.pdf *= sampled.pmf;
    li.light_index = sampled.light_index;
    return li;
}

LuxInline vec3 infinite_lights_le(const SceneLightSampler& sampler, const Ray& ray) {
    if (sampler.light_sampling == nullptr) return vec3(0);
    scene_light_sampler_detail::require_current(sampler);
    vec3 Le(0);
    const LightSelectionTable& selection = sampler.light_sampling->light_selection;
    for (const Light& light : selection.lights) {
        Le += infinite_light_le(light, ray);
    }
    return Le;
}

LuxInline Float infinite_lights_pdf_li(const SceneLightSampler& sampler, const vec3& wi) {
    if (sampler.light_sampling == nullptr) return 0;
    scene_light_sampler_detail::require_current(sampler);
    const LightSelectionTable& selection = sampler.light_sampling->light_selection;
    Float pdf = 0;
    LightSampleContext ctx;
    for (int light_index = 0; light_index < static_cast<int>(selection.lights.size());
         ++light_index) {
        const Light& light = selection.lights[light_index];
        if (light_type(light) != LightType::Infinite) continue;
        pdf += light_pmf(sampler, ctx, light_index) * light_pdf_li(light, ctx, wi);
    }
    return pdf;
}

LuxInline vec3 diffuse_area_light_emission_value(const Scene& scene,
                                                 const DiffuseAreaLight& light,
                                                 const vec2& uv) {
    if (light.emission_source == EmissionSourceKind::Image) {
        if (light.emission_image_id < 0
            || light.emission_image_id >= static_cast<int>(scene.image_light_assets.size())) {
            return vec3(0);
        }
        return image_asset_lookup_nearest(
            scene.image_light_assets[light.emission_image_id],
            uv, light.emission_scale);
    }
    return light.emission;
}

LuxInline bool area_light_matches(const DiffuseAreaLight& light,
                                  const PrimitiveRef& primitive_ref,
                                  int triangle_id) {
    if (primitive_ref_valid(light.primitive_ref) && primitive_ref_valid(primitive_ref)) {
        return same_primitive_ref(light.primitive_ref, primitive_ref);
    }
    return light.primitive_id == triangle_id;
}

LuxInline vec3 surface_light_le(const SceneLightSampler& sampler,
                                PrimitiveRef primitive_ref,
                                int triangle_id,
                                const vec2& uv,
                                const vec3& ng,
                                const vec3& wo) {
    if (sampler.scene == nullptr || sampler.light_sampling == nullptr) return vec3(0);
    scene_light_sampler_detail::require_current(sampler);
    const Scene& scene = *sampler.scene;
    const LightSelectionTable& selection = sampler.light_sampling->light_selection;
    const SurfaceLightLookupTable& lookup = sampler.light_sampling->surface_lookup;

    if (triangle_id >= 0
        && triangle_id < static_cast<int>(lookup.primitive_light_indices.size())) {
        int mapped_light_index = lookup.primitive_light_indices[triangle_id];
        if (mapped_light_index >= 0
            && mapped_light_index < static_cast<int>(selection.lights.size())) {
            const Light& light = selection.lights[mapped_light_index];
            if (light.kind == LightKind::DiffuseArea
                && area_light_matches(light.area, primitive_ref, triangle_id)) {
                return eval_emission(
                    diffuse_area_light_emission_value(scene, light.area, uv),
                    light.area.sidedness, ng, wo);
            }
        }
    }

    vec3 Le(0);
    bool found_area_light = false;
    for (const Light& light : selection.lights) {
        if (light.kind != LightKind::DiffuseArea
            || !area_light_matches(light.area, primitive_ref, triangle_id)) {
            continue;
        }
        found_area_light = true;
        Le += eval_emission(
            diffuse_area_light_emission_value(scene, light.area, uv),
            light.area.sidedness, ng, wo);
    }
    if (found_area_light) return Le;

    if (!scene_primitive_ref_exists(scene, primitive_ref)) return vec3(0);
    SceneTriangle triangle = scene_triangle_view(scene, primitive_ref);
    if (triangle.material_id < 0
        || triangle.material_id >= static_cast<int>(scene.materials.size())) {
        return vec3(0);
    }
    const Material& material = scene.materials[triangle.material_id];
    if (!material.is_emissive()) return vec3(0);
    return scene_eval_emission(scene, material, uv, ng, wo);
}

LuxInline vec3 surface_light_le(const SceneLightSampler& sampler,
                                int triangle_id,
                                const vec2& uv,
                                const vec3& ng,
                                const vec3& wo) {
    if (sampler.scene == nullptr) return vec3(0);
    return surface_light_le(
        sampler, scene_primitive_ref(*sampler.scene, triangle_id),
        triangle_id, uv, ng, wo);
}

LuxInline Float area_light_pdf_solid_angle(const SceneLightSampler& sampler,
                                           PrimitiveRef primitive_ref,
                                           int triangle_id,
                                           const vec3& position,
                                           const vec3& reference_position) {
    if (sampler.scene == nullptr || sampler.light_sampling == nullptr) return 0;
    scene_light_sampler_detail::require_current(sampler);
    const Scene& scene = *sampler.scene;
    const PreparedLightSampling& light_sampling = *sampler.light_sampling;
    const LightSelectionTable& selection = light_sampling.light_selection;
    const SurfaceLightLookupTable& lookup = light_sampling.surface_lookup;
    if (!scene_primitive_ref_exists(scene, primitive_ref)) return 0;
    SceneTriangle triangle = scene_triangle_view(scene, primitive_ref);
    if (triangle.material_id < 0
        || triangle.material_id >= static_cast<int>(scene.materials.size())) {
        return 0;
    }
    const Material& material = scene.materials[triangle.material_id];

    if (triangle_id >= 0
        && triangle_id < static_cast<int>(lookup.primitive_light_indices.size())) {
        int mapped_light_index = lookup.primitive_light_indices[triangle_id];
        if (mapped_light_index >= 0
            && mapped_light_index < static_cast<int>(selection.lights.size())) {
            const Light& candidate = selection.lights[mapped_light_index];
            if (candidate.kind == LightKind::DiffuseArea
                && area_light_matches(candidate.area, primitive_ref, triangle_id)
                && candidate.area.area > 0) {
                LightSample light;
                light.triangle_id = triangle_id;
                light.primitive_ref = primitive_ref;
                light.position = position;
                light.normal = triangle_normal(triangle.triangle);
                light.uv = interpolate_triangle_uv(triangle, 0, 0);
                light.emission = candidate.area.emission;
                light.emission_sidedness = candidate.area.sidedness;
                light.area = candidate.area.area;
                light.pdf_area = Float(1) / candidate.area.area;
                LightSampleContext ctx{reference_position, vec3(0), vec3(0)};
                return light_pmf(sampler, ctx, mapped_light_index)
                    * pdf_solid_angle(light, reference_position);
            }
        }
    }

    Float generic_pdf = 0;
    for (int light_index = 0; light_index < static_cast<int>(selection.lights.size());
        ++light_index) {
        const Light& candidate = selection.lights[light_index];
        if (candidate.kind != LightKind::DiffuseArea
            || !area_light_matches(candidate.area, primitive_ref, triangle_id)
            || candidate.area.area <= 0) {
            continue;
        }
        LightSample light;
        light.triangle_id = triangle_id;
        light.primitive_ref = primitive_ref;
        light.position = position;
        light.normal = triangle_normal(triangle.triangle);
        light.uv = interpolate_triangle_uv(triangle, 0, 0);
        light.emission = candidate.area.emission;
        light.emission_sidedness = candidate.area.sidedness;
        light.area = candidate.area.area;
        light.pdf_area = Float(1) / candidate.area.area;
        LightSampleContext ctx{reference_position, vec3(0), vec3(0)};
        generic_pdf += light_pmf(sampler, ctx, light_index)
            * pdf_solid_angle(light, reference_position);
    }
    if (generic_pdf > 0) return generic_pdf;
    if (!material.is_emissive()) return 0;
    const DiscreteSamplingTable& area_distribution =
        light_sampling.area_lights.area_distribution;
    if (area_distribution.total_weight <= 0) return 0;

    LightSample light;
    light.triangle_id = triangle_id;
    light.primitive_ref = primitive_ref;
    light.position = position;
    light.normal = triangle_normal(triangle.triangle);
    light.emission = material.emission;
    light.emission_sidedness = material.emission_sidedness;
    light.area = triangle_area(triangle.triangle);
    light.pdf_area = Float(1) / area_distribution.total_weight;
    return pdf_solid_angle(light, reference_position);
}

LuxInline Float area_light_pdf_solid_angle(const SceneLightSampler& sampler,
                                           int triangle_id,
                                           const vec3& position,
                                           const vec3& reference_position) {
    if (sampler.scene == nullptr) return 0;
    return area_light_pdf_solid_angle(
        sampler, scene_primitive_ref(*sampler.scene, triangle_id), triangle_id,
        position, reference_position);
}

#endif // LUX_SCENE_LIGHT_SAMPLER_H
