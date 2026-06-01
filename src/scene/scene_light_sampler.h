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
    const LightDistribution* distribution = nullptr;
    std::uint64_t scene_revision = 0;

    bool empty() const;
};

namespace scene_light_sampler_detail {

LuxInline bool close_float(Float a, Float b) {
    Float scale = fmaxf(Float(1), fmaxf(fabsf(a), fabsf(b)));
    return fabsf(a - b) <= Float(1e-5) * scale;
}

LuxInline bool same_distribution(const LightDistribution& actual,
                                 const LightDistribution& expected) {
    if (actual.primitive_refs.size() != expected.primitive_refs.size()
        || actual.primitive_areas.size() != expected.primitive_areas.size()
        || actual.area_cdf.size() != expected.area_cdf.size()
        || actual.lights.size() != expected.lights.size()
        || actual.light_power_cdf.size() != expected.light_power_cdf.size()) {
        return false;
    }
    if (!close_float(actual.total_area, expected.total_area)) return false;
    if (!close_float(actual.total_light_power, expected.total_light_power)) return false;

    for (std::size_t i = 0; i < expected.primitive_refs.size(); ++i) {
        if (!same_primitive_ref(actual.primitive_refs[i], expected.primitive_refs[i])) return false;
        if (!close_float(actual.primitive_areas[i], expected.primitive_areas[i])) return false;
        if (!close_float(actual.area_cdf[i], expected.area_cdf[i])) return false;
    }
    for (std::size_t i = 0; i < expected.lights.size(); ++i) {
        if (actual.lights[i].kind != expected.lights[i].kind) return false;
        if (!close_float(actual.light_power_cdf[i], expected.light_power_cdf[i])) return false;
    }
    return true;
}

LuxInline void require_current(const SceneLightSampler& sampler) {
    if (sampler.scene == nullptr) return;
    if (sampler.scene->revision != sampler.scene_revision) {
        throw std::runtime_error(
            "SceneLightSampler cannot be used after its Scene has been modified");
    }
}

} // namespace scene_light_sampler_detail

LuxInline bool SceneLightSampler::empty() const {
    scene_light_sampler_detail::require_current(*this);
    return distribution == nullptr || distribution->empty();
}

LuxInline SceneLightSampler make_scene_light_sampler(
        const Scene& scene,
        const LightDistribution& distribution) {
    LightDistribution expected = build_scene_light_distribution(scene);
    if (!scene_light_sampler_detail::same_distribution(distribution, expected)) {
        throw std::runtime_error(
            "SceneLightSampler requires a LightDistribution built from the current Scene");
    }
    return SceneLightSampler{&scene, &distribution, scene.revision};
}

LuxInline bool sample_area_light(const SceneLightSampler& sampler,
                                 Float select_u, const vec2& surface_u,
                                 LightSample& sample) {
    if (sampler.scene == nullptr || sampler.distribution == nullptr) return false;
    scene_light_sampler_detail::require_current(sampler);
    const Scene& scene = *sampler.scene;
    const LightDistribution& distribution = *sampler.distribution;
    if (distribution.primitive_refs.empty() || distribution.total_area <= 0) return false;

    int light_index = sample_cdf_index(
        distribution.area_cdf.data(),
        static_cast<int>(distribution.area_cdf.size()),
        distribution.total_area, select_u);
    if (light_index < 0) return false;

    PrimitiveRef light_ref = distribution.primitive_refs[light_index];
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
    sample.area = distribution.primitive_areas[light_index];
    sample.pdf_area = Float(1) / distribution.total_area;
    return sample.area > 0 && max_component(sample.emission) > 0;
}

LuxInline Float light_pmf(const SceneLightSampler& sampler,
                          const LightSampleContext& ctx,
                          int light_index);

LuxInline SampledLight sample_light(const SceneLightSampler& sampler,
                                    const LightSampleContext& ctx,
                                    Float select_u) {
    SampledLight sampled;
    if (sampler.distribution == nullptr) return sampled;
    scene_light_sampler_detail::require_current(sampler);
    const LightDistribution& distribution = *sampler.distribution;
    if (distribution.lights.empty() || distribution.total_light_power <= 0) return sampled;

    int light_index = sample_cdf_index(
        distribution.light_power_cdf.data(),
        static_cast<int>(distribution.light_power_cdf.size()),
        distribution.total_light_power, select_u);
    if (light_index < 0) return sampled;
    sampled.light_index = light_index;
    sampled.light = distribution.lights[light_index];
    sampled.pmf = light_pmf(sampler, ctx, light_index);
    sampled.valid = sampled.pmf > 0;
    return sampled;
}

LuxInline SampledLight sample_light(const SceneLightSampler& sampler, Float select_u) {
    return sample_light(sampler, LightSampleContext{}, select_u);
}

LuxInline Float light_pmf(const SceneLightSampler& sampler,
                          const LightSampleContext& ctx,
                          int light_index) {
    (void)ctx;
    if (sampler.distribution == nullptr) return 0;
    scene_light_sampler_detail::require_current(sampler);
    const LightDistribution& distribution = *sampler.distribution;
    if (light_index < 0 || light_index >= static_cast<int>(distribution.lights.size())
        || distribution.total_light_power <= 0) {
        return 0;
    }
    return cdf_interval_pmf(
        distribution.light_power_cdf.data(),
        static_cast<int>(distribution.light_power_cdf.size()),
        distribution.total_light_power, light_index);
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
    if (sampler.distribution == nullptr) return vec3(0);
    scene_light_sampler_detail::require_current(sampler);
    vec3 Le(0);
    const LightDistribution& distribution = *sampler.distribution;
    for (const Light& light : distribution.lights) {
        Le += infinite_light_le(light, ray);
    }
    return Le;
}

LuxInline Float infinite_lights_pdf_li(const SceneLightSampler& sampler, const vec3& wi) {
    if (sampler.distribution == nullptr) return 0;
    scene_light_sampler_detail::require_current(sampler);
    const LightDistribution& distribution = *sampler.distribution;
    Float pdf = 0;
    LightSampleContext ctx;
    for (int light_index = 0; light_index < static_cast<int>(distribution.lights.size());
         ++light_index) {
        const Light& light = distribution.lights[light_index];
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
    if (sampler.scene == nullptr || sampler.distribution == nullptr) return vec3(0);
    scene_light_sampler_detail::require_current(sampler);
    const Scene& scene = *sampler.scene;
    const LightDistribution& distribution = *sampler.distribution;

    vec3 Le(0);
    bool found_area_light = false;
    for (const Light& light : distribution.lights) {
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
    if (sampler.scene == nullptr || sampler.distribution == nullptr) return 0;
    scene_light_sampler_detail::require_current(sampler);
    const Scene& scene = *sampler.scene;
    const LightDistribution& distribution = *sampler.distribution;
    if (!scene_primitive_ref_exists(scene, primitive_ref)) return 0;
    SceneTriangle triangle = scene_triangle_view(scene, primitive_ref);
    if (triangle.material_id < 0
        || triangle.material_id >= static_cast<int>(scene.materials.size())) {
        return 0;
    }
    const Material& material = scene.materials[triangle.material_id];

    Float generic_pdf = 0;
    for (int light_index = 0; light_index < static_cast<int>(distribution.lights.size());
        ++light_index) {
        const Light& candidate = distribution.lights[light_index];
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
    if (distribution.total_area <= 0) return 0;

    LightSample light;
    light.triangle_id = triangle_id;
    light.primitive_ref = primitive_ref;
    light.position = position;
    light.normal = triangle_normal(triangle.triangle);
    light.emission = material.emission;
    light.emission_sidedness = material.emission_sidedness;
    light.area = triangle_area(triangle.triangle);
    light.pdf_area = Float(1) / distribution.total_area;
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
