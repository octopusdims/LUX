#pragma once

#ifndef LUX_SCENE_GPU_LIGHT_SAMPLER_CUH
#define LUX_SCENE_GPU_LIGHT_SAMPLER_CUH

#include "light/light_sampler.h"
#include "scene/gpu_scene_view.h"

LuxDeviceInline Float light_pmf(GpuScene scene,
                                const LightSampleContext& ctx,
                                int light_index) {
    (void)ctx;
    if (light_index < 0 || light_index >= scene.generic_light_count
        || scene.total_light_power <= 0) {
        return 0;
    }
    return cdf_interval_pmf(
        scene.light_power_cdf, scene.generic_light_count,
        scene.total_light_power, light_index);
}

LuxDeviceInline Float light_pmf(GpuScene scene, int light_index) {
    return light_pmf(scene, LightSampleContext{}, light_index);
}

LuxDeviceInline SampledLight sample_scene_light(GpuScene scene,
                                                const LightSampleContext& ctx,
                                                Float select_u) {
    SampledLight sampled;
    if (scene.generic_light_count <= 0 || scene.total_light_power <= 0) return sampled;

    int light_index = sample_cdf_index(
        scene.light_power_cdf, scene.generic_light_count,
        scene.total_light_power, select_u);
    if (light_index < 0) return sampled;
    sampled.light_index = light_index;
    sampled.light = scene.lights[light_index];
    sampled.pmf = light_pmf(scene, ctx, light_index);
    sampled.valid = sampled.pmf > 0;
    return sampled;
}

LuxDeviceInline SampledLight sample_scene_light(GpuScene scene, Float select_u) {
    return sample_scene_light(scene, LightSampleContext{}, select_u);
}

LuxDeviceInline vec3 gpu_diffuse_area_light_emission_value(
    GpuScene scene, const DiffuseAreaLight& light, const vec2& uv);

LuxDeviceInline bool sample_area_light(GpuScene scene, Float select_u,
                                       const vec2& surface_u,
                                       LightSample& sample) {
    if (scene.light_count <= 0 || scene.total_light_area <= 0) return false;

    int light_index = sample_cdf_index(
        scene.light_area_cdf, scene.light_count,
        scene.total_light_area, select_u);
    if (light_index < 0) return false;

    PrimitiveRef light_ref = scene.light_primitive_refs[light_index];
    GpuSceneTriangle light_triangle = gpu_scene_triangle(scene, light_ref);
    int material_id = light_triangle.material_id;
    if (!light_triangle.valid || material_id < 0 || material_id >= scene.material_count) {
        return false;
    }

    const Material& material = scene.materials[material_id];
    sample.triangle_id = gpu_scene_primitive_index(scene, light_ref);
    sample.primitive_ref = light_ref;
    sample.position = sample_triangle_area(light_triangle.triangle, surface_u);
    sample.normal = triangle_normal(light_triangle.triangle);
    vec2 bary_uv = sample_triangle_barycentric_uv(surface_u);
    sample.uv = gpu_scene_interpolate_triangle_uv(light_triangle, bary_uv.x, bary_uv.y);
    sample.emission = gpu_material_emission_value(scene, material, sample.uv);
    sample.emission_sidedness = material.emission_sidedness;
    sample.area = scene.light_primitive_areas[light_index];
    sample.pdf_area = Float(1) / scene.total_light_area;
    return sample.area > 0 && max_component(sample.emission) > 0;
}

LuxDeviceInline LightLiSample sample_light_li(GpuScene scene,
                                              const LightSampleContext& ctx,
                                              Float light_select_u,
                                              Float light_component_u,
                                              const vec2& light_u) {
    SampledLight sampled = sample_scene_light(scene, ctx, light_select_u);
    LightLiSample li;
    if (!sampled.valid) return li;

    if (sampled.light.kind == LightKind::DiffuseArea) {
        GpuSceneTriangle light_triangle = gpu_scene_triangle(
            scene, sampled.light.area.primitive_ref);
        if (!light_triangle.valid) return li;
        int light_triangle_id = sampled.light.area.primitive_id >= 0
            ? sampled.light.area.primitive_id
            : gpu_scene_primitive_index(scene, sampled.light.area.primitive_ref);
        LightSample area_sample;
        area_sample.triangle_id = light_triangle_id;
        area_sample.primitive_ref = sampled.light.area.primitive_ref;
        area_sample.position = sample_triangle_area(light_triangle.triangle, light_u);
        area_sample.normal = triangle_normal(light_triangle.triangle);
        vec2 bary_uv = sample_triangle_barycentric_uv(light_u);
        area_sample.uv = gpu_scene_interpolate_triangle_uv(
            light_triangle, bary_uv.x, bary_uv.y);
        area_sample.emission = gpu_diffuse_area_light_emission_value(
            scene, sampled.light.area, area_sample.uv);
        area_sample.emission_sidedness = sampled.light.area.sidedness;
        area_sample.area = sampled.light.area.area > 0
            ? sampled.light.area.area
            : triangle_area(light_triangle.triangle);
        area_sample.pdf_area = area_sample.area > 0 ? Float(1) / area_sample.area : 0;

        vec3 to_light = area_sample.position - ctx.position;
        Float distance2 = length2(to_light);
        if (distance2 <= kRayEpsilon * kRayEpsilon) return li;
        li.distance = sqrtf(distance2);
        li.wi = to_light / li.distance;
        li.position = area_sample.position;
        li.normal = area_sample.normal;
        li.radiance = eval_emission(
            area_sample.emission, area_sample.emission_sidedness,
            area_sample.normal, -li.wi);
        li.pdf = sampled.pmf * pdf_solid_angle(area_sample, ctx.position);
        li.pdf_measure = PdfMeasure::SolidAngle;
        li.light_index = sampled.light_index;
        li.primitive_id = light_triangle_id;
        li.valid = max_component(li.radiance) > 0 && li.pdf > 0;
        return li;
    }

    li = sample_light_li(sampled.light, ctx, light_component_u, light_u);
    li.pdf *= sampled.pmf;
    li.light_index = sampled.light_index;
    return li;
}

LuxDeviceInline vec3 infinite_lights_le(GpuScene scene, const Ray& ray) {
    vec3 Le(0);
    for (int i = 0; i < scene.generic_light_count; ++i) {
        Le += infinite_light_le(scene.lights[i], ray);
    }
    return Le;
}

LuxDeviceInline Float infinite_lights_pdf_li(GpuScene scene, const vec3& wi) {
    Float pdf = 0;
    LightSampleContext ctx;
    for (int light_index = 0; light_index < scene.generic_light_count; ++light_index) {
        const Light& light = scene.lights[light_index];
        if (light_type(light) != LightType::Infinite) continue;
        pdf += light_pmf(scene, ctx, light_index) * light_pdf_li(light, ctx, wi);
    }
    return pdf;
}

LuxDeviceInline vec3 gpu_diffuse_area_light_emission_value(GpuScene scene,
                                                           const DiffuseAreaLight& light,
                                                           const vec2& uv) {
    if (light.emission_source == EmissionSourceKind::Image) {
        return gpu_image_asset_lookup_nearest(
            scene, light.emission_image_id, uv, light.emission_scale);
    }
    return light.emission;
}

LuxDeviceInline bool gpu_area_light_matches(const DiffuseAreaLight& light,
                                            PrimitiveRef ref,
                                            int triangle_id) {
    if (primitive_ref_valid(light.primitive_ref) && primitive_ref_valid(ref)) {
        return same_primitive_ref(light.primitive_ref, ref);
    }
    return light.primitive_id == triangle_id;
}

LuxDeviceInline vec3 surface_light_le(GpuScene scene,
                                      PrimitiveRef primitive_ref,
                                      int triangle_id,
                                      const vec2& uv, const vec3& ng,
                                      const vec3& wo) {
    vec3 Le(0);
    bool found_area_light = false;
    for (int light_index = 0; light_index < scene.generic_light_count; ++light_index) {
        const Light& light = scene.lights[light_index];
        if (light.kind != LightKind::DiffuseArea
            || !gpu_area_light_matches(light.area, primitive_ref, triangle_id)) {
            continue;
        }
        found_area_light = true;
        Le += eval_emission(
            gpu_diffuse_area_light_emission_value(scene, light.area, uv),
            light.area.sidedness, ng, wo);
    }
    if (found_area_light) return Le;

    GpuSceneTriangle triangle = gpu_scene_triangle(scene, primitive_ref);
    int material_id = triangle.material_id;
    if (material_id < 0 || material_id >= scene.material_count) return vec3(0);
    const Material& material = scene.materials[material_id];
    if (!material.is_emissive()) return vec3(0);
    return gpu_eval_emission(scene, material, uv, ng, wo);
}

LuxDeviceInline vec3 surface_light_le(GpuScene scene, int triangle_id,
                                      const vec2& uv, const vec3& ng,
                                      const vec3& wo) {
    return surface_light_le(
        scene, gpu_scene_primitive_ref(scene, triangle_id), triangle_id, uv, ng, wo);
}

LuxDeviceInline Float area_light_pdf_solid_angle(GpuScene scene,
                                                 PrimitiveRef primitive_ref,
                                                 int triangle_id,
                                                 const vec3& position,
                                                 const vec3& reference_position) {
    GpuSceneTriangle triangle = gpu_scene_triangle(scene, primitive_ref);
    int material_id = triangle.material_id;
    if (material_id < 0 || material_id >= scene.material_count) return 0;
    const Material& material = scene.materials[material_id];

    Float generic_pdf = 0;
    for (int light_index = 0; light_index < scene.generic_light_count; ++light_index) {
        const Light& candidate = scene.lights[light_index];
        if (candidate.kind != LightKind::DiffuseArea
            || !gpu_area_light_matches(candidate.area, primitive_ref, triangle_id)
            || candidate.area.area <= 0) {
            continue;
        }
        LightSample light;
        light.triangle_id = triangle_id;
        light.primitive_ref = primitive_ref;
        light.position = position;
        light.normal = triangle_normal(triangle.triangle);
        light.emission = candidate.area.emission;
        light.emission_sidedness = candidate.area.sidedness;
        light.area = candidate.area.area;
        light.pdf_area = Float(1) / candidate.area.area;
        LightSampleContext ctx{reference_position, vec3(0), vec3(0)};
        generic_pdf += light_pmf(scene, ctx, light_index)
            * pdf_solid_angle(light, reference_position);
    }
    if (generic_pdf > 0) return generic_pdf;
    if (!material.is_emissive()) return 0;
    if (scene.total_light_area <= 0) return 0;

    LightSample light;
    light.triangle_id = triangle_id;
    light.primitive_ref = primitive_ref;
    light.position = position;
    light.normal = triangle_normal(triangle.triangle);
    light.emission = material.emission;
    light.emission_sidedness = material.emission_sidedness;
    light.area = triangle_area(triangle.triangle);
    light.pdf_area = Float(1) / scene.total_light_area;
    return pdf_solid_angle(light, reference_position);
}

LuxDeviceInline Float area_light_pdf_solid_angle(GpuScene scene, int triangle_id,
                                                 const vec3& position,
                                                 const vec3& reference_position) {
    return area_light_pdf_solid_angle(
        scene, gpu_scene_primitive_ref(scene, triangle_id), triangle_id,
        position, reference_position);
}

#endif // LUX_SCENE_GPU_LIGHT_SAMPLER_CUH
