#pragma once

#ifndef LUX_INTEGRATOR_WAVEFRONT_DIRECT_LIGHTING_CUH
#define LUX_INTEGRATOR_WAVEFRONT_DIRECT_LIGHTING_CUH

#include "core/constants.h"
#include "core/ray_offset.cuh"
#include "integrator/pathstate.cuh"
#include "light/light_sampler.h"
#include "material/bsdf.h"
#include "sampler/sampler.h"
#include "scene/gpu_scene_light_sampler.cuh"
#include "scene/gpu_scene_view.h"

namespace lux_wavefront_direct {

template <MaterialType MaterialKind>
LuxDeviceInline DirectLightEstimate sample_direct_light(
        GpuScene scene,
        const Material& material,
        const SurfaceInteraction& interaction,
        SamplerState& sampler,
        LightLiSample& light,
        bool& sampled_light) {
    sampled_light = false;
    if (!has_direct_lighting_lobe(material)) {
        return DirectLightEstimate{};
    }

    if (scene.generic_light_count <= 0 || scene.total_light_power <= 0) {
        return DirectLightEstimate{};
    }

    LightSampleContext ctx{interaction.position, interaction.ng, interaction.ns};
    Float light_select_u = sampler_get_1d(sampler);
    Float light_component_u = sampler_get_1d(sampler);
    vec2 light_u = sampler_get_2d(sampler);
    light = sample_light_li(
        scene, ctx, light_select_u, light_component_u, light_u);
    sampled_light = light.valid;
    return sampled_light
        ? estimate_unoccluded_direct_light_li_typed<MaterialKind>(
            material, interaction, light)
        : DirectLightEstimate{};
}

LuxDeviceInline ShadowRayWorkItem make_shadow_work_item(
        int path_id,
        int triangle_id,
        PrimitiveRef primitive_ref,
        int material_id,
        int depth,
        const vec3& position,
        const SurfaceInteraction& interaction,
        const LightSample& light,
        const DirectLightEstimate& direct,
        const vec3& contribution) {
    vec3 shadow_origin = offset_ray_origin_along_normal(
        position, interaction.ng, direct.wi);
    vec3 shadow_dir = light.position - shadow_origin;
    Float shadow_dist = length(shadow_dir);

    return ShadowRayWorkItem{
        path_id,
        triangle_id,
        primitive_ref,
        material_id,
        depth,
        Ray(shadow_origin, shadow_dir / shadow_dist, kRayEpsilon,
            shadow_dist - kRayEpsilon),
        contribution,
        interaction.ng,
        interaction.ns,
        direct.pdf_light_solid_angle,
        direct.pdf_bsdf_solid_angle
    };
}

LuxDeviceInline ShadowRayWorkItem make_shadow_work_item_li(
        int path_id,
        int triangle_id,
        PrimitiveRef primitive_ref,
        int material_id,
        int depth,
        const vec3& position,
        const SurfaceInteraction& interaction,
        const LightLiSample& light,
        const DirectLightEstimate& direct,
        const vec3& contribution) {
    vec3 shadow_origin = offset_ray_origin_along_normal(
        position, interaction.ng, direct.wi);
    vec3 shadow_dir = direct.wi;
    Float shadow_tmax = INFINITY;
    vec3 shadow_contribution = contribution;
    if (light.distance < INFINITY) {
        shadow_dir = light.position - shadow_origin;
        Float shadow_dist = length(shadow_dir);
        if (shadow_dist <= kRayEpsilon) {
            shadow_contribution = vec3(0);
            shadow_dir = direct.wi;
            shadow_tmax = kRayEpsilon;
        } else {
            shadow_dir = shadow_dir / shadow_dist;
            shadow_tmax = shadow_dist - kRayEpsilon;
        }
    }

    return ShadowRayWorkItem{
        path_id,
        triangle_id,
        primitive_ref,
        material_id,
        depth,
        Ray(shadow_origin, shadow_dir, kRayEpsilon, shadow_tmax),
        shadow_contribution,
        interaction.ng,
        interaction.ns,
        direct.pdf_light_solid_angle,
        direct.pdf_bsdf_solid_angle
    };
}

} // namespace lux_wavefront_direct

#endif // LUX_INTEGRATOR_WAVEFRONT_DIRECT_LIGHTING_CUH
