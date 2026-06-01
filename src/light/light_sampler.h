#pragma once

#ifndef LUX_LIGHT_SAMPLER_H
#define LUX_LIGHT_SAMPLER_H

#include "core/constants.h"
#include "core/vec2.cuh"
#include "geometry/triangle_mesh_view.h"
#include "light/light.h"
#include "material/bsdf.h"
#include "sampler/sample_record.h"
#include "scene/primitive_ref.h"

struct LightSample {
    int triangle_id = -1;
    PrimitiveRef primitive_ref;
    vec3 position = vec3(0);
    vec3 normal = vec3(0);
    vec2 uv = vec2(0);
    // Raw material emission. Use eval_emission() for directional contribution.
    vec3 emission = vec3(0);
    EmissionSidedness emission_sidedness = EmissionSidedness::SingleSided;
    Float pdf_area = 0;
    Float area = 0;
};

struct DirectLightEstimate {
    vec3 wi = vec3(0);
    vec3 contribution = vec3(0);
    Float distance2 = 0;
    Float cos_surface = 0;
    Float cos_light = 0;
    Float pdf_light_area = 0;
    Float pdf_light_solid_angle = 0;
    Float pdf_bsdf_solid_angle = 0;
    SampleRecord light_sample;
    bool valid = false;
};

LuxHDInline Float clamp_cdf_sample(Float u) {
    return fminf(u, Float(0.99999994));
}

LuxHDInline int sample_cdf_index(const Float* cdf, int count,
                                 Float total_weight, Float u) {
    if (cdf == nullptr || count <= 0 || total_weight <= 0) return -1;
    Float target = clamp_cdf_sample(u) * total_weight;
    int index = count - 1;
    for (int i = 0; i < count; ++i) {
        if (target <= cdf[i]) {
            index = i;
            break;
        }
    }
    return index;
}

LuxHDInline Float cdf_interval_weight(const Float* cdf, int count, int index) {
    if (cdf == nullptr || index < 0 || index >= count) return 0;
    Float prev = index == 0 ? 0 : cdf[index - 1];
    return cdf[index] - prev;
}

LuxHDInline Float cdf_interval_pmf(const Float* cdf, int count,
                                   Float total_weight, int index) {
    if (total_weight <= 0) return 0;
    return cdf_interval_weight(cdf, count, index) / total_weight;
}

LuxHDInline vec3 sample_triangle_area(const Triangle& triangle, const vec2& u) {
    Float sqrt_u0 = sqrtf(u.x);
    Float b0 = Float(1) - sqrt_u0;
    Float b1 = sqrt_u0 * (Float(1) - u.y);
    Float b2 = sqrt_u0 * u.y;
    return b0 * triangle.v0 + b1 * triangle.v1 + b2 * triangle.v2;
}

LuxHDInline vec2 sample_triangle_barycentric_uv(const vec2& u) {
    Float sqrt_u0 = sqrtf(u.x);
    Float b1 = sqrt_u0 * (Float(1) - u.y);
    Float b2 = sqrt_u0 * u.y;
    return vec2(b1, b2);
}

LuxHDInline Float pdf_solid_angle(const LightSample& light, const vec3& reference_position) {
    vec3 to_light = light.position - reference_position;
    Float distance2 = length2(to_light);
    if (distance2 <= kRayEpsilon * kRayEpsilon || light.pdf_area <= 0) return 0;

    vec3 wi = to_light / sqrtf(distance2);
    Float cos_light = emission_projected_cosine(
        light.normal, -wi, light.emission_sidedness);
    if (cos_light <= 0) return 0;
    return light.pdf_area * distance2 / cos_light;
}

LuxHDInline DirectLightEstimate estimate_unoccluded_direct_light(
        const Material& material,
        const SurfaceInteraction& interaction,
        const LightSample& light) {
    DirectLightEstimate estimate;
    vec3 to_light = light.position - interaction.position;
    estimate.distance2 = length2(to_light);
    if (estimate.distance2 <= kRayEpsilon * kRayEpsilon || light.pdf_area <= 0) {
        return estimate;
    }

    estimate.wi = to_light / sqrtf(estimate.distance2);
    vec3 f = eval_bsdf(material, interaction, estimate.wi);
    vec3 Le = eval_emission(
        light.emission, light.emission_sidedness, light.normal, -estimate.wi);
    estimate.cos_surface = abs_cosine(interaction.ns, estimate.wi);
    estimate.cos_light = emission_projected_cosine(
        light.normal, -estimate.wi, light.emission_sidedness);
    estimate.pdf_light_area = light.pdf_area;
    estimate.pdf_light_solid_angle = pdf_solid_angle(light, interaction.position);
    estimate.pdf_bsdf_solid_angle = pdf_bsdf(material, interaction, estimate.wi);
    estimate.light_sample = SampleRecord{
        SampleSource::Light,
        solid_angle_pdf(estimate.pdf_light_solid_angle),
        0
    };
    estimate.valid = max_component(f) > 0
        && max_component(Le) > 0
        && estimate.cos_surface > 0
        && estimate.cos_light > 0
        && estimate.pdf_light_solid_angle > 0;
    if (!estimate.valid) return estimate;

    estimate.contribution = f * Le
        * (estimate.cos_surface / estimate.pdf_light_solid_angle);
    return estimate;
}

LuxHDInline DirectLightEstimate estimate_unoccluded_direct_light_li(
        const Material& material,
        const SurfaceInteraction& interaction,
        const LightLiSample& light) {
    DirectLightEstimate estimate;
    if (!light.valid || light.pdf <= 0) return estimate;
    estimate.wi = light.wi;
    estimate.distance2 = light.distance < INFINITY
        ? light.distance * light.distance
        : INFINITY;
    vec3 f = eval_bsdf(material, interaction, estimate.wi);
    estimate.cos_surface = abs_cosine(interaction.ns, estimate.wi);
    estimate.cos_light = 1;
    estimate.pdf_light_area = 0;
    estimate.pdf_light_solid_angle = light.pdf;
    estimate.pdf_bsdf_solid_angle = pdf_bsdf(material, interaction, estimate.wi);
    estimate.light_sample = SampleRecord{
        SampleSource::Light,
        light.pdf_measure == PdfMeasure::Discrete ? discrete_pdf(light.pdf)
                                                  : solid_angle_pdf(light.pdf),
        0
    };
    estimate.valid = max_component(f) > 0
        && max_component(light.radiance) > 0
        && estimate.cos_surface > 0
        && estimate.pdf_light_solid_angle > 0;
    if (!estimate.valid) return estimate;
    estimate.contribution = f * light.radiance
        * (estimate.cos_surface / estimate.pdf_light_solid_angle);
    return estimate;
}

template <MaterialType Type>
LuxHDInline DirectLightEstimate estimate_unoccluded_direct_light_li_typed(
        const Material& material,
        const SurfaceInteraction& interaction,
        const LightLiSample& light) {
    DirectLightEstimate estimate;
    if (!light.valid || light.pdf <= 0) return estimate;
    estimate.wi = light.wi;
    estimate.distance2 = light.distance < INFINITY
        ? light.distance * light.distance
        : INFINITY;
    vec3 f = eval_bsdf_typed<Type>(material, interaction, estimate.wi);
    estimate.cos_surface = abs_cosine(interaction.ns, estimate.wi);
    estimate.cos_light = 1;
    estimate.pdf_light_area = 0;
    estimate.pdf_light_solid_angle = light.pdf;
    estimate.pdf_bsdf_solid_angle = pdf_bsdf_typed<Type>(material, interaction, estimate.wi);
    estimate.light_sample = SampleRecord{
        SampleSource::Light,
        light.pdf_measure == PdfMeasure::Discrete ? discrete_pdf(light.pdf)
                                                  : solid_angle_pdf(light.pdf),
        0
    };
    estimate.valid = max_component(f) > 0
        && max_component(light.radiance) > 0
        && estimate.cos_surface > 0
        && estimate.pdf_light_solid_angle > 0;
    if (!estimate.valid) return estimate;
    estimate.contribution = f * light.radiance
        * (estimate.cos_surface / estimate.pdf_light_solid_angle);
    return estimate;
}

template <MaterialType Type>
LuxHDInline DirectLightEstimate estimate_unoccluded_direct_light_typed(
        const Material& material,
        const SurfaceInteraction& interaction,
        const LightSample& light) {
    DirectLightEstimate estimate;
    vec3 to_light = light.position - interaction.position;
    estimate.distance2 = length2(to_light);
    if (estimate.distance2 <= kRayEpsilon * kRayEpsilon || light.pdf_area <= 0) {
        return estimate;
    }

    estimate.wi = to_light / sqrtf(estimate.distance2);
    vec3 f = eval_bsdf_typed<Type>(material, interaction, estimate.wi);
    vec3 Le = eval_emission(
        light.emission, light.emission_sidedness, light.normal, -estimate.wi);
    estimate.cos_surface = abs_cosine(interaction.ns, estimate.wi);
    estimate.cos_light = emission_projected_cosine(
        light.normal, -estimate.wi, light.emission_sidedness);
    estimate.pdf_light_area = light.pdf_area;
    estimate.pdf_light_solid_angle = pdf_solid_angle(light, interaction.position);
    estimate.pdf_bsdf_solid_angle = pdf_bsdf_typed<Type>(material, interaction, estimate.wi);
    estimate.light_sample = SampleRecord{
        SampleSource::Light,
        solid_angle_pdf(estimate.pdf_light_solid_angle),
        0
    };
    estimate.valid = max_component(f) > 0
        && max_component(Le) > 0
        && estimate.cos_surface > 0
        && estimate.cos_light > 0
        && estimate.pdf_light_solid_angle > 0;
    if (!estimate.valid) return estimate;

    estimate.contribution = f * Le
        * (estimate.cos_surface / estimate.pdf_light_solid_angle);
    return estimate;
}

#endif // LUX_LIGHT_SAMPLER_H
