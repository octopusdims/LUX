#pragma once

#ifndef LUX_INTEGRATOR_PATH_TRANSPORT_CUH
#define LUX_INTEGRATOR_PATH_TRANSPORT_CUH

#include "core/ray_offset.cuh"
#include "light/light_sampler.h"
#include "material/bsdf.h"

struct PreviousSample {
    Float pdf = 0;
    PdfMeasure measure = PdfMeasure::None;
    SampleSource source = SampleSource::None;
    unsigned flags = 0;
    vec3 reference_position = vec3(0);
};

LuxHDInline bool previous_sample_is_delta(const PreviousSample& sample) {
    return is_delta_bsdf(sample.flags);
}

LuxHDInline bool previous_sample_is_solid_angle(const PreviousSample& sample) {
    return sample.measure == PdfMeasure::SolidAngle && sample.pdf > 0;
}

LuxHDInline SurfaceInteraction make_surface_interaction(const vec3& position,
                                                        const vec3& geometric_normal,
                                                        vec3 shading_normal,
                                                        const vec3& wo) {
    if (dot(geometric_normal, shading_normal) < 0) {
        shading_normal = -shading_normal;
    }
    return SurfaceInteraction{position, geometric_normal, shading_normal, wo};
}

LuxHDInline bool emission_uses_unweighted_hit(int depth,
                                              const PreviousSample& previous_sample) {
    return depth == 0 || previous_sample_is_delta(previous_sample);
}

LuxHDInline bool emission_uses_mis(const PreviousSample& previous_sample) {
    return previous_sample_is_solid_angle(previous_sample);
}

LuxHDInline Float emission_mis_weight(const PreviousSample& previous_sample,
                                      Float light_pdf) {
    return power_heuristic(previous_sample.pdf, light_pdf);
}

LuxHDInline Float direct_light_mis_weight(const DirectLightEstimate& estimate) {
    if (estimate.light_sample.pdf.measure == PdfMeasure::Discrete) {
        return Float(1);
    }
    return power_heuristic(estimate.pdf_light_solid_angle,
                           estimate.pdf_bsdf_solid_angle);
}

LuxHDInline bool should_apply_russian_roulette(int depth) {
    return depth > 2;
}

LuxHDInline Float bsdf_eta_scale_factor(const BSDFSample& sample) {
    if (!is_transmission_bsdf(sample.flags) || sample.eta <= 0) return Float(1);
    return sample.eta * sample.eta;
}

LuxHDInline Float russian_roulette_probability(const vec3& throughput,
                                               Float eta_scale = Float(1)) {
    return fminf(Float(0.95), max_component(throughput) * eta_scale);
}

LuxHDInline bool survives_russian_roulette(int depth,
                                           const vec3& throughput,
                                           Float eta_scale,
                                           Float random_value,
                                           Float& survival_probability) {
    if (!should_apply_russian_roulette(depth)) {
        survival_probability = Float(1);
        return true;
    }

    survival_probability = russian_roulette_probability(throughput, eta_scale);
    return random_value <= survival_probability;
}

LuxHDInline bool survives_russian_roulette(int depth,
                                           const vec3& throughput,
                                           Float random_value,
                                           Float& survival_probability) {
    return survives_russian_roulette(
        depth, throughput, Float(1), random_value, survival_probability);
}

LuxHDInline vec3 apply_passthrough(const vec3& throughput,
                                   const Material& material) {
    return throughput * passthrough_transmittance(material);
}

LuxHDInline Ray spawn_scatter_ray(const vec3& position,
                                  const vec3& geometric_normal,
                                  const vec3& wi) {
    return Ray(offset_ray_origin_along_normal(position, geometric_normal, wi),
               wi, kRayEpsilon);
}

LuxHDInline Ray spawn_passthrough_ray(const vec3& position, const vec3& wi) {
    return Ray(offset_ray_origin_along_direction(position, wi), wi, kRayEpsilon);
}

#endif // LUX_INTEGRATOR_PATH_TRANSPORT_CUH
