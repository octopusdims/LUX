#pragma once

#ifndef LUX_MATERIAL_BSDF_H
#define LUX_MATERIAL_BSDF_H

#include "core/constants.h"
#include "core/vec2.cuh"
#include "material.h"
#include "sampler/sample_record.h"

struct SurfaceInteraction {
    vec3 position;
    vec3 ng;
    vec3 ns;
    vec3 wo;
};

struct BSDFSample {
    vec3 wi = vec3(0);
    vec3 f = vec3(0);       // finite-density BSDF value; delta lobes keep this zero
    vec3 weight = vec3(0);  // throughput multiplier for the sampled direction
    Float pdf = 0;
    PdfMeasure pdf_measure = PdfMeasure::None;
    unsigned flags = 0;
};

enum BsdfFlags : unsigned {
    BsdfFlagNone = 0,
    BsdfFlagDiffuse = 1u << 0,
    BsdfFlagGlossy = 1u << 1,
    BsdfFlagDeltaReflection = 1u << 2,
    BsdfFlagDeltaTransmission = 1u << 3,
    BsdfFlagTransmission = 1u << 4,
};

LuxHDInline bool has_bsdf_flag(unsigned flags, BsdfFlags flag) {
    return (flags & static_cast<unsigned>(flag)) != 0;
}

LuxHDInline bool is_delta_bsdf(unsigned flags) {
    return has_bsdf_flag(flags, BsdfFlagDeltaReflection)
        || has_bsdf_flag(flags, BsdfFlagDeltaTransmission);
}

LuxHDInline bool is_transmission_bsdf(unsigned flags) {
    return has_bsdf_flag(flags, BsdfFlagTransmission)
        || has_bsdf_flag(flags, BsdfFlagDeltaTransmission);
}

LuxHDInline vec3 make_tangent(const vec3& normal) {
    return fabsf(normal.x) > 0.9f
        ? normalize(cross(vec3(0, 0, 1), normal))
        : normalize(cross(vec3(1, 0, 0), normal));
}

LuxHDInline vec3 cosine_sample_hemisphere(const vec3& normal, const vec2& u) {
    Float radius = sqrtf(u.x);
    Float phi = 2.0f * kPi * u.y;
    Float x = radius * cosf(phi);
    Float z = radius * sinf(phi);
    Float y = sqrtf(fmaxf(Float(0), 1.0f - u.x));

    vec3 tangent = make_tangent(normal);
    vec3 bitangent = cross(normal, tangent);
    return normalize(x * tangent + y * normal + z * bitangent);
}

LuxHDInline bool same_hemisphere(const vec3& normal, const vec3& a, const vec3& b) {
    return dot(normal, a) * dot(normal, b) > 0;
}

LuxHDInline Float abs_cosine(const vec3& normal, const vec3& direction) {
    return fabsf(dot(normal, direction));
}

LuxHDInline void finalize_smooth_bsdf_sample(BSDFSample& sample,
                                             const SurfaceInteraction& interaction) {
    if (sample.pdf > 0) {
        sample.pdf_measure = PdfMeasure::SolidAngle;
    }

    Float cos_theta = abs_cosine(interaction.ns, sample.wi);
    if (sample.pdf > 0 && cos_theta > 0 && max_component(sample.f) > 0) {
        sample.weight = sample.f * (cos_theta / sample.pdf);
    }
}

LuxHDInline vec3 orient_to_same_hemisphere(const vec3& normal, const vec3& reference) {
    return dot(normal, reference) < 0 ? -normal : normal;
}

LuxHDInline vec3 eval_lambert(const Material& material,
                               const SurfaceInteraction& interaction,
                               const vec3& wi) {
    if (!same_hemisphere(interaction.ns, interaction.wo, wi)) {
        return vec3(0);
    }
    return material.albedo * kInvPi;
}

LuxHDInline vec3 eval_thin_lambert(const Material& material,
                                    const SurfaceInteraction& interaction,
                                    const vec3& wi) {
    return material.albedo * (Float(0.5) * kInvPi);
}

LuxHDInline Float pdf_lambert(const Material& material,
                              const SurfaceInteraction& interaction,
                              const vec3& wi) {
    if (!same_hemisphere(interaction.ns, interaction.wo, wi)) {
        return 0;
    }
    return abs_cosine(interaction.ns, wi) * kInvPi;
}

LuxHDInline Float pdf_thin_lambert(const Material& material,
                                   const SurfaceInteraction& interaction,
                                   const vec3& wi) {
    return Float(0.5) * abs_cosine(interaction.ns, wi) * kInvPi;
}

LuxHDInline BSDFSample sample_lambert(const Material& material,
                                       const SurfaceInteraction& interaction,
                                       const vec2& u) {
    BSDFSample sample;
    vec3 normal = orient_to_same_hemisphere(interaction.ns, interaction.wo);
    sample.wi = cosine_sample_hemisphere(normal, u);
    sample.f = eval_lambert(material, interaction, sample.wi);
    sample.pdf = pdf_lambert(material, interaction, sample.wi);
    sample.flags = BsdfFlagDiffuse;
    finalize_smooth_bsdf_sample(sample, interaction);
    return sample;
}

LuxHDInline BSDFSample sample_thin_lambert(const Material& material,
                                           const SurfaceInteraction& interaction,
                                           const vec2& u) {
    BSDFSample sample;
    bool sample_reflection = u.x < Float(0.5);
    Float remapped_x = sample_reflection ? u.x * Float(2)
                                         : (u.x - Float(0.5)) * Float(2);
    vec3 wo_side_normal = orient_to_same_hemisphere(interaction.ns, interaction.wo);
    vec3 normal = sample_reflection ? wo_side_normal : -wo_side_normal;
    sample.wi = cosine_sample_hemisphere(normal, vec2(remapped_x, u.y));
    sample.f = eval_thin_lambert(material, interaction, sample.wi);
    sample.pdf = pdf_thin_lambert(material, interaction, sample.wi);
    sample.flags = sample_reflection ? BsdfFlagDiffuse
                                     : (BsdfFlagDiffuse | BsdfFlagTransmission);
    finalize_smooth_bsdf_sample(sample, interaction);
    return sample;
}

// ---------------------------------------------------------------------------
// GGX (Trowbridge-Reitz) microfacet BRDF
// ---------------------------------------------------------------------------

LuxHDInline vec3 mix(const vec3& a, const vec3& b, Float t) {
    return a * (1.0f - t) + b * t;
}

LuxHDInline Float clamp01(Float value) {
    return fmaxf(Float(0), fminf(Float(1), value));
}

LuxHDInline Float ggx_metallic(const Material& material) {
    return clamp01(material.metallic);
}

LuxHDInline vec3 ggx_F0(const Material& material) {
    return mix(vec3(0.04f), material.albedo, ggx_metallic(material));
}

LuxHDInline Float ggx_diffuse_scale(const Material& material) {
    vec3 F0 = ggx_F0(material);
    return (Float(1) - ggx_metallic(material))
         * fmaxf(Float(0), Float(1) - max_component(F0));
}

LuxHDInline Float ggx_alpha(const Material& material) {
    // Remap roughness to alpha for better perceptual linearity, with a small
    // floor to keep D/PDF finite for near-mirror roughness values.
    Float roughness = fmaxf(material.roughness, Float(0.02));
    return fmaxf(roughness * roughness, Float(1e-4));
}

LuxHDInline Float ggx_D(const vec3& h, const vec3& n, Float alpha) {
    Float cos_theta = dot(n, h);
    if (cos_theta <= 0) return 0;
    Float alpha2 = alpha * alpha;
    Float denom = (cos_theta * cos_theta) * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (kPi * denom * denom);
}

LuxHDInline Float ggx_G1(const vec3& v, const vec3& n, Float alpha) {
    Float cos_v = dot(n, v);
    if (cos_v <= 0) return 0;
    Float alpha2 = alpha * alpha;
    Float tan2 = (1.0f - cos_v * cos_v) / (cos_v * cos_v);
    return 2.0f / (1.0f + sqrtf(1.0f + alpha2 * tan2));
}

LuxHDInline Float ggx_Lambda(const vec3& v, const vec3& n, Float alpha) {
    Float cos_v = dot(n, v);
    if (cos_v <= 0) return 0;
    Float cos2 = cos_v * cos_v;
    Float tan2 = fmaxf(Float(0), Float(1) - cos2) / cos2;
    return (sqrtf(Float(1) + alpha * alpha * tan2) - Float(1)) * Float(0.5);
}

LuxHDInline Float ggx_G(const vec3& wo, const vec3& wi, const vec3& n, Float alpha) {
    return Float(1) / (Float(1) + ggx_Lambda(wo, n, alpha) + ggx_Lambda(wi, n, alpha));
}

LuxHDInline vec3 fresnel_schlick(const vec3& F0, Float cos_theta) {
    cos_theta = clamp01(cos_theta);
    Float pow5 = powf(1.0f - cos_theta, 5.0f);
    return F0 + (vec3(1.0f) - F0) * pow5;
}

LuxHDInline vec3 eval_ggx_diffuse(const Material& material,
                                   const SurfaceInteraction& interaction,
                                   const vec3& wi) {
    vec3 wo = interaction.wo;
    vec3 n = interaction.ns;
    if (dot(n, wo) <= 0 || dot(n, wi) <= 0) return vec3(0);
    return material.albedo * (ggx_diffuse_scale(material) * kInvPi);
}

LuxHDInline vec3 eval_ggx_specular(const Material& material,
                                    const SurfaceInteraction& interaction,
                                    const vec3& wi) {
    vec3 wo = interaction.wo;
    vec3 n = interaction.ns;
    if (dot(n, wo) <= 0 || dot(n, wi) <= 0) return vec3(0);

    vec3 h = normalize(wo + wi);
    Float alpha = ggx_alpha(material);
    Float D = ggx_D(h, n, alpha);
    Float G = ggx_G(wo, wi, n, alpha);
    vec3 F = fresnel_schlick(ggx_F0(material), dot(wo, h));

    Float denom = 4.0f * dot(n, wo) * dot(n, wi);
    if (denom <= 0) return vec3(0);
    return (D * G * F) / denom;
}

LuxHDInline Float pdf_ggx_specular(const Material& material,
                                    const SurfaceInteraction& interaction,
                                    const vec3& wi) {
    vec3 wo = interaction.wo;
    vec3 n = interaction.ns;
    if (dot(n, wo) <= 0 || dot(n, wi) <= 0) return 0;

    vec3 h = normalize(wo + wi);
    Float alpha = ggx_alpha(material);
    Float D = ggx_D(h, n, alpha);
    Float abs_cos_wo = fabsf(dot(n, wo));
    Float abs_wo_h = fabsf(dot(wo, h));
    if (abs_cos_wo <= 0 || abs_wo_h <= 0) return 0;
    Float pdf_wm = D * ggx_G1(wo, n, alpha) * abs_wo_h / abs_cos_wo;
    return pdf_wm / (Float(4) * abs_wo_h);
}

LuxHDInline vec3 ggx_to_local_z(const vec3& v,
                                const vec3& tangent,
                                const vec3& bitangent,
                                const vec3& normal) {
    return vec3(dot(v, tangent), dot(v, bitangent), dot(v, normal));
}

LuxHDInline vec3 ggx_from_local_z(const vec3& v,
                                  const vec3& tangent,
                                  const vec3& bitangent,
                                  const vec3& normal) {
    return normalize(v.x * tangent + v.y * bitangent + v.z * normal);
}

LuxHDInline vec3 sample_ggx_visible_normal(const vec3& wo,
                                           const vec3& n,
                                           Float alpha,
                                           const vec2& u) {
    vec3 tangent = make_tangent(n);
    vec3 bitangent = cross(n, tangent);
    vec3 wo_local = ggx_to_local_z(wo, tangent, bitangent, n);

    vec3 wh = normalize(vec3(alpha * wo_local.x, alpha * wo_local.y, wo_local.z));
    if (wh.z < 0) wh = -wh;

    vec3 t1 = wh.z < Float(0.99999)
        ? normalize(cross(vec3(0, 0, 1), wh))
        : vec3(1, 0, 0);
    vec3 t2 = cross(wh, t1);

    Float r = sqrtf(u.x);
    Float phi = Float(2) * kPi * u.y;
    vec2 p(r * cosf(phi), r * sinf(phi));

    Float h = sqrtf(fmaxf(Float(0), Float(1) - p.x * p.x));
    Float blend = (Float(1) + wh.z) * Float(0.5);
    p.y = (Float(1) - blend) * h + blend * p.y;

    Float pz = sqrtf(fmaxf(Float(0), Float(1) - p.x * p.x - p.y * p.y));
    vec3 nh = p.x * t1 + p.y * t2 + pz * wh;
    vec3 wm_local = normalize(vec3(
        alpha * nh.x, alpha * nh.y, fmaxf(Float(1e-6), nh.z)));
    vec3 wm = ggx_from_local_z(wm_local, tangent, bitangent, n);
    return dot(wm, n) < 0 ? -wm : wm;
}

LuxHDInline BSDFSample sample_ggx_specular(const Material& material,
                                            const SurfaceInteraction& interaction,
                                            const vec2& u) {
    BSDFSample sample;
    vec3 wo = interaction.wo;
    vec3 n = interaction.ns;
    if (dot(n, wo) <= 0) return sample;

    Float alpha = ggx_alpha(material);
    vec3 h = sample_ggx_visible_normal(wo, n, alpha, u);

    // Reflect wo around h to get wi
    sample.wi = normalize(2.0f * dot(wo, h) * h - wo);
    if (dot(n, sample.wi) <= 0) return sample;

    sample.f = eval_ggx_specular(material, interaction, sample.wi);
    sample.pdf = pdf_ggx_specular(material, interaction, sample.wi);
    sample.flags = BsdfFlagGlossy;
    finalize_smooth_bsdf_sample(sample, interaction);
    return sample;
}

struct GgxLobeProbabilities {
    Float diffuse = 0;
    Float specular = 0;
};

LuxHDInline GgxLobeProbabilities ggx_lobe_probabilities(const Material& material) {
    bool diffuse_valid = max_component(material.albedo) > 0
        && ggx_diffuse_scale(material) > 0;
    bool specular_valid = max_component(ggx_F0(material)) > 0;

    if (specular_valid && ggx_metallic(material) >= Float(1) - Float(1e-4)) {
        return GgxLobeProbabilities{0, 1};
    }
    if (diffuse_valid && specular_valid) {
        return GgxLobeProbabilities{Float(0.5), Float(0.5)};
    }
    if (diffuse_valid) return GgxLobeProbabilities{1, 0};
    if (specular_valid) return GgxLobeProbabilities{0, 1};
    return GgxLobeProbabilities{};
}

LuxHDInline vec3 eval_ggx(const Material& material,
                           const SurfaceInteraction& interaction,
                           const vec3& wi) {
    return eval_ggx_diffuse(material, interaction, wi)
         + eval_ggx_specular(material, interaction, wi);
}

LuxHDInline Float pdf_ggx(const Material& material,
                           const SurfaceInteraction& interaction,
                           const vec3& wi) {
    GgxLobeProbabilities probabilities = ggx_lobe_probabilities(material);
    return probabilities.diffuse * pdf_lambert(material, interaction, wi)
         + probabilities.specular * pdf_ggx_specular(material, interaction, wi);
}

LuxHDInline BSDFSample sample_ggx(const Material& material,
                                   const SurfaceInteraction& interaction,
                                   const vec2& u) {
    BSDFSample sample;
    vec3 n = interaction.ns;
    if (dot(n, interaction.wo) <= 0) return sample;

    GgxLobeProbabilities probabilities = ggx_lobe_probabilities(material);
    if (probabilities.diffuse <= 0 && probabilities.specular <= 0) return sample;

    if (probabilities.diffuse > 0
        && (u.x < probabilities.diffuse || probabilities.specular <= 0)) {
        vec2 lobe_u(u.x / probabilities.diffuse, u.y);
        sample.wi = cosine_sample_hemisphere(n, lobe_u);
        sample.flags = BsdfFlagDiffuse;
    } else {
        vec2 lobe_u((u.x - probabilities.diffuse) / probabilities.specular, u.y);
        BSDFSample specular_sample = sample_ggx_specular(material, interaction, lobe_u);
        if (specular_sample.pdf <= 0) return sample;
        sample.wi = specular_sample.wi;
        sample.flags = BsdfFlagGlossy;
    }

    sample.f = eval_ggx(material, interaction, sample.wi);
    sample.pdf = pdf_ggx(material, interaction, sample.wi);
    finalize_smooth_bsdf_sample(sample, interaction);
    return sample;
}

// ---------------------------------------------------------------------------
// Dielectric (Fresnel reflection / refraction)
// ---------------------------------------------------------------------------

LuxHDInline Float fresnel_dielectric(Float cos_theta_i, Float eta_i, Float eta_t) {
    // Snell's law: eta_i * sin(theta_i) = eta_t * sin(theta_t)
    Float sin_theta_i = sqrtf(fmaxf(Float(0), 1.0f - cos_theta_i * cos_theta_i));
    Float sin_theta_t = eta_i / eta_t * sin_theta_i;
    if (sin_theta_t >= 1.0f) return 1.0f; // Total internal reflection
    Float cos_theta_t = sqrtf(fmaxf(Float(0), 1.0f - sin_theta_t * sin_theta_t));

    Float r_parl = (eta_t * cos_theta_i - eta_i * cos_theta_t)
                 / (eta_t * cos_theta_i + eta_i * cos_theta_t);
    Float r_perp = (eta_i * cos_theta_i - eta_t * cos_theta_t)
                 / (eta_i * cos_theta_i + eta_t * cos_theta_t);
    return (r_parl * r_parl + r_perp * r_perp) * 0.5f;
}

LuxHDInline vec3 eval_dielectric(const Material& material,
                                  const SurfaceInteraction& interaction,
                                  const vec3& wi) {
    return vec3(0); // delta BSDF
}

LuxHDInline Float pdf_dielectric(const Material& material,
                                  const SurfaceInteraction& interaction,
                                  const vec3& wi) {
    return 0; // delta BSDF
}

LuxHDInline BSDFSample sample_dielectric(const Material& material,
                                          const SurfaceInteraction& interaction,
                                          const vec2& u) {
    BSDFSample sample;
    vec3 wo = interaction.wo;
    vec3 n = interaction.ns;

    Float cos_theta = dot(wo, n);
    bool entering = cos_theta > 0;
    Float eta_i = entering ? 1.0f : material.eta;
    Float eta_t = entering ? material.eta : 1.0f;
    vec3 normal = entering ? n : -n;
    cos_theta = fabsf(cos_theta);

    Float F = fresnel_dielectric(cos_theta, eta_i, eta_t);
    if (u.x < F) {
        // Reflect
        sample.wi = normalize(2.0f * cos_theta * normal - wo);
        sample.weight = material.albedo;
        sample.pdf = F;
        sample.pdf_measure = PdfMeasure::Discrete;
        sample.flags = BsdfFlagDeltaReflection;
    } else {
        // Refract
        Float sin_theta_t = (eta_i / eta_t) * sqrtf(fmaxf(Float(0), 1.0f - cos_theta * cos_theta));
        if (sin_theta_t >= 1.0f) {
            // Total internal reflection (shouldn't happen if F computed correctly)
            sample.wi = normalize(2.0f * cos_theta * normal - wo);
            sample.weight = material.albedo;
            sample.pdf = 1.0f;
            sample.pdf_measure = PdfMeasure::Discrete;
            sample.flags = BsdfFlagDeltaReflection;
            return sample;
        }
        Float cos_theta_t = sqrtf(fmaxf(Float(0), 1.0f - sin_theta_t * sin_theta_t));
        sample.wi = normalize((eta_i / eta_t) * (-wo) + ((eta_i / eta_t) * cos_theta - cos_theta_t) * normal);
        sample.weight = material.albedo;
        sample.pdf = 1.0f - F;
        sample.pdf_measure = PdfMeasure::Discrete;
        sample.flags = BsdfFlagDeltaTransmission | BsdfFlagTransmission;
    }
    return sample;
}

// ---------------------------------------------------------------------------
// Interface / Transparent (ideal straight-through transmission)
// ---------------------------------------------------------------------------

LuxHDInline vec3 eval_interface(const Material& material,
                                const SurfaceInteraction& interaction,
                                const vec3& wi) {
    return vec3(0); // null interface, handled by the integrator fast path
}

LuxHDInline Float pdf_interface(const Material& material,
                                const SurfaceInteraction& interaction,
                                const vec3& wi) {
    return 0; // null interface, handled by the integrator fast path
}

LuxHDInline BSDFSample sample_interface(const Material& material,
                                        const SurfaceInteraction& interaction,
                                        const vec2& u) {
    (void)material;
    (void)u;
    BSDFSample sample;
    sample.wi = normalize(-interaction.wo);
    sample.weight = vec3(1);
    sample.pdf = 1.0f;
    sample.pdf_measure = PdfMeasure::Discrete;
    sample.flags = BsdfFlagDeltaTransmission | BsdfFlagTransmission;
    return sample;
}

LuxHDInline vec3 eval_transparent(const Material& material,
                                  const SurfaceInteraction& interaction,
                                  const vec3& wi) {
    return vec3(0); // tinted null surface, handled by the integrator fast path
}

LuxHDInline Float pdf_transparent(const Material& material,
                                  const SurfaceInteraction& interaction,
                                  const vec3& wi) {
    return 0; // tinted null surface, handled by the integrator fast path
}

LuxHDInline BSDFSample sample_transparent(const Material& material,
                                          const SurfaceInteraction& interaction,
                                          const vec2& u) {
    (void)u;
    BSDFSample sample;
    sample.wi = normalize(-interaction.wo);
    sample.weight = material.albedo;
    sample.pdf = 1.0f;
    sample.pdf_measure = PdfMeasure::Discrete;
    sample.flags = BsdfFlagDeltaTransmission | BsdfFlagTransmission;
    return sample;
}

template <MaterialType Type>
LuxHDInline vec3 eval_bsdf_typed(const Material& material,
                                  const SurfaceInteraction& interaction,
                                  const vec3& wi) {
    if constexpr (Type == MaterialType::Lambert) {
        return eval_lambert(material, interaction, wi);
    } else if constexpr (Type == MaterialType::ThinLambert) {
        return eval_thin_lambert(material, interaction, wi);
    } else if constexpr (Type == MaterialType::GGX) {
        return eval_ggx(material, interaction, wi);
    } else if constexpr (Type == MaterialType::Dielectric) {
        return eval_dielectric(material, interaction, wi);
    } else if constexpr (Type == MaterialType::Interface) {
        return eval_interface(material, interaction, wi);
    } else if constexpr (Type == MaterialType::Transparent) {
        return eval_transparent(material, interaction, wi);
    } else {
        return vec3(0);
    }
}

template <MaterialType Type>
LuxHDInline Float pdf_bsdf_typed(const Material& material,
                                  const SurfaceInteraction& interaction,
                                  const vec3& wi) {
    if constexpr (Type == MaterialType::Lambert) {
        return pdf_lambert(material, interaction, wi);
    } else if constexpr (Type == MaterialType::ThinLambert) {
        return pdf_thin_lambert(material, interaction, wi);
    } else if constexpr (Type == MaterialType::GGX) {
        return pdf_ggx(material, interaction, wi);
    } else if constexpr (Type == MaterialType::Dielectric) {
        return pdf_dielectric(material, interaction, wi);
    } else if constexpr (Type == MaterialType::Interface) {
        return pdf_interface(material, interaction, wi);
    } else if constexpr (Type == MaterialType::Transparent) {
        return pdf_transparent(material, interaction, wi);
    } else {
        return 0;
    }
}

template <MaterialType Type>
LuxHDInline BSDFSample sample_bsdf_typed(const Material& material,
                                          const SurfaceInteraction& interaction,
                                          const vec2& u) {
    if constexpr (Type == MaterialType::Lambert) {
        return sample_lambert(material, interaction, u);
    } else if constexpr (Type == MaterialType::ThinLambert) {
        return sample_thin_lambert(material, interaction, u);
    } else if constexpr (Type == MaterialType::GGX) {
        return sample_ggx(material, interaction, u);
    } else if constexpr (Type == MaterialType::Dielectric) {
        return sample_dielectric(material, interaction, u);
    } else if constexpr (Type == MaterialType::Interface) {
        return sample_interface(material, interaction, u);
    } else if constexpr (Type == MaterialType::Transparent) {
        return sample_transparent(material, interaction, u);
    } else {
        return BSDFSample{};
    }
}

// ---------------------------------------------------------------------------
// Unified BSDF interface
// ---------------------------------------------------------------------------

LuxHDInline vec3 eval_bsdf(const Material& material,
                            const SurfaceInteraction& interaction,
                            const vec3& wi) {
    switch (material.type) {
    case MaterialType::Lambert:
        return eval_lambert(material, interaction, wi);
    case MaterialType::ThinLambert:
        return eval_thin_lambert(material, interaction, wi);
    case MaterialType::GGX:
        return eval_ggx(material, interaction, wi);
    case MaterialType::Dielectric:
        return eval_dielectric(material, interaction, wi);
    case MaterialType::Interface:
        return eval_interface(material, interaction, wi);
    case MaterialType::Transparent:
        return eval_transparent(material, interaction, wi);
    }
    return vec3(0);
}

LuxHDInline Float pdf_bsdf(const Material& material,
                            const SurfaceInteraction& interaction,
                            const vec3& wi) {
    switch (material.type) {
    case MaterialType::Lambert:
        return pdf_lambert(material, interaction, wi);
    case MaterialType::ThinLambert:
        return pdf_thin_lambert(material, interaction, wi);
    case MaterialType::GGX:
        return pdf_ggx(material, interaction, wi);
    case MaterialType::Dielectric:
        return pdf_dielectric(material, interaction, wi);
    case MaterialType::Interface:
        return pdf_interface(material, interaction, wi);
    case MaterialType::Transparent:
        return pdf_transparent(material, interaction, wi);
    }
    return 0;
}

LuxHDInline BSDFSample sample_bsdf(const Material& material,
                                    const SurfaceInteraction& interaction,
                                    const vec2& u) {
    switch (material.type) {
    case MaterialType::Lambert:
        return sample_lambert(material, interaction, u);
    case MaterialType::ThinLambert:
        return sample_thin_lambert(material, interaction, u);
    case MaterialType::GGX:
        return sample_ggx(material, interaction, u);
    case MaterialType::Dielectric:
        return sample_dielectric(material, interaction, u);
    case MaterialType::Interface:
        return sample_interface(material, interaction, u);
    case MaterialType::Transparent:
        return sample_transparent(material, interaction, u);
    }
    return BSDFSample{};
}

#endif // LUX_MATERIAL_BSDF_H
