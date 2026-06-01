#pragma once

#ifndef LUX_MATERIAL_H
#define LUX_MATERIAL_H

#include "core/vec3.cuh"

enum class MaterialType {
    Lambert,
    ThinLambert,
    GGX,
    Dielectric,
    Interface,
    Transparent,
};

enum class EmissionSidedness {
    SingleSided,
    DoubleSided,
};

enum class EmissionSourceKind : unsigned char {
    Constant,
    Image,
};

constexpr Float kEmissionCosineEpsilon = Float(1e-7);

struct Material {
    MaterialType type = MaterialType::Lambert;
    vec3 albedo = vec3(0.8f);
    vec3 emission = vec3(0);
    EmissionSidedness emission_sidedness = EmissionSidedness::SingleSided;
    EmissionSourceKind emission_source = EmissionSourceKind::Constant;
    int emission_image_id = -1;
    vec3 emission_scale = vec3(1);
    Float roughness = 0.5f;
    Float metallic = 0;
    Float eta = 1.5f;

    LuxHDInline Material() = default;

    LuxHDInline Material(MaterialType type, const vec3& albedo,
                          const vec3& emission = vec3(0), Float roughness = 0.5f,
                          Float metallic = 0, Float eta = 1.5f,
                          EmissionSidedness emission_sidedness = EmissionSidedness::SingleSided)
        : type(type), albedo(albedo), emission(emission),
          emission_sidedness(emission_sidedness),
          roughness(roughness), metallic(metallic), eta(eta) {}

    LuxHDInline static Material lambert(const vec3& albedo) {
        return Material(MaterialType::Lambert, albedo);
    }

    LuxHDInline static Material thin_lambert(const vec3& albedo) {
        return Material(MaterialType::ThinLambert, albedo);
    }

    LuxHDInline static Material ggx(const vec3& albedo, Float roughness,
                                     Float metallic = 0) {
        return Material(MaterialType::GGX, albedo, vec3(0), roughness, metallic);
    }

    LuxHDInline static Material dielectric(const vec3& albedo, Float eta = 1.5f) {
        return Material(MaterialType::Dielectric, albedo, vec3(0), 0, 0, eta);
    }

    LuxHDInline static Material glass(const vec3& albedo = vec3(1), Float eta = 1.5f) {
        return dielectric(albedo, eta);
    }

    LuxHDInline static Material interface() {
        return Material(MaterialType::Interface, vec3(1));
    }

    LuxHDInline static Material transparent(const vec3& transmittance = vec3(1)) {
        return Material(MaterialType::Transparent, transmittance);
    }

    LuxHDInline static Material tinted_filter(const vec3& transmittance = vec3(1)) {
        return transparent(transmittance);
    }

    LuxHDInline static Material emissive(
            const vec3& emission,
            const vec3& albedo = vec3(0),
            EmissionSidedness emission_sidedness = EmissionSidedness::SingleSided) {
        return Material(MaterialType::Lambert, albedo, emission, 0.5f, 0, 1.5f,
                        emission_sidedness);
    }

    LuxHDInline static Material image_emissive(
            int image_id,
            const vec3& scale = vec3(1),
            const vec3& albedo = vec3(0),
            EmissionSidedness emission_sidedness = EmissionSidedness::SingleSided) {
        Material material(MaterialType::Lambert, albedo, vec3(0), 0.5f, 0, 1.5f,
                          emission_sidedness);
        material.emission_source = EmissionSourceKind::Image;
        material.emission_image_id = image_id;
        material.emission_scale = scale;
        return material;
    }

    LuxHDInline bool is_emissive() const {
        return max_component(emission) > 0
            || (emission_source == EmissionSourceKind::Image && emission_image_id >= 0);
    }

    LuxHDInline bool is_scattering() const {
        switch (type) {
            case MaterialType::Lambert:
            case MaterialType::ThinLambert: return max_component(albedo) > 0;
            case MaterialType::GGX:
            case MaterialType::Dielectric: return true;
            case MaterialType::Interface:
            case MaterialType::Transparent: return false;
        }
        return false;
    }
};

// wo points from the emitting surface toward the observer / previous path vertex.
// This is the sidedness source of truth for both Le evaluation and light PDF.
LuxHDInline Float emission_projected_cosine(const vec3& ng, const vec3& wo,
                                            EmissionSidedness sidedness) {
    Float c = dot(ng, wo);
    if (sidedness == EmissionSidedness::SingleSided) {
        return c > kEmissionCosineEpsilon ? c : Float(0);
    }

    c = fabsf(c);
    return c > kEmissionCosineEpsilon ? c : Float(0);
}

LuxHDInline vec3 eval_emission(const vec3& emission,
                               EmissionSidedness sidedness,
                               const vec3& ng,
                               const vec3& wo) {
    if (max_component(emission) <= 0) return vec3(0);
    if (emission_projected_cosine(ng, wo, sidedness) <= 0) return vec3(0);
    return emission;
}

LuxHDInline vec3 eval_emission(const Material& material,
                               const vec3& ng,
                               const vec3& wo) {
    return eval_emission(material.emission, material.emission_sidedness, ng, wo);
}

LuxHDInline bool is_passthrough_material(MaterialType type) {
    return type == MaterialType::Interface || type == MaterialType::Transparent;
}

LuxHDInline bool is_passthrough_material(const Material& material) {
    return is_passthrough_material(material.type);
}

LuxHDInline vec3 passthrough_transmittance(const Material& material) {
    return material.type == MaterialType::Transparent ? material.albedo : vec3(1);
}

LuxHDInline bool has_direct_lighting_lobe(MaterialType type) {
    switch (type) {
        case MaterialType::Lambert:
        case MaterialType::ThinLambert:
        case MaterialType::GGX: return true;
        case MaterialType::Dielectric:
        case MaterialType::Interface:
        case MaterialType::Transparent: return false;
    }
    return false;
}

LuxHDInline bool has_direct_lighting_lobe(const Material& material) {
    return has_direct_lighting_lobe(material.type);
}

#endif // LUX_MATERIAL_H
