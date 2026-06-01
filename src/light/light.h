#pragma once

#ifndef LUX_LIGHT_H
#define LUX_LIGHT_H

#include "core/constants.h"
#include "core/ray.cuh"
#include "core/types.h"
#include "core/vec2.cuh"
#include "core/vec3.cuh"
#include "material/material.h"
#include "sampler/sample_record.h"
#include "scene/primitive_ref.h"

enum class LightType : unsigned char {
    DeltaPosition,
    DeltaDirection,
    Area,
    Infinite,
};

#define LUX_LIGHT_KIND_LIST(X) \
    X(DiffuseArea,      Area,           "diffuse area light",      sample_diffuse_area_light_li,      diffuse_area_light_pdf_li,      diffuse_area_light_le) \
    X(Point,            DeltaPosition,  "point light",             sample_point_light_li,             point_light_pdf_li,             point_light_le) \
    X(Distant,          DeltaDirection, "distant light",           sample_distant_light_li,           distant_light_pdf_li,           distant_light_le) \
    X(UniformInfinite,  Infinite,       "uniform infinite light",  sample_uniform_infinite_light_li,  uniform_infinite_light_pdf_li,  uniform_infinite_light_le) \
    X(ImageInfinite,    Infinite,       "image infinite light",    sample_image_infinite_light_li,    image_infinite_light_pdf_li,    image_infinite_light_le)

enum class LightKind : unsigned char {
#define LUX_DECLARE_LIGHT_KIND(kind, type, label, sample_fn, pdf_fn, le_fn) kind,
    LUX_LIGHT_KIND_LIST(LUX_DECLARE_LIGHT_KIND)
#undef LUX_DECLARE_LIGHT_KIND
};

struct LightSampleContext {
    vec3 position = vec3(0);
    vec3 ng = vec3(0);
    vec3 ns = vec3(0);
};

struct LightLiSample {
    vec3 wi = vec3(0);
    vec3 position = vec3(0);
    vec3 normal = vec3(0);
    vec3 radiance = vec3(0);
    Float distance = INFINITY;
    Float pdf = 0;
    PdfMeasure pdf_measure = PdfMeasure::SolidAngle;
    int light_index = -1;
    int primitive_id = -1;
    bool valid = false;
};

struct DiffuseAreaLight {
    int primitive_id = -1;
    vec3 emission = vec3(0);
    EmissionSidedness sidedness = EmissionSidedness::SingleSided;
    EmissionSourceKind emission_source = EmissionSourceKind::Constant;
    int emission_image_id = -1;
    vec3 emission_scale = vec3(1);
    Float area = 0;
    PrimitiveRef primitive_ref;
};

struct PointLight {
    vec3 position = vec3(0);
    vec3 intensity = vec3(0);
};

struct DistantLight {
    vec3 direction_to_light = vec3(0, 1, 0);
    vec3 radiance = vec3(0);
};

struct UniformInfiniteLight {
    vec3 radiance = vec3(0);
};

struct ImageInfiniteLight {
    const vec3* pixels = nullptr;
    const Float* cdf = nullptr;
    int image_id = -1;
    int width = 0;
    int height = 0;
    Float total_weight = 0;
    vec3 scale = vec3(1);
};

struct Light {
    LightKind kind = LightKind::DiffuseArea;
    DiffuseAreaLight area;
    PointLight point;
    DistantLight distant;
    UniformInfiniteLight uniform_infinite;
    ImageInfiniteLight image_infinite;
};

struct SampledLight {
    int light_index = -1;
    Light light;
    Float pmf = 0;
    bool valid = false;
};

LuxHDInline LightType light_type(LightKind kind) {
    switch (kind) {
#define LUX_LIGHT_TYPE_CASE(kind, type, label, sample_fn, pdf_fn, le_fn) \
        case LightKind::kind: return LightType::type;
        LUX_LIGHT_KIND_LIST(LUX_LIGHT_TYPE_CASE)
#undef LUX_LIGHT_TYPE_CASE
    }
    return LightType::Area;
}

LuxInline const char* light_kind_label(LightKind kind) {
    switch (kind) {
#define LUX_LIGHT_LABEL_CASE(kind, type, label, sample_fn, pdf_fn, le_fn) \
        case LightKind::kind: return label;
        LUX_LIGHT_KIND_LIST(LUX_LIGHT_LABEL_CASE)
#undef LUX_LIGHT_LABEL_CASE
    }
    return "unknown light";
}

LuxHDInline LightType light_type(const Light& light) {
    return light_type(light.kind);
}

LuxHDInline bool is_delta_light(LightType type) {
    return type == LightType::DeltaPosition || type == LightType::DeltaDirection;
}

LuxHDInline Float luminance(const vec3& value) {
    return Float(0.2126) * value.x + Float(0.7152) * value.y + Float(0.0722) * value.z;
}

LuxHDInline vec3 uniform_sphere_direction(const vec2& u) {
    Float z = Float(1) - Float(2) * u.x;
    Float r = sqrtf(fmaxf(Float(0), Float(1) - z * z));
    Float phi = Float(2) * kPi * u.y;
    return vec3(r * cosf(phi), z, r * sinf(phi));
}

LuxHDInline vec2 direction_to_latlong_uv(const vec3& wi) {
    vec3 w = normalize(wi);
    Float phi = atan2f(w.z, w.x);
    if (phi < 0) phi += Float(2) * kPi;
    Float theta = acosf(fmaxf(Float(-1), fminf(Float(1), w.y)));
    return vec2(phi * (Float(0.5) * kInvPi), theta * kInvPi);
}

LuxHDInline vec3 latlong_uv_to_direction(const vec2& uv) {
    Float phi = Float(2) * kPi * uv.x;
    Float theta = kPi * uv.y;
    Float sin_theta = sinf(theta);
    return vec3(cosf(phi) * sin_theta, cosf(theta), sinf(phi) * sin_theta);
}

LuxHDInline vec3 image_env_lookup(const ImageInfiniteLight& light, const vec3& wi) {
    if (!light.pixels || light.width <= 0 || light.height <= 0) return vec3(0);
    vec2 uv = direction_to_latlong_uv(wi);
    int x = static_cast<int>(fminf(Float(light.width - 1), floorf(uv.x * light.width)));
    int y = static_cast<int>(fminf(Float(light.height - 1), floorf(uv.y * light.height)));
    return light.pixels[y * light.width + x] * light.scale;
}

LuxHDInline LightLiSample sample_point_light(const PointLight& light,
                                             const LightSampleContext& ctx) {
    LightLiSample sample;
    vec3 to_light = light.position - ctx.position;
    Float distance2 = length2(to_light);
    if (distance2 <= kRayEpsilon * kRayEpsilon) return sample;
    sample.distance = sqrtf(distance2);
    sample.wi = to_light / sample.distance;
    sample.position = light.position;
    sample.radiance = light.intensity / distance2;
    sample.pdf = 1;
    sample.pdf_measure = PdfMeasure::Discrete;
    sample.valid = max_component(sample.radiance) > 0;
    return sample;
}

LuxHDInline LightLiSample sample_distant_light(const DistantLight& light,
                                               const LightSampleContext&) {
    LightLiSample sample;
    sample.wi = normalize(light.direction_to_light);
    sample.distance = INFINITY;
    sample.radiance = light.radiance;
    sample.pdf = 1;
    sample.pdf_measure = PdfMeasure::Discrete;
    sample.valid = max_component(sample.radiance) > 0;
    return sample;
}

LuxHDInline LightLiSample sample_uniform_infinite_light(const UniformInfiniteLight& light,
                                                       const vec2& u) {
    LightLiSample sample;
    sample.wi = uniform_sphere_direction(u);
    sample.distance = INFINITY;
    sample.radiance = light.radiance;
    sample.pdf = Float(1) / (Float(4) * kPi);
    sample.valid = max_component(sample.radiance) > 0;
    return sample;
}

LuxHDInline Float uniform_infinite_pdf() {
    return Float(1) / (Float(4) * kPi);
}

LuxHDInline Float image_infinite_pdf_for_texel(const ImageInfiniteLight& light,
                                               int index, Float uv_y) {
    if (!light.cdf || light.width <= 0 || light.height <= 0
        || light.total_weight <= 0) {
        return 0;
    }
    int count = light.width * light.height;
    if (index < 0 || index >= count) return 0;

    Float texel_weight = index == 0 ? light.cdf[0]
                                    : light.cdf[index] - light.cdf[index - 1];
    if (texel_weight <= 0) return 0;

    Float theta = kPi * uv_y;
    Float sin_theta = fmaxf(Float(1e-6), sinf(theta));
    Float uv_pdf = (texel_weight / light.total_weight) * Float(count);
    return uv_pdf / (Float(2) * kPi * kPi * sin_theta);
}

LuxHDInline LightLiSample sample_image_infinite_light(const ImageInfiniteLight& light,
                                                     Float select_u,
                                                     const vec2& remap_u) {
    LightLiSample sample;
    if (!light.pixels || !light.cdf || light.width <= 0 || light.height <= 0
        || light.total_weight <= 0) {
        return sample;
    }

    int count = light.width * light.height;
    Float target = fminf(select_u, Float(0.99999994)) * light.total_weight;
    int index = count - 1;
    for (int i = 0; i < count; ++i) {
        if (target <= light.cdf[i]) {
            index = i;
            break;
        }
    }
    int x = index % light.width;
    int y = index / light.width;
    vec2 uv((Float(x) + remap_u.x) / Float(light.width),
            (Float(y) + remap_u.y) / Float(light.height));
    sample.wi = latlong_uv_to_direction(uv);
    sample.distance = INFINITY;
    sample.radiance = image_env_lookup(light, sample.wi);
    sample.pdf = image_infinite_pdf_for_texel(light, index, uv.y);
    sample.valid = max_component(sample.radiance) > 0 && sample.pdf > 0;
    return sample;
}

LuxHDInline Float image_infinite_pdf(const ImageInfiniteLight& light, const vec3& wi) {
    if (!light.pixels || !light.cdf || light.width <= 0 || light.height <= 0
        || light.total_weight <= 0) {
        return 0;
    }
    vec2 uv = direction_to_latlong_uv(wi);
    int x = static_cast<int>(fminf(Float(light.width - 1), floorf(uv.x * light.width)));
    int y = static_cast<int>(fminf(Float(light.height - 1), floorf(uv.y * light.height)));
    int index = y * light.width + x;
    return image_infinite_pdf_for_texel(light, index, uv.y);
}

LuxHDInline LightLiSample sample_diffuse_area_light_li(
        const Light&, const LightSampleContext&, Float, const vec2&) {
    return LightLiSample{};
}

LuxHDInline LightLiSample sample_point_light_li(
        const Light& light, const LightSampleContext& ctx, Float, const vec2&) {
    return sample_point_light(light.point, ctx);
}

LuxHDInline LightLiSample sample_distant_light_li(
        const Light& light, const LightSampleContext& ctx, Float, const vec2&) {
    return sample_distant_light(light.distant, ctx);
}

LuxHDInline LightLiSample sample_uniform_infinite_light_li(
        const Light& light, const LightSampleContext&, Float, const vec2& u) {
    return sample_uniform_infinite_light(light.uniform_infinite, u);
}

LuxHDInline LightLiSample sample_image_infinite_light_li(
        const Light& light, const LightSampleContext&, Float select_u, const vec2& u) {
    return sample_image_infinite_light(light.image_infinite, select_u, u);
}

LuxHDInline Float diffuse_area_light_pdf_li(
        const Light&, const LightSampleContext&, const vec3&) {
    return 0;
}

LuxHDInline Float point_light_pdf_li(
        const Light&, const LightSampleContext&, const vec3&) {
    return 0;
}

LuxHDInline Float distant_light_pdf_li(
        const Light&, const LightSampleContext&, const vec3&) {
    return 0;
}

LuxHDInline Float uniform_infinite_light_pdf_li(
        const Light&, const LightSampleContext&, const vec3&) {
    return uniform_infinite_pdf();
}

LuxHDInline Float image_infinite_light_pdf_li(
        const Light& light, const LightSampleContext&, const vec3& wi) {
    return image_infinite_pdf(light.image_infinite, wi);
}

LuxHDInline vec3 diffuse_area_light_le(const Light&, const Ray&) {
    return vec3(0);
}

LuxHDInline vec3 point_light_le(const Light&, const Ray&) {
    return vec3(0);
}

LuxHDInline vec3 distant_light_le(const Light&, const Ray&) {
    return vec3(0);
}

LuxHDInline vec3 uniform_infinite_light_le(const Light& light, const Ray&) {
    return light.uniform_infinite.radiance;
}

LuxHDInline vec3 image_infinite_light_le(const Light& light, const Ray& ray) {
    return image_env_lookup(light.image_infinite, ray.direction);
}

LuxHDInline vec3 infinite_light_le(const Light& light, const Ray& ray) {
    switch (light.kind) {
#define LUX_LIGHT_LE_CASE(kind, type, label, sample_fn, pdf_fn, le_fn) \
        case LightKind::kind: return le_fn(light, ray);
        LUX_LIGHT_KIND_LIST(LUX_LIGHT_LE_CASE)
#undef LUX_LIGHT_LE_CASE
    }
    return vec3(0);
}

LuxHDInline Float light_pdf_li(const Light& light, const LightSampleContext& ctx,
                               const vec3& wi) {
    switch (light.kind) {
#define LUX_LIGHT_PDF_CASE(kind, type, label, sample_fn, pdf_fn, le_fn) \
        case LightKind::kind: return pdf_fn(light, ctx, wi);
        LUX_LIGHT_KIND_LIST(LUX_LIGHT_PDF_CASE)
#undef LUX_LIGHT_PDF_CASE
    }
    return 0;
}

LuxHDInline LightLiSample sample_light_li(const Light& light,
                                          const LightSampleContext& ctx,
                                          Float select_u,
                                          const vec2& u) {
    switch (light.kind) {
#define LUX_LIGHT_SAMPLE_CASE(kind, type, label, sample_fn, pdf_fn, le_fn) \
        case LightKind::kind: return sample_fn(light, ctx, select_u, u);
        LUX_LIGHT_KIND_LIST(LUX_LIGHT_SAMPLE_CASE)
#undef LUX_LIGHT_SAMPLE_CASE
    }
    return LightLiSample{};
}

#endif // LUX_LIGHT_H
