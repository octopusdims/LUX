#include "integrator/cpu_path_tracer.h"
#include "integrator/cpu_path_tracer_diagnostics.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "bvh/cpu_bvh.h"
#include "camera/camera.h"
#include "core/constants.h"
#include "core/pcg.cuh"
#include "core/ray.cuh"
#include "core/ray_offset.cuh"
#include "core/types.h"
#include "geometry/triangle.h"
#include "integrator/path_transport.cuh"
#include "light/light_sampler.h"
#include "material/bsdf.h"
#include "scene/scene_light_sampler.h"
#include "scene/scene_intersection.h"

namespace cpu_path_tracer_detail {

ShadowTraceResult trace_shadow(const Scene& scene, const CpuBvh& bvh,
                               const vec3& from, const vec3& to,
                               PrimitiveRef skip_primitive) {
    ShadowTraceResult result;
    vec3 to_light = to - from;
    Float distance = length(to_light);
    if (distance <= kRayEpsilon) {
        result.transmittance = vec3(0);
        return result;
    }

    Ray shadow_ray(from, to_light / distance, kRayEpsilon, distance - kRayEpsilon);
    PrimitiveRef ignored_primitive = skip_primitive;
    while (true) {
        SurfaceHit shadow_hit;
        shadow_hit.t = shadow_ray.t_max;
        if (!intersect_scene_bvh_skip_ref(
                scene, bvh, shadow_ray, ignored_primitive, shadow_hit)) {
            return result;
        }

        SceneTriangle scene_triangle = scene_triangle_view(scene, shadow_hit.triangle_id);
        const Material& material = scene.materials[scene_triangle.material_id];
        if (!is_passthrough_material(material)) {
            result.transmittance = vec3(0);
            result.opaque_hit_found = true;
            result.opaque_hit = shadow_hit;
            return result;
        }

        result.transmittance = result.transmittance * passthrough_transmittance(material);
        ++result.passthrough_hits;
        if (max_component(result.transmittance) <= 0) {
            return result;
        }

        Float advance = shadow_hit.t + kRayEpsilon;
        if (advance >= shadow_ray.t_max) {
            return result;
        }
        shadow_ray.origin = shadow_ray.at(advance);
        shadow_ray.t_max -= advance;
        shadow_ray.t_min = kRayEpsilon;
        ignored_primitive = invalid_primitive_ref();
    }
    return result;
}

LuxInline ShadowTraceResult trace_shadow_ray(const Scene& scene, const CpuBvh& bvh,
                                             Ray shadow_ray,
                                             PrimitiveRef skip_primitive) {
    ShadowTraceResult result;
    PrimitiveRef ignored_primitive = skip_primitive;
    while (true) {
        SurfaceHit shadow_hit;
        shadow_hit.t = shadow_ray.t_max;
        if (!intersect_scene_bvh_skip_ref(
                scene, bvh, shadow_ray, ignored_primitive, shadow_hit)) {
            return result;
        }

        SceneTriangle scene_triangle = scene_triangle_view(scene, shadow_hit.triangle_id);
        const Material& material = scene.materials[scene_triangle.material_id];
        if (!is_passthrough_material(material)) {
            result.transmittance = vec3(0);
            result.opaque_hit_found = true;
            result.opaque_hit = shadow_hit;
            return result;
        }

        result.transmittance = result.transmittance * passthrough_transmittance(material);
        ++result.passthrough_hits;
        if (max_component(result.transmittance) <= 0) {
            return result;
        }

        Float advance = shadow_hit.t + kRayEpsilon;
        if (advance >= shadow_ray.t_max) {
            return result;
        }
        shadow_ray.origin = shadow_ray.at(advance);
        shadow_ray.t_max -= advance;
        shadow_ray.t_min = kRayEpsilon;
        ignored_primitive = invalid_primitive_ref();
    }
    return result;
}

vec3 shadow_transmittance(const Scene& scene, const CpuBvh& bvh,
                          const vec3& from, const vec3& to,
                          int skip_triangle_id) {
    return trace_shadow(
        scene, bvh, from, to, scene_primitive_ref(scene, skip_triangle_id)).transmittance;
}

LuxInline vec3 estimate_direct_light(const Scene& scene, const CpuBvh& bvh,
                                     const SceneLightSampler& lights,
                                     const SurfaceInteraction& interaction,
                                     const Material& material, PrimitiveRef source_primitive,
                                     SamplerState& sampler) {
    if (!has_direct_lighting_lobe(material)) {
        return vec3(0);
    }

        LightSampleContext ctx{interaction.position, interaction.ng, interaction.ns};
        LightLiSample light = sample_light_li(
            lights, ctx, sampler_get_1d(sampler), sampler_get_1d(sampler),
            sampler_get_2d(sampler));
        if (!light.valid) {
            return vec3(0);
        }

        DirectLightEstimate estimate =
            estimate_unoccluded_direct_light_li(material, interaction, light);
        if (!estimate.valid) return vec3(0);

        vec3 shadow_origin = offset_ray_origin_along_normal(
            interaction.position, interaction.ng, estimate.wi);
        ShadowTraceResult shadow = light.distance < INFINITY
            ? trace_shadow(scene, bvh, shadow_origin, light.position, source_primitive)
            : trace_shadow_ray(
                scene, bvh, Ray(shadow_origin, estimate.wi, kRayEpsilon),
                source_primitive);
        if (max_component(shadow.transmittance) <= 0) {
            return vec3(0);
        }

    Float mis_weight = direct_light_mis_weight(estimate);
    return estimate.contribution * shadow.transmittance * mis_weight;
}

struct CpuAovTargets {
    Film* shadow_debug = nullptr;
    Film* primary_debug = nullptr;
    Film* geometric_normal = nullptr;
    Film* shading_normal = nullptr;

    bool enabled() const {
        return shadow_debug || primary_debug || geometric_normal || shading_normal;
    }
};

LuxInline void validate_aov_film(Film* aov, const Film& beauty, const char* label) {
    if (!aov) return;
    if (aov->width != beauty.width || aov->height != beauty.height
        || aov->samples_per_pixel != beauty.samples_per_pixel
        || aov->pixels.size() != beauty.pixels.size()) {
        throw std::runtime_error(
            std::string("CPU path tracer AOV Film mismatch: ") + label);
    }
}

LuxInline CpuAovTargets make_cpu_aov_targets(RenderOutputs* outputs,
                                             const Film& beauty) {
    CpuAovTargets targets;
    if (!outputs) return targets;

    targets.shadow_debug = outputs->get(RenderAov::ShadowDebug);
    targets.primary_debug = outputs->get(RenderAov::PrimaryDebug);
    targets.geometric_normal = outputs->get(RenderAov::GeometricNormal);
    targets.shading_normal = outputs->get(RenderAov::ShadingNormal);

    validate_aov_film(targets.shadow_debug, beauty, "shadow debug");
    validate_aov_film(targets.primary_debug, beauty, "primary debug");
    validate_aov_film(targets.geometric_normal, beauty, "geometric normal");
    validate_aov_film(targets.shading_normal, beauty, "shading normal");

    if (targets.shadow_debug) {
        std::fill(targets.shadow_debug->pixels.begin(),
                  targets.shadow_debug->pixels.end(), vec3(0));
    }
    if (targets.primary_debug) {
        std::fill(targets.primary_debug->pixels.begin(),
                  targets.primary_debug->pixels.end(), vec3(0));
    }
    if (targets.geometric_normal) {
        std::fill(targets.geometric_normal->pixels.begin(),
                  targets.geometric_normal->pixels.end(), vec3(0));
    }
    if (targets.shading_normal) {
        std::fill(targets.shading_normal->pixels.begin(),
                  targets.shading_normal->pixels.end(), vec3(0));
    }
    return targets;
}

LuxInline vec3 encode_normal_color(const vec3& normal) {
    return Float(0.5) * (normalize(normal) + vec3(1));
}

LuxInline bool is_near_boundary_edge(const SceneTriangle& triangle, Float u, Float v) {
    constexpr Float kBoundaryBarycentricEpsilon = Float(0.02);
    if (triangle.boundary_edges == 0) return false;

    Float w = Float(1) - u - v;
    bool near_v0_v1 = (triangle.boundary_edges & 1u) != 0
        && v <= kBoundaryBarycentricEpsilon;
    bool near_v1_v2 = (triangle.boundary_edges & 2u) != 0
        && w <= kBoundaryBarycentricEpsilon;
    bool near_v2_v0 = (triangle.boundary_edges & 4u) != 0
        && u <= kBoundaryBarycentricEpsilon;
    return near_v0_v1 || near_v1_v2 || near_v2_v0;
}

LuxInline vec3 primary_debug_classification_color(const Scene& scene,
                                                  const SurfaceHit& hit,
                                                  const Ray& ray) {
    if (hit.triangle_id < 0) return vec3(0);
    SceneTriangle scene_triangle = scene_triangle_view(scene, hit.triangle_id);
    const Material& material = scene.materials[scene_triangle.material_id];
    if (material.is_emissive()) {
        return vec3(1, 1, 0);
    }

    Float ng_wo = dot(triangle_normal(scene_triangle.triangle), -ray.direction);
    return ng_wo < 0 ? vec3(1, 0, 1) : vec3(0, 1, 1);
}

LuxInline vec3 primary_normal_debug_color(const Scene& scene,
                                          const SurfaceHit& hit,
                                          bool shading_normal) {
    if (hit.triangle_id < 0) return vec3(0);
    SceneTriangle scene_triangle = scene_triangle_view(scene, hit.triangle_id);
    vec3 ng = triangle_normal(scene_triangle.triangle);
    if (!shading_normal) {
        return encode_normal_color(ng);
    }

    vec3 ns = interpolate_triangle_normal(scene_triangle, hit.u, hit.v);
    if (dot(ng, ns) < 0) {
        ns = -ns;
    }
    return encode_normal_color(ns);
}

LuxInline vec3 shadow_debug_classification_color(
        const Scene& scene,
        const SceneTriangle& source_triangle,
        int source_material_id,
        const vec3& source_geometric_normal,
        const vec3& source_shading_normal,
        const Ray& shadow_ray,
        int occluder_id,
        Float hit_t,
        Float hit_u,
        Float hit_v,
        int passthrough_hits) {
    Float shading_cos = dot(source_shading_normal, shadow_ray.direction);
    Float geometric_cos = dot(source_geometric_normal, shadow_ray.direction);
    if (shading_cos > 0 && geometric_cos <= 0) {
        return vec3(1, 0, 0);
    }

    if (occluder_id < 0) {
        return passthrough_hits > 0 ? vec3(0, 0.35f, 0.8f) : vec3(0.03f);
    }

    SceneTriangle occluder = scene_triangle_view(scene, occluder_id);
    bool same_material = occluder.material_id == source_material_id;
    if (same_material && hit_t < Float(16) * kRayEpsilon) {
        return vec3(1, 1, 0);
    }

    if (same_material) {
        bool backface_hit = dot(triangle_normal(occluder.triangle), -shadow_ray.direction) <= 0;
        if (backface_hit) {
            return is_near_boundary_edge(occluder, hit_u, hit_v)
                ? vec3(1, 0, 1)
                : vec3(0, 0, 1);
        }
        return vec3(0, 1, 1);
    }

    (void)source_triangle;
    return vec3(0, 1, 0);
}

LuxInline void accumulate_shadow_debug_aov(const Scene& scene, const CpuBvh& bvh,
                                           const SceneLightSampler& lights,
                                           const SurfaceHit& source_hit,
                                           const Ray& camera_ray,
                                           SamplerState& sampler,
                                           Film& shadow_debug,
                                           int pixel) {
    if (lights.empty()) return;

    SceneTriangle source_triangle = scene_triangle_view(scene, source_hit.triangle_id);
    int source_material_id = source_triangle.material_id;
    const Material& source_material = scene.materials[source_material_id];
    if (!has_direct_lighting_lobe(source_material)) return;

    SurfaceInteraction interaction = make_surface_interaction(
        source_hit.position, source_hit.ng, source_hit.ns, -camera_ray.direction);
    LightSampleContext ctx{interaction.position, interaction.ng, interaction.ns};
    LightLiSample light = sample_light_li(
        lights, ctx, sampler_get_1d(sampler), sampler_get_1d(sampler),
        sampler_get_2d(sampler));
    if (!light.valid) {
        return;
    }
    DirectLightEstimate estimate =
        estimate_unoccluded_direct_light_li(source_material, interaction, light);

    if (!estimate.valid) {
        shadow_debug[pixel] += estimate.cos_light <= 0 ? vec3(1, 0.5f, 0)
                                                       : vec3(0.5f, 0, 1);
        return;
    }

    vec3 shadow_origin = offset_ray_origin_along_normal(
        source_hit.position, interaction.ng, estimate.wi);
    Ray shadow_ray = light.distance < INFINITY
        ? Ray(shadow_origin, estimate.wi, kRayEpsilon,
              light.distance - kRayEpsilon)
        : Ray(shadow_origin, estimate.wi, kRayEpsilon);
    ShadowTraceResult shadow = light.distance < INFINITY
        ? trace_shadow(scene, bvh, shadow_origin, light.position,
                       source_hit.primitive_ref)
        : trace_shadow_ray(scene, bvh, shadow_ray, source_hit.primitive_ref);
    int occluder_id = shadow.opaque_hit_found ? shadow.opaque_hit.triangle_id : -1;
    shadow_debug[pixel] += shadow_debug_classification_color(
        scene, source_triangle, source_material_id, interaction.ng, interaction.ns,
        shadow_ray, occluder_id, shadow.opaque_hit.t, shadow.opaque_hit.u,
        shadow.opaque_hit.v, shadow.passthrough_hits);
}

LuxInline void accumulate_cpu_aovs(const Scene& scene, const CpuBvh& bvh,
                                   const SceneLightSampler& lights,
                                   const Ray& camera_ray, SamplerState& debug_sampler,
                                   const CpuAovTargets& targets,
                                   int pixel, int max_depth) {
    if (!targets.enabled()) return;

    SurfaceHit hit;
    if (!intersect_scene_bvh(scene, bvh, camera_ray, hit)) {
        return;
    }

    if (targets.primary_debug) {
        (*targets.primary_debug)[pixel] +=
            primary_debug_classification_color(scene, hit, camera_ray);
    }
    if (targets.geometric_normal) {
        (*targets.geometric_normal)[pixel] +=
            primary_normal_debug_color(scene, hit, false);
    }
    if (targets.shading_normal) {
        (*targets.shading_normal)[pixel] +=
            primary_normal_debug_color(scene, hit, true);
    }
    if (targets.shadow_debug && max_depth > 0) {
        accumulate_shadow_debug_aov(
            scene, bvh, lights, hit, camera_ray, debug_sampler, *targets.shadow_debug, pixel);
    }
}

} // namespace cpu_path_tracer_detail

vec3 trace_path(const Scene& scene, const CpuBvh& bvh,
                const SceneLightSampler& lights, Ray ray, SamplerState& sampler,
                int max_depth) {
    vec3 radiance(0);
    vec3 throughput(1);
    PreviousSample previous_sample{
        0, PdfMeasure::None, SampleSource::Camera, 0, vec3(0)};
    int depth = 0;
    int passthrough_hits = 0;

    for (int step = 0; step <= max_depth; ++step) {
        SurfaceHit hit;
        if (!intersect_scene_bvh(scene, bvh, ray, hit)) {
            vec3 Le = infinite_lights_le(lights, ray);
            if (max_component(Le) > 0) {
                if (emission_uses_unweighted_hit(depth, previous_sample)) {
                    radiance += throughput * Le;
                } else if (emission_uses_mis(previous_sample)) {
                    Float light_pdf = infinite_lights_pdf_li(lights, ray.direction);
                    Float mis_weight = emission_mis_weight(previous_sample, light_pdf);
                    radiance += throughput * Le * mis_weight;
                }
            }
            break;
        }

        SceneTriangle scene_triangle = scene_triangle_view(scene, hit.triangle_id);
        const Material& material = scene.materials[scene_triangle.material_id];
        vec2 emission_uv = interpolate_triangle_uv(scene_triangle, hit.u, hit.v);
        vec3 Le = surface_light_le(
            lights, hit.triangle_id, emission_uv, hit.ng, -ray.direction);
        if (max_component(Le) > 0
            && emission_uses_unweighted_hit(depth, previous_sample)) {
            radiance += throughput * Le;
        } else if (max_component(Le) > 0 && emission_uses_mis(previous_sample)) {
            Float light_pdf = area_light_pdf_solid_angle(
                lights, hit.triangle_id, hit.position,
                previous_sample.reference_position);
            Float mis_weight = emission_mis_weight(previous_sample, light_pdf);
            radiance += throughput * Le * mis_weight;
        }

        if (is_passthrough_material(material)) {
            throughput = apply_passthrough(throughput, material);
            if (max_component(throughput) <= 0) break;
            ++passthrough_hits;
            ray = spawn_passthrough_ray(hit.position, ray.direction);
            continue;
        }

        if (!material.is_scattering()) break;
        if (step >= max_depth || depth >= max_depth) break;

        SurfaceInteraction interaction =
            make_surface_interaction(hit.position, hit.ng, hit.ns, -ray.direction);
        if (has_direct_lighting_lobe(material)) {
            radiance += throughput * cpu_path_tracer_detail::estimate_direct_light(
                scene, bvh, lights, interaction, material, hit.primitive_ref, sampler);
        }

        BSDFSample sample = sample_bsdf(
            material, interaction,
            sampler_get_2d(sampler));

        vec3 new_throughput;
        bool sample_delta = is_delta_bsdf(sample.flags);
        if (sample.pdf <= 0 || max_component(sample.weight) <= 0) break;
        new_throughput = throughput * sample.weight;

        if (should_apply_russian_roulette(depth)) {
            Float rr_prob = 1;
            if (!survives_russian_roulette(depth, new_throughput,
                                           sampler_get_1d(sampler), rr_prob)) {
                break;
            }
            new_throughput /= rr_prob;
        }

        throughput = new_throughput;
        previous_sample = PreviousSample{
            sample.pdf,
            sample.pdf_measure,
            SampleSource::BSDF,
            sample.flags,
            hit.position
        };

        ray = spawn_scatter_ray(hit.position, interaction.ng, sample.wi);
        ++depth;
    }

    return radiance;
}

vec3 trace_path(const Scene& scene, const CpuBvh& bvh,
                const LightDistribution& lights, Ray ray, SamplerState& sampler,
                int max_depth) {
    SceneLightSampler light_sampler = make_scene_light_sampler(scene, lights);
    return trace_path(scene, bvh, light_sampler, ray, sampler, max_depth);
}

void render_cpu_path_tracer(const Scene& scene, const CpuBvh& bvh,
                            const LightDistribution& lights,
                            Film& film,
                            const RenderSettings& settings,
                            RenderOutputs* outputs) {
    if (film.width <= 0 || film.height <= 0 || film.samples_per_pixel <= 0) {
        throw std::runtime_error("CPU path tracer requires a non-empty Film");
    }
    if (settings.max_depth < 0) {
        throw std::runtime_error("CPU path tracer requires max_depth >= 0");
    }

    cpu_path_tracer_detail::CpuAovTargets aov_targets =
        cpu_path_tracer_detail::make_cpu_aov_targets(outputs, film);
    SceneLightSampler light_sampler = make_scene_light_sampler(scene, lights);
    const Camera& camera = scene.camera;

    for (int y = 0; y < film.height; ++y) {
        std::printf("\rRendering row %d / %d", y + 1, film.height);
        std::fflush(stdout);
        for (int x = 0; x < film.width; ++x) {
            int pixel = y * film.width + x;
            vec3 color(0);
            for (int s = 0; s < film.samples_per_pixel; ++s) {
                SamplerState sampler = make_render_sampler(settings, pixel, s);
                Float camera_jitter_x = sampler_get_1d(sampler);
                Float camera_jitter_y = sampler_get_1d(sampler);
                Float u = (Float(x) + camera_jitter_x) / Float(film.width);
                Float v = (Float(film.height - 1 - y) + camera_jitter_y)
                    / Float(film.height);
                CameraSample camera_sample;
                camera_sample.p_film = vec2(u, v);
                camera_sample.p_lens = sampler_get_2d(sampler);
                camera_sample.time = sampler_get_1d(sampler);
                CameraRay generated = generate_camera_ray(camera, camera_sample);
                if (!generated.valid) continue;
                Ray camera_ray = generated.ray;
                if (aov_targets.enabled()) {
                    SamplerState debug_sampler = make_render_sampler(settings, pixel, s);
                    (void)sampler_get_1d(debug_sampler);
                    (void)sampler_get_1d(debug_sampler);
                    (void)sampler_get_2d(debug_sampler);
                    (void)sampler_get_1d(debug_sampler);
                    cpu_path_tracer_detail::accumulate_cpu_aovs(
                        scene, bvh, light_sampler, camera_ray, debug_sampler, aov_targets,
                        pixel, settings.max_depth);
                }
                color += trace_path(scene, bvh, light_sampler, camera_ray, sampler,
                                    settings.max_depth);
            }
            film[pixel] = color;
        }
    }
    std::printf("\n");
}

bool CpuPathTracer::requires_host_bvh() const {
    return true;
}

bool CpuPathTracer::requires_gpu_scene() const {
    return false;
}

void CpuPathTracer::render(const PreparedScene& scene, Film& film,
                           const RenderSettings& settings,
                           RenderOutputs* outputs) {
    render_cpu_path_tracer(
        scene.host_scene(), scene.host_bvh(), scene.light_distribution(),
        film, settings, outputs);
}
