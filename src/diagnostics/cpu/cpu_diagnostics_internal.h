#pragma once

#ifndef LUX_DIAGNOSTICS_CPU_INTERNAL_H
#define LUX_DIAGNOSTICS_CPU_INTERNAL_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bvh/cpu_bvh.h"
#include "bvh/cpu_bvh_traversal.h"
#include "camera/camera.h"
#include "core/constants.h"
#include "film/pfm.h"
#include "film/ppm.h"
#include "geometry/triangle.h"
#include "integrator/path_transport.cuh"
#include "integrator/render_context.h"
#include "light/light_sampler.h"
#include "material/bsdf.h"
#include "scene/scene.h"
#include "scene/scene_intersection.h"
#include "scene/scene_light_sampler.h"
#include "diagnostics/diagnostic_config.h"

namespace lux_cpu_diagnostics_detail {
namespace {

bool is_pfm_path(const std::string& path) {
    return path.size() >= 4 && path.substr(path.size() - 4) == ".pfm";
}

void write_debug_pixels(const std::string& path, const std::vector<vec3>& pixels,
                        int width, int height, int samples_per_pixel) {
    if (is_pfm_path(path)) {
        write_pfm(path, pixels, width, height, samples_per_pixel);
    } else {
        write_debug_ppm_linear(path, pixels, width, height, samples_per_pixel);
    }
}

Float barycentric_w(Float u, Float v) {
    return Float(1) - u - v;
}

bool is_near_boundary_edge(const SceneTriangle& triangle, Float u, Float v) {
    constexpr Float kBoundaryBarycentricEpsilon = Float(0.02);
    if (triangle.boundary_edges == 0) return false;

    Float w = barycentric_w(u, v);
    bool near_v0_v1 = (triangle.boundary_edges & 1u) != 0 && v <= kBoundaryBarycentricEpsilon;
    bool near_v1_v2 = (triangle.boundary_edges & 2u) != 0 && w <= kBoundaryBarycentricEpsilon;
    bool near_v2_v0 = (triangle.boundary_edges & 4u) != 0 && u <= kBoundaryBarycentricEpsilon;
    return near_v0_v1 || near_v1_v2 || near_v2_v0;
}

bool same_position(const vec3& a, const vec3& b) {
    constexpr Float kSharedVertexEpsilon = Float(1e-7);
    return length2(a - b) <= kSharedVertexEpsilon * kSharedVertexEpsilon;
}

int count_shared_vertices(const Triangle& a, const Triangle& b) {
    const vec3 av[3] = {a.v0, a.v1, a.v2};
    const vec3 bv[3] = {b.v0, b.v1, b.v2};
    int count = 0;
    for (const vec3& va : av) {
        for (const vec3& vb : bv) {
            if (same_position(va, vb)) {
                ++count;
                break;
            }
        }
    }
    return count;
}

bool find_matching_vertex_index(const Triangle& triangle, const vec3& p, int& index) {
    const vec3 vertices[3] = {triangle.v0, triangle.v1, triangle.v2};
    for (int i = 0; i < 3; ++i) {
        if (same_position(vertices[i], p)) {
            index = i;
            return true;
        }
    }
    return false;
}

int shared_edge_orientation(const Triangle& source, int edge_index,
                            const Triangle& neighbor,
                            int& neighbor_from, int& neighbor_to) {
    const vec3 source_vertices[3] = {source.v0, source.v1, source.v2};
    int source_from = edge_index;
    int source_to = (edge_index + 1) % 3;

    if (!find_matching_vertex_index(neighbor, source_vertices[source_from],
                                    neighbor_from)
        || !find_matching_vertex_index(neighbor, source_vertices[source_to],
                                       neighbor_to)) {
        return 0;
    }

    if ((neighbor_from + 1) % 3 == neighbor_to) {
        return 1;
    }
    if ((neighbor_to + 1) % 3 == neighbor_from) {
        return -1;
    }
    return 0;
}

vec3 triangle_centroid(const Triangle& triangle) {
    return (triangle.v0 + triangle.v1 + triangle.v2) / Float(3);
}

vec3 material_area_centroid(const Scene& scene, int material_id) {
    vec3 weighted_centroid(0);
    Float total_area = 0;
    for (int triangle_id = 0; triangle_id < scene_triangle_count(scene); ++triangle_id) {
        SceneTriangle scene_triangle = scene_triangle_view(scene, triangle_id);
        if (scene_triangle.material_id != material_id) continue;
        Float area = triangle_area(scene_triangle.triangle);
        weighted_centroid += triangle_centroid(scene_triangle.triangle) * area;
        total_area += area;
    }
    return total_area > 0 ? weighted_centroid / total_area : vec3(0);
}

struct VertexKey {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;

    bool operator==(const VertexKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator<(const VertexKey& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

struct EdgeKey {
    VertexKey a;
    VertexKey b;

    bool operator==(const EdgeKey& other) const {
        return a == other.a && b == other.b;
    }
};

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& edge) const {
        auto mix = [](size_t h, uint32_t value) {
            return h ^ (size_t(value) + size_t(0x9e3779b9u) + (h << 6) + (h >> 2));
        };
        size_t h = 0;
        h = mix(h, edge.a.x);
        h = mix(h, edge.a.y);
        h = mix(h, edge.a.z);
        h = mix(h, edge.b.x);
        h = mix(h, edge.b.y);
        h = mix(h, edge.b.z);
        return h;
    }
};

uint32_t float_bits(Float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

VertexKey make_vertex_key(const vec3& p) {
    return VertexKey{float_bits(p.x), float_bits(p.y), float_bits(p.z)};
}

EdgeKey make_edge_key(const vec3& p0, const vec3& p1) {
    VertexKey a = make_vertex_key(p0);
    VertexKey b = make_vertex_key(p1);
    return b < a ? EdgeKey{b, a} : EdgeKey{a, b};
}

std::vector<vec3> compute_material_area_centroids(const Scene& scene) {
    std::vector<vec3> centers(scene.materials.size(), vec3(0));
    std::vector<Float> areas(scene.materials.size(), Float(0));
    for (int triangle_id = 0; triangle_id < scene_triangle_count(scene); ++triangle_id) {
        SceneTriangle scene_triangle = scene_triangle_view(scene, triangle_id);
        if (scene_triangle.material_id < 0
            || scene_triangle.material_id >= static_cast<int>(scene.materials.size())) {
            continue;
        }
        Float area = triangle_area(scene_triangle.triangle);
        centers[scene_triangle.material_id] += triangle_centroid(scene_triangle.triangle) * area;
        areas[scene_triangle.material_id] += area;
    }
    for (int i = 0; i < static_cast<int>(centers.size()); ++i) {
        if (areas[i] > 0) centers[i] /= areas[i];
    }
    return centers;
}

int largest_non_emissive_material(const Scene& scene) {
    std::vector<int> triangle_counts(scene.materials.size(), 0);
    for (int triangle_id = 0; triangle_id < scene_triangle_count(scene); ++triangle_id) {
        SceneTriangle scene_triangle = scene_triangle_view(scene, triangle_id);
        if (scene_triangle.material_id >= 0
            && scene_triangle.material_id < static_cast<int>(triangle_counts.size())) {
            ++triangle_counts[scene_triangle.material_id];
        }
    }

    int best_material = -1;
    int best_count = -1;
    for (int i = 0; i < static_cast<int>(scene.materials.size()); ++i) {
        if (scene.materials[i].is_emissive()) continue;
        if (triangle_counts[i] > best_count) {
            best_count = triangle_counts[i];
            best_material = i;
        }
    }
    return best_material;
}

std::vector<Float> compute_radial_orientation_scores(
        const Scene& scene,
        const std::vector<vec3>& material_centers) {
    std::vector<Float> scores(scene_triangle_count(scene), Float(0));
    for (int i = 0; i < scene_triangle_count(scene); ++i) {
        SceneTriangle scene_triangle = scene_triangle_view(scene, i);
        vec3 center = scene_triangle.material_id >= 0
            && scene_triangle.material_id < static_cast<int>(material_centers.size())
            ? material_centers[scene_triangle.material_id]
            : vec3(0);
        vec3 radial = triangle_centroid(scene_triangle.triangle) - center;
        scores[i] = length2(radial) > 0
            ? dot(triangle_normal(scene_triangle.triangle), normalize(radial))
            : Float(0);
    }
    return scores;
}

struct PrimaryProbeHit {
    int triangle_id = -1;
    Float t = std::numeric_limits<Float>::infinity();
    Float u = 0;
    Float v = 0;
};

std::vector<PrimaryProbeHit> collect_primary_hits_bruteforce(const Scene& scene,
                                                             const Ray& ray) {
    std::vector<PrimaryProbeHit> hits;
    for (int i = 0; i < scene_triangle_count(scene); ++i) {
        Float t, u, v;
        if (!intersect_triangle(ray, scene_triangle_geometry(scene, i), t, u, v)) continue;
        hits.push_back(PrimaryProbeHit{i, t, u, v});
    }
    std::sort(hits.begin(), hits.end(), [](const PrimaryProbeHit& a,
                                           const PrimaryProbeHit& b) {
        return a.t < b.t;
    });
    return hits;
}

bool intersect_scene_bruteforce_skip_ref(const Scene& scene, const Ray& ray,
                                         PrimitiveRef ignored_primitive,
                                         SurfaceHit& hit) {
    bool found = false;
    for (int i = 0; i < scene_triangle_count(scene); ++i) {
        if (should_skip_primitive(scene_primitive_ref(scene, i), ignored_primitive)) {
            continue;
        }
        Float t, u, v;
        if (!intersect_triangle(ray, scene_triangle_geometry(scene, i), t, u, v)) continue;
        if (t >= hit.t) continue;

        found = true;
        hit.t = t;
        hit.triangle_id = i;
        hit.primitive_ref = scene_primitive_ref(scene, i);
        hit.primitive_kind = hit.primitive_ref.kind;
        hit.u = u;
        hit.v = v;
        hit.position = ray.at(t);
        SceneTriangle scene_triangle = scene_triangle_view(scene, i);
        hit.ng = triangle_normal(scene_triangle.triangle);
        hit.ns = interpolate_triangle_normal(scene_triangle, u, v);
        if (dot(hit.ng, hit.ns) < 0) hit.ns = -hit.ns;
    }
    return found;
}

bool intersect_scene_bruteforce_skip(const Scene& scene, const Ray& ray, int ignored_triangle,
                                     SurfaceHit& hit) {
    return intersect_scene_bruteforce_skip_ref(
        scene, ray, scene_primitive_ref(scene, ignored_triangle), hit);
}

struct ShadowProbeResult {
    bool occluded = false;
    bool stack_overflow = false;
    vec3 transmittance = vec3(1);
    int passthrough_hits = 0;
    SurfaceHit opaque_hit;
};

ShadowProbeResult trace_shadow_bruteforce_skip(const Scene& scene, const Ray& input_ray,
                                               PrimitiveRef ignored_primitive) {
    ShadowProbeResult result;
    Ray ray = input_ray;
    PrimitiveRef skipped_primitive = ignored_primitive;
    while (true) {
        SurfaceHit hit;
        hit.t = ray.t_max;
        if (!intersect_scene_bruteforce_skip_ref(scene, ray, skipped_primitive, hit)) {
            return result;
        }

        SceneTriangle scene_triangle = scene_triangle_view(scene, hit.triangle_id);
        const Material& material = scene.materials[scene_triangle.material_id];
        if (!is_passthrough_material(material)) {
            result.occluded = true;
            result.transmittance = vec3(0);
            result.opaque_hit = hit;
            return result;
        }

        result.transmittance = result.transmittance * passthrough_transmittance(material);
        ++result.passthrough_hits;
        if (max_component(result.transmittance) <= 0) {
            return result;
        }

        Float advance = hit.t + kRayEpsilon;
        if (advance >= ray.t_max) {
            return result;
        }
        ray = Ray(ray.at(advance), ray.direction, kRayEpsilon,
                  ray.t_max - advance, ray.time);
        skipped_primitive = invalid_primitive_ref();
    }
    return result;
}

ShadowProbeResult trace_shadow_bvh_skip(const Scene& scene, const CpuBvh& bvh,
                                        const Ray& input_ray,
                                        PrimitiveRef ignored_primitive) {
    ShadowProbeResult result;
    Ray ray = input_ray;
    PrimitiveRef skipped_primitive = ignored_primitive;
    while (true) {
        SurfaceHit hit;
        hit.t = ray.t_max;
        bool stack_overflow = false;
        bool found = intersect_scene_bvh_skip_ref(
            scene, bvh, ray, skipped_primitive, hit, &stack_overflow);
        if (stack_overflow) {
            result.stack_overflow = true;
            result.transmittance = vec3(0);
            return result;
        }
        if (!found) {
            return result;
        }

        SceneTriangle scene_triangle = scene_triangle_view(scene, hit.triangle_id);
        const Material& material = scene.materials[scene_triangle.material_id];
        if (!is_passthrough_material(material)) {
            result.occluded = true;
            result.transmittance = vec3(0);
            result.opaque_hit = hit;
            return result;
        }

        result.transmittance = result.transmittance * passthrough_transmittance(material);
        ++result.passthrough_hits;
        if (max_component(result.transmittance) <= 0) {
            return result;
        }

        Float advance = hit.t + kRayEpsilon;
        if (advance >= ray.t_max) {
            return result;
        }
        ray = Ray(ray.at(advance), ray.direction, kRayEpsilon,
                  ray.t_max - advance, ray.time);
        skipped_primitive = invalid_primitive_ref();
    }
    return result;
}

bool same_shadow_probe_result(const ShadowProbeResult& a,
                              const ShadowProbeResult& b) {
    if (a.stack_overflow != b.stack_overflow
        || a.occluded != b.occluded
        || a.passthrough_hits != b.passthrough_hits) {
        return false;
    }
    if (length2(a.transmittance - b.transmittance) > Float(1e-8)) {
        return false;
    }
    if (!a.occluded) return true;
    return a.opaque_hit.triangle_id == b.opaque_hit.triangle_id
        && fabsf(a.opaque_hit.t - b.opaque_hit.t) <= Float(1e-5);
}

void print_shadow_hit_result(const char* label, const Scene& scene,
                             const SceneTriangle& source_triangle,
                             int source_material_id, const Ray& shadow_ray,
                             const ShadowProbeResult& result) {
    if (result.stack_overflow) {
        std::printf("  %s: stack_overflow\n", label);
        return;
    }
    if (!result.occluded) {
        if (result.passthrough_hits > 0) {
            std::printf("  %s: visible passthrough_hits=%d trans=(%.8g,%.8g,%.8g)\n",
                        label, result.passthrough_hits,
                        double(result.transmittance.x), double(result.transmittance.y),
                        double(result.transmittance.z));
        } else {
            std::printf("  %s: visible\n", label);
        }
        return;
    }

    const SurfaceHit& hit = result.opaque_hit;
    SceneTriangle occ_triangle = scene_triangle_view(scene, hit.triangle_id);
    vec3 occ_ng = triangle_normal(occ_triangle.triangle);
    bool same_material = occ_triangle.material_id == source_material_id;
    bool backface = dot(occ_ng, -shadow_ray.direction) <= 0;
    bool near_boundary = is_near_boundary_edge(occ_triangle, hit.u, hit.v);
    int shared_vertices = count_shared_vertices(
        source_triangle.triangle, occ_triangle.triangle);
    std::printf("  %s: hit tri=%d mat=%d t=%.8g t_eps=%.3g "
                "bary=(%.8g,%.8g,%.8g) same_mat=%d backface=%d "
                "boundary=0x%x near_boundary=%d shared_vertices=%d "
                "passthrough_hits=%d\n",
                label, hit.triangle_id, occ_triangle.material_id,
                double(hit.t), double(hit.t / kRayEpsilon),
                double(barycentric_w(hit.u, hit.v)), double(hit.u), double(hit.v),
                same_material ? 1 : 0, backface ? 1 : 0,
                static_cast<unsigned>(occ_triangle.boundary_edges),
                near_boundary ? 1 : 0, shared_vertices, result.passthrough_hits);
}

void print_primary_probe_hit(const Scene& scene, const Ray& camera_ray,
                             const PrimaryProbeHit& hit, int rank) {
    SceneTriangle scene_triangle = scene_triangle_view(scene, hit.triangle_id);
    vec3 raw_ng = triangle_normal(scene_triangle.triangle);
    vec3 raw_ns = interpolate_triangle_normal(scene_triangle, hit.u, hit.v);
    bool ns_flipped = dot(raw_ng, raw_ns) < 0;
    vec3 ns = ns_flipped ? -raw_ns : raw_ns;
    vec3 position = camera_ray.at(hit.t);
    vec3 view_direction = -camera_ray.direction;
    vec3 material_center = material_area_centroid(scene, scene_triangle.material_id);
    vec3 radial = triangle_centroid(scene_triangle.triangle) - material_center;
    Float radial_dot = length2(radial) > 0 ? dot(raw_ng, normalize(radial)) : 0;
    bool front = dot(raw_ng, camera_ray.direction) < 0;
    bool near_boundary = is_near_boundary_edge(scene_triangle, hit.u, hit.v);

    std::printf("  hit %02d: tri=%d mat=%d t=%.8g side=%s "
                "bary=(%.8g,%.8g,%.8g) has_vn=%d ns_flip=%d "
                "boundary=0x%x near_boundary=%d\n",
                rank, hit.triangle_id, scene_triangle.material_id, double(hit.t),
                front ? "front" : "back",
                double(barycentric_w(hit.u, hit.v)), double(hit.u), double(hit.v),
                has_vertex_normals(scene_triangle) ? 1 : 0, ns_flipped ? 1 : 0,
                static_cast<unsigned>(scene_triangle.boundary_edges),
                near_boundary ? 1 : 0);
    std::printf("          p=(%.8g,%.8g,%.8g) ng=(%.8g,%.8g,%.8g) "
                "ns=(%.8g,%.8g,%.8g)\n",
                double(position.x), double(position.y), double(position.z),
                double(raw_ng.x), double(raw_ng.y), double(raw_ng.z),
                double(ns.x), double(ns.y), double(ns.z));
    std::printf("          dots: ng_ray=%.8g ng_view=%.8g ns_view=%.8g "
                "dot_ns_ng=%.8g ng_material_radial=%.8g area=%.8g\n",
                double(dot(raw_ng, camera_ray.direction)),
                double(dot(raw_ng, view_direction)),
                double(dot(ns, view_direction)),
                double(dot(ns, raw_ng)), double(radial_dot),
                double(triangle_area(scene_triangle.triangle)));
}

bool validate_debug_pixel(int x, int y, int width, int height, const char* label) {
    if (x < width && y < height) return true;
    return false;
}

Ray sample_camera_probe_ray(const Camera& camera, int width, int height,
                            int x, int y, SamplerState& sampler) {
    CameraSample camera_sample;
    camera_sample.p_film = vec2(
        (Float(x) + sampler_get_1d(sampler)) / Float(width),
        (Float(height - 1 - y) + sampler_get_1d(sampler)) / Float(height));
    camera_sample.p_lens = sampler_get_2d(sampler);
    camera_sample.time = sampler_get_1d(sampler);
    return generate_camera_ray(camera, camera_sample).ray;
}


} // namespace

void print_primary_or_peel_probe(const Scene& scene, const CpuBvh& bvh,
                                 const Camera& camera,
                                 int width, int height,
                                 const RenderSettings& settings,
                                 const DebugRequest& request);
void print_path_probe(const Scene& scene, const CpuBvh& bvh,
                      const SceneLightSampler& lights, const Camera& camera,
                      int width, int height, const RenderSettings& settings,
                      const DebugRequest& request);
void print_nee_probe(const Scene& scene, const CpuBvh& bvh,
                     const SceneLightSampler& lights, const Camera& camera,
                     int width, int height, const RenderSettings& settings,
                     const DebugRequest& request);
void print_winding_probe(const Scene& scene, int triangle_id);
void print_orientation_audit(const Scene& scene);
void write_cpu_orientation_aov_image(const Scene& scene, const CpuBvh& bvh,
                                     const Camera& camera,
                                     int width, int height,
                                     const RenderSettings& settings,
                                     const DebugRequest& request);

} // namespace lux_cpu_diagnostics_detail

#endif // LUX_DIAGNOSTICS_CPU_INTERNAL_H
