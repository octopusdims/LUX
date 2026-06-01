#include "cpu_diagnostics_internal.h"

namespace lux_cpu_diagnostics_detail {

vec3 orientation_debug_color(const Scene& scene, const SurfaceHit& hit,
                             const Ray& ray, int target_material,
                             const std::vector<Float>& scores) {
    if (hit.triangle_id < 0) return vec3(0);
    SceneTriangle scene_triangle = scene_triangle_view(scene, hit.triangle_id);
    const Material& material = scene.materials[scene_triangle.material_id];
    if (material.is_emissive()) {
        return vec3(1, 0.85f, 0.1f);
    }
    if (scene_triangle.material_id != target_material) {
        return vec3(0.18f);
    }

    vec3 ng = triangle_normal(scene_triangle.triangle);
    bool front = dot(ng, ray.direction) < 0;
    bool inward = scores[hit.triangle_id] < Float(0);

    if (inward && !front) return vec3(1, 0, 1);
    if (inward) return vec3(1, 0.05f, 0.05f);
    if (!front) return vec3(0.05f, 0.25f, 1);
    return vec3(0.62f);
}

void write_orientation_debug_image(const Scene& scene, const CpuBvh& bvh,
                                   const Camera& camera,
                                   int width, int height,
                                   const RenderSettings& settings,
                                   const std::string& output_path) {
    std::vector<vec3> material_centers = compute_material_area_centroids(scene);
    std::vector<Float> scores = compute_radial_orientation_scores(scene, material_centers);
    int target_material = largest_non_emissive_material(scene);
    if (target_material < 0) {
        throw std::runtime_error("orientation AOV requires at least one non-emissive material");
    }

    std::vector<vec3> pixels(width * height, vec3(0));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int pixel_index = y * width + x;
            SamplerState sampler = make_render_sampler(settings, pixel_index, 0);
            CameraSample camera_sample;
            camera_sample.p_film = vec2(
                (Float(x) + sampler_get_1d(sampler)) / Float(width),
                (Float(height - 1 - y) + sampler_get_1d(sampler)) / Float(height));
            camera_sample.p_lens = sampler_get_2d(sampler);
            camera_sample.time = sampler_get_1d(sampler);
            Ray ray = generate_camera_ray(camera, camera_sample).ray;
            SurfaceHit hit;
            if (!intersect_scene_bvh(scene, bvh, ray, hit)) {
                continue;
            }
            pixels[y * width + x] = orientation_debug_color(
                scene, hit, ray, target_material, scores);
        }
    }

    write_debug_pixels(output_path, pixels, width, height, 1);
    std::printf("Wrote %s (CPU primary-hit orientation AOV, target material %d)\n",
                output_path.c_str(), target_material);
    std::printf("  colors: grey=normal target, red=inward radial, blue=backface, "
                "magenta=inward+backface, dark grey=other, yellow=light\n");
}

void write_cpu_orientation_aov_image(const Scene& scene, const CpuBvh& bvh,
                                     const Camera& camera,
                                     int width, int height,
                                     const RenderSettings& settings,
                                     const DebugRequest& request) {
    if (request.wants_aov(DebugAovOrientation)) {
        write_orientation_debug_image(
            scene, bvh, camera, width, height, settings,
            debug_aov_output_path(request.output_path, DebugAovOrientation));
    }
}

} // namespace lux_cpu_diagnostics_detail
