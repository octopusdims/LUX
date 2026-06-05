#include "cpu_diagnostics_internal.h"

namespace lux_cpu_diagnostics_detail {

void print_camera_hits_probe(const Scene& scene, const CpuBvh& bvh,
                             const Camera& camera,
                             int width, int height,
                             const RenderSettings& settings,
                             const DebugRequest& request) {
    const char* label = "camera-hits probe";
    if (!validate_debug_pixel(request.pixel_x, request.pixel_y, width, height, label)) {
        throw std::runtime_error(std::string(label) + " pixel is outside the image");
    }

    int global_pixel = request.pixel_y * width + request.pixel_x;
    std::printf("Debug %s pixel=(%d,%d), global=%d, samples=%d, max_hits=%d\n",
                label, request.pixel_x, request.pixel_y, global_pixel,
                request.sample_count, request.max_hits);

    for (int s = 0; s < request.sample_count; ++s) {
        SamplerState sampler = make_render_sampler(settings, global_pixel, s);
        Ray camera_ray = sample_camera_probe_ray(
            camera, width, height, request.pixel_x, request.pixel_y, sampler);

        std::printf("sample %d: ray origin=(%.8g,%.8g,%.8g) "
                    "dir=(%.8g,%.8g,%.8g)\n",
                    s, double(camera_ray.origin.x), double(camera_ray.origin.y),
                    double(camera_ray.origin.z), double(camera_ray.direction.x),
                    double(camera_ray.direction.y), double(camera_ray.direction.z));

        std::vector<CameraHitProbeHit> hits = collect_camera_hits_bruteforce(
            scene, camera_ray);
        SurfaceHit bvh_hit;
        bool stack_overflow = false;
        bool bvh_found = intersect_scene_bvh(scene, bvh, camera_ray, bvh_hit, &stack_overflow);

        int front_count = 0;
        int back_count = 0;
        int first_front = -1;
        int first_back = -1;
        int first_material_id = hits.empty()
            ? -1 : scene_triangle_view(scene, hits[0].triangle_id).material_id;
        int same_material_front_count = 0;
        int same_material_back_count = 0;
        int first_same_material_front = -1;
        int first_same_material_back = -1;
        for (int i = 0; i < static_cast<int>(hits.size()); ++i) {
            SceneTriangle scene_triangle = scene_triangle_view(scene, hits[i].triangle_id);
            vec3 raw_ng = triangle_normal(scene_triangle.triangle);
            bool front = dot(raw_ng, camera_ray.direction) < 0;
            bool same_material = scene_triangle.material_id == first_material_id;
            if (front) {
                ++front_count;
                if (first_front < 0) first_front = i;
                if (same_material) {
                    ++same_material_front_count;
                    if (first_same_material_front < 0) first_same_material_front = i;
                }
            } else {
                ++back_count;
                if (first_back < 0) first_back = i;
                if (same_material) {
                    ++same_material_back_count;
                    if (first_same_material_back < 0) first_same_material_back = i;
                }
            }
        }

        std::printf("  hits=%zu front=%d back=%d first_front=%d first_back=%d\n",
                    hits.size(), front_count, back_count, first_front, first_back);
        std::printf("  first_hit_material=%d same_mat_front=%d same_mat_back=%d "
                    "first_same_mat_front=%d first_same_mat_back=%d\n",
                    first_material_id, same_material_front_count, same_material_back_count,
                    first_same_material_front, first_same_material_back);

        if (stack_overflow) {
            std::printf("  cpu_bvh: stack_overflow\n");
        } else if (!bvh_found) {
            std::printf("  cpu_bvh: miss\n");
        } else {
            std::printf("  cpu_bvh: tri=%d t=%.8g bary=(%.8g,%.8g,%.8g)\n",
                        bvh_hit.triangle_id, double(bvh_hit.t),
                        double(barycentric_w(bvh_hit.u, bvh_hit.v)),
                        double(bvh_hit.u), double(bvh_hit.v));
        }

        if (!hits.empty() && bvh_found && !stack_overflow) {
            bool match = hits[0].triangle_id == bvh_hit.triangle_id
                && fabsf(hits[0].t - bvh_hit.t) <= Float(1e-5);
            std::printf("  nearest_compare: %s\n", match ? "match" : "mismatch");
        }

        int printed_hits = std::min(request.max_hits, static_cast<int>(hits.size()));
        for (int i = 0; i < printed_hits; ++i) {
            print_camera_hit_probe_hit(scene, camera_ray, hits[i], i);
        }
        if (printed_hits < static_cast<int>(hits.size())) {
            std::printf("  truncated_hits=%zu (increase -debug-max-hits to print more)\n",
                        hits.size() - static_cast<size_t>(printed_hits));
        }
    }
}

void print_path_probe(const Scene& scene, const CpuBvh& bvh,
                      const SceneLightSampler& lights, const Camera& camera,
                      int width, int height, const RenderSettings& settings,
    const DebugRequest& request) {
    if (!validate_debug_pixel(request.pixel_x, request.pixel_y, width, height, "path probe")) {
        throw std::runtime_error("path probe pixel is outside the image");
    }

    int global_pixel = request.pixel_y * width + request.pixel_x;
    std::printf("Path probe pixel=(%d,%d), global=%d, samples=%d, depth=%d\n",
                request.pixel_x, request.pixel_y, global_pixel,
                request.sample_count, settings.max_depth);

    for (int s = 0; s < request.sample_count; ++s) {
        SamplerState sampler = make_render_sampler(settings, global_pixel, s);
        Ray ray = sample_camera_probe_ray(
            camera, width, height, request.pixel_x, request.pixel_y, sampler);
        vec3 throughput(1);
        PreviousSample previous_sample{
            0, PdfMeasure::None, SampleSource::Camera, 0, vec3(0)};
        int depth = 0;
        int passthrough_hits = 0;
        Float eta_scale = 1;
        int max_steps = settings.max_depth + 1;

        std::printf("sample %d:\n", s);
        for (int step = 0; step < max_steps; ++step) {
            std::printf("  step %d depth %d passthrough %d ray: origin=(%.8g,%.8g,%.8g) "
                        "dir=(%.8g,%.8g,%.8g) t=[%.8g,%.8g]\n",
                        step, depth, passthrough_hits,
                        double(ray.origin.x), double(ray.origin.y),
                        double(ray.origin.z), double(ray.direction.x),
                        double(ray.direction.y), double(ray.direction.z),
                        double(ray.t_min), double(ray.t_max));

            SurfaceHit hit;
            bool found = intersect_scene_bvh(scene, bvh, ray, hit);
            if (!found) {
                std::printf("    miss\n");
                break;
            }

            SceneTriangle scene_triangle = scene_triangle_view(scene, hit.triangle_id);
            const Material& material = scene.materials[scene_triangle.material_id];
            bool front = dot(hit.ng, ray.direction) < 0;
            bool near_boundary = is_near_boundary_edge(scene_triangle, hit.u, hit.v);
            Float ns_ng = dot(hit.ns, hit.ng);
            Float ng_ray = dot(hit.ng, ray.direction);
            Float ng_wo = dot(hit.ng, -ray.direction);
            Float ns_wo = dot(hit.ns, -ray.direction);

            std::printf("    hit: tri=%d mat=%d t=%.8g bary=(%.8g,%.8g,%.8g) "
                        "front=%d emissive=%d scattering=%d has_vn=%d boundary=0x%x "
                        "near_boundary=%d\n",
                        hit.triangle_id, scene_triangle.material_id, double(hit.t),
                        double(barycentric_w(hit.u, hit.v)), double(hit.u), double(hit.v),
                        front ? 1 : 0, material.is_emissive() ? 1 : 0,
                        material.is_scattering() ? 1 : 0,
                        has_vertex_normals(scene_triangle) ? 1 : 0,
                        static_cast<unsigned>(scene_triangle.boundary_edges),
                        near_boundary ? 1 : 0);
            std::printf("    p=(%.8g,%.8g,%.8g) ng=(%.8g,%.8g,%.8g) "
                        "ns=(%.8g,%.8g,%.8g)\n",
                        double(hit.position.x), double(hit.position.y),
                        double(hit.position.z), double(hit.ng.x), double(hit.ng.y),
                        double(hit.ng.z), double(hit.ns.x), double(hit.ns.y),
                        double(hit.ns.z));
            std::printf("    dots: ng_ray=%.8g ng_wo=%.8g ns_wo=%.8g "
                        "dot_ns_ng=%.8g throughput=(%.8g,%.8g,%.8g)\n",
                        double(ng_ray), double(ng_wo), double(ns_wo),
                        double(ns_ng), double(throughput.x), double(throughput.y),
                        double(throughput.z));

            vec3 Le = eval_emission(material, hit.ng, -ray.direction);
            if (max_component(Le) > 0
                && emission_uses_unweighted_hit(depth, previous_sample)) {
                vec3 emitted = throughput * Le;
                std::printf("    emission: add=(%.8g,%.8g,%.8g)\n",
                            double(emitted.x), double(emitted.y), double(emitted.z));
            } else if (max_component(Le) > 0 && emission_uses_mis(previous_sample)) {
                Float light_pdf = area_light_pdf_solid_angle(
                    lights, hit.triangle_id, hit.position,
                    previous_sample.reference_position);
                Float mis_weight = emission_mis_weight(previous_sample, light_pdf);
                vec3 emitted = throughput * Le * mis_weight;
                std::printf("    emission MIS: prev_pdf=%.8g light_pdf=%.8g "
                            "weight=%.8g ref=(%.8g,%.8g,%.8g) "
                            "add=(%.8g,%.8g,%.8g)\n",
                            double(previous_sample.pdf), double(light_pdf),
                            double(mis_weight),
                            double(previous_sample.reference_position.x),
                            double(previous_sample.reference_position.y),
                            double(previous_sample.reference_position.z),
                            double(emitted.x), double(emitted.y), double(emitted.z));
            }

            if (is_passthrough_material(material)) {
                throughput = apply_passthrough(throughput, material);
                ++passthrough_hits;
                std::printf("    passthrough: trans=(%.8g,%.8g,%.8g) "
                            "next throughput=(%.8g,%.8g,%.8g)\n",
                            double(passthrough_transmittance(material).x),
                            double(passthrough_transmittance(material).y),
                            double(passthrough_transmittance(material).z),
                            double(throughput.x), double(throughput.y),
                            double(throughput.z));
                if (max_component(throughput) <= 0) {
                    std::printf("    terminate: zero throughput\n");
                    break;
                }
                ray = spawn_passthrough_ray(hit.position, ray.direction);
                continue;
            }

            if (!material.is_scattering()) {
                std::printf("    terminate: non-scattering material\n");
                break;
            }
            if (step >= settings.max_depth || depth >= settings.max_depth) {
                std::printf("    terminate: max depth\n");
                break;
            }

            SurfaceInteraction interaction =
                make_surface_interaction(hit.position, hit.ng, hit.ns, -ray.direction);

            if (has_direct_lighting_lobe(material)) {
                LightSampleContext light_ctx{
                    interaction.position, interaction.ng, interaction.ns};
                Float light_select_u = sampler_get_1d(sampler);
                Float light_component_u = sampler_get_1d(sampler);
                vec2 light_u = sampler_get_2d(sampler);
                LightLiSample light = sample_light_li(
                    lights, light_ctx, light_select_u, light_component_u, light_u);
                if (!light.valid) {
                    std::printf("    nee reject=sample\n");
                } else {
                    DirectLightEstimate direct =
                        estimate_unoccluded_direct_light_li(material, interaction, light);
                    vec3 f = eval_bsdf(material, interaction, direct.wi);
                    bool same_hemi_ns =
                        same_hemisphere(interaction.ns, interaction.wo, direct.wi);

                    std::printf("    nee: light=%d primitive=%d point=(%.8g,%.8g,%.8g) "
                                "wi=(%.8g,%.8g,%.8g) dist=%.8g\n",
                                light.light_index, light.primitive_id, double(light.position.x),
                                double(light.position.y), double(light.position.z),
                                double(direct.wi.x), double(direct.wi.y),
                                double(direct.wi.z), double(light.distance));
                    std::printf("    nee dots: ns_wo=%.8g ns_wi=%.8g ng_wo=%.8g "
                                "ng_wi=%.8g same_hemi_ns=%d cos_abs=%.8g "
                                "cos_l=%.8g fmax=%.8g\n",
                                double(dot(interaction.ns, interaction.wo)),
                                double(dot(interaction.ns, direct.wi)),
                                double(dot(interaction.ng, interaction.wo)),
                                double(dot(interaction.ng, direct.wi)),
                                same_hemi_ns ? 1 : 0, double(direct.cos_surface),
                                double(direct.cos_light), double(max_component(f)));

                    if (!direct.valid) {
                        const char* reason = max_component(f) <= 0 || direct.cos_surface <= 0
                            ? "surface" : "light";
                        std::printf("    nee reject=%s\n", reason);
                    } else {
                        vec3 offset_normal = dot(direct.wi, interaction.ng) >= 0
                            ? interaction.ng : -interaction.ng;
                        vec3 shadow_origin = hit.position + offset_normal * kRayEpsilon;
                        Ray shadow_ray = light.distance < INFINITY
                            ? Ray(shadow_origin, direct.wi, kRayEpsilon,
                                  light.distance - kRayEpsilon)
                            : Ray(shadow_origin, direct.wi, kRayEpsilon);
                        std::printf("    shadow_ray: origin=(%.8g,%.8g,%.8g) "
                                    "dir=(%.8g,%.8g,%.8g) t=[%.8g,%.8g] "
                                    "unshadowed_nee=(%.8g,%.8g,%.8g)\n",
                                    double(shadow_ray.origin.x), double(shadow_ray.origin.y),
                                    double(shadow_ray.origin.z), double(shadow_ray.direction.x),
                                    double(shadow_ray.direction.y), double(shadow_ray.direction.z),
                                    double(shadow_ray.t_min), double(shadow_ray.t_max),
                                    double(direct.contribution.x),
                                    double(direct.contribution.y),
                                    double(direct.contribution.z));

                        ShadowProbeResult shadow_result =
                            trace_shadow_bvh_skip(
                                scene, bvh, shadow_ray, hit.primitive_ref);
                        print_shadow_hit_result(
                            "    shadow", scene, scene_triangle, scene_triangle.material_id,
                            shadow_ray, shadow_result);
                    }
                }
            }

            Float bsdf_u0 = sampler_get_1d(sampler);
            Float bsdf_u1 = sampler_get_1d(sampler);
            BSDFSample sample =
                sample_bsdf(material, interaction, vec2(bsdf_u0, bsdf_u1));
            Float cos_theta = abs_cosine(interaction.ns, sample.wi);
            std::printf("    bsdf sample: wi=(%.8g,%.8g,%.8g) fmax=%.8g weight_max=%.8g "
                        "pdf=%.8g cos_abs=%.8g delta=%d\n",
                        double(sample.wi.x), double(sample.wi.y), double(sample.wi.z),
                        double(max_component(sample.f)), double(max_component(sample.weight)),
                        double(sample.pdf),
                        double(cos_theta), is_delta_bsdf(sample.flags) ? 1 : 0);

            vec3 new_throughput;
            bool sample_delta = is_delta_bsdf(sample.flags);
            if (sample.pdf <= 0 || max_component(sample.weight) <= 0) {
                std::printf("    terminate: invalid %s sample\n",
                            sample_delta ? "delta" : "smooth");
                break;
            }
            new_throughput = throughput * sample.weight;
            Float new_eta_scale = eta_scale * bsdf_eta_scale_factor(sample);

            if (should_apply_russian_roulette(depth)) {
                Float rr_prob = 1;
                Float rr_u = sampler_get_1d(sampler);
                bool survived = survives_russian_roulette(
                    depth, new_throughput, new_eta_scale, rr_u, rr_prob);
                std::printf("    rr: u=%.8g p=%.8g survived=%d\n",
                            double(rr_u), double(rr_prob), survived ? 1 : 0);
                if (!survived) {
                    std::printf("    terminate: russian roulette\n");
                    break;
                }
                new_throughput /= rr_prob;
            }

            throughput = new_throughput;
            eta_scale = new_eta_scale;
            previous_sample = PreviousSample{
                sample.pdf,
                sample.pdf_measure,
                SampleSource::BSDF,
                sample.flags,
                hit.position
            };

            ray = spawn_scatter_ray(hit.position, interaction.ng, sample.wi);
            ++depth;
            std::printf("    next throughput=(%.8g,%.8g,%.8g)\n",
                        double(throughput.x), double(throughput.y),
                        double(throughput.z));
        }
    }
}

void print_nee_probe(const Scene& scene, const CpuBvh& bvh,
                     const SceneLightSampler& lights, const Camera& camera,
                     int width, int height, const RenderSettings& settings,
    const DebugRequest& request) {
    if (!validate_debug_pixel(request.pixel_x, request.pixel_y, width, height, "nee probe")) {
        throw std::runtime_error("nee probe pixel is outside the image");
    }

    int global_pixel = request.pixel_y * width + request.pixel_x;
    std::printf("NEE probe pixel=(%d,%d), global=%d, samples=%d\n",
                request.pixel_x, request.pixel_y, global_pixel, request.sample_count);
    if (lights.empty()) {
        std::printf("Scene has no lights to probe\n");
        return;
    }

    for (int s = 0; s < request.sample_count; ++s) {
        SamplerState sampler = make_render_sampler(settings, global_pixel, s);
        Ray camera_ray = sample_camera_probe_ray(
            camera, width, height, request.pixel_x, request.pixel_y, sampler);

        SurfaceHit source_hit;
        if (!intersect_scene_bvh(scene, bvh, camera_ray, source_hit)) {
            std::printf("sample %d: camera miss\n", s);
            continue;
        }

        SceneTriangle source_triangle = scene_triangle_view(scene, source_hit.triangle_id);
        int source_material_id = source_triangle.material_id;
        const Material& source_material = scene.materials[source_material_id];
        SurfaceInteraction interaction = make_surface_interaction(
            source_hit.position, source_hit.ng, source_hit.ns, -camera_ray.direction);

        LightSampleContext light_ctx{interaction.position, interaction.ng, interaction.ns};
        Float light_select_u = sampler_get_1d(sampler);
        Float light_component_u = sampler_get_1d(sampler);
        vec2 light_u = sampler_get_2d(sampler);
        LightLiSample light = sample_light_li(
            lights, light_ctx, light_select_u, light_component_u, light_u);
        if (!light.valid) {
            std::printf("sample %d: failed to sample light\n", s);
            continue;
        }
        DirectLightEstimate direct =
            estimate_unoccluded_direct_light_li(source_material, interaction, light);
        vec3 f = eval_bsdf(source_material, interaction, direct.wi);
        Float ns_wo = dot(interaction.ns, interaction.wo);
        Float ns_wi = dot(interaction.ns, direct.wi);
        Float ng_wo = dot(interaction.ng, interaction.wo);
        Float ng_wi = dot(interaction.ng, direct.wi);
        bool same_hemi_ns = same_hemisphere(interaction.ns, interaction.wo, direct.wi);
        bool source_near_boundary = is_near_boundary_edge(
            source_triangle, source_hit.u, source_hit.v);

        std::printf("sample %d:\n", s);
        std::printf("  src: tri=%d mat=%d t=%.8g bary=(%.8g,%.8g,%.8g) "
                    "has_vn=%d boundary=0x%x near_boundary=%d\n",
                    source_hit.triangle_id, source_material_id, double(source_hit.t),
                    double(barycentric_w(source_hit.u, source_hit.v)),
                    double(source_hit.u), double(source_hit.v),
                    has_vertex_normals(source_triangle) ? 1 : 0,
                    static_cast<unsigned>(source_triangle.boundary_edges),
                    source_near_boundary ? 1 : 0);
        std::printf("  dots: ns_wo=%.8g ns_wi=%.8g ng_wo=%.8g ng_wi=%.8g "
                    "same_hemi_ns=%d cos_abs=%.8g cos_l=%.8g fmax=%.8g\n",
                    double(ns_wo), double(ns_wi), double(ng_wo), double(ng_wi),
                    same_hemi_ns ? 1 : 0, double(direct.cos_surface),
                    double(direct.cos_light), double(max_component(f)));
        std::printf("  light: index=%d primitive=%d point=(%.8g,%.8g,%.8g) "
                    "normal=(%.8g,%.8g,%.8g) dist=%.8g pdf=%.8g\n",
                    light.light_index, light.primitive_id,
                    double(light.position.x), double(light.position.y),
                    double(light.position.z), double(light.normal.x), double(light.normal.y),
                    double(light.normal.z), double(light.distance), double(light.pdf));

        if (!direct.valid) {
            const char* reason = max_component(f) <= 0 || direct.cos_surface <= 0
                ? "surface" : "light";
            std::printf("  reject=%s\n", reason);
            continue;
        }

        vec3 offset_normal = dot(direct.wi, interaction.ng) >= 0
            ? interaction.ng : -interaction.ng;
        vec3 shadow_origin = source_hit.position + offset_normal * kRayEpsilon;
        Ray shadow_ray = light.distance < INFINITY
            ? Ray(shadow_origin, direct.wi, kRayEpsilon,
                  light.distance - kRayEpsilon)
            : Ray(shadow_origin, direct.wi, kRayEpsilon);
        std::printf("  shadow_ray: origin=(%.8g,%.8g,%.8g) dir=(%.8g,%.8g,%.8g) "
                    "t=[%.8g,%.8g] dist=%.8g\n",
                    double(shadow_ray.origin.x), double(shadow_ray.origin.y),
                    double(shadow_ray.origin.z), double(shadow_ray.direction.x),
                    double(shadow_ray.direction.y), double(shadow_ray.direction.z),
                    double(shadow_ray.t_min), double(shadow_ray.t_max),
                    double(light.distance));

        ShadowProbeResult brute_result =
            trace_shadow_bruteforce_skip(
                scene, shadow_ray, source_hit.primitive_ref);
        ShadowProbeResult bvh_result =
            trace_shadow_bvh_skip(
                scene, bvh, shadow_ray, source_hit.primitive_ref);

        print_shadow_hit_result(
            "bruteforce", scene, source_triangle, source_material_id,
            shadow_ray, brute_result);
        print_shadow_hit_result(
            "cpu_bvh", scene, source_triangle, source_material_id,
            shadow_ray, bvh_result);
        if (!same_shadow_probe_result(brute_result, bvh_result)) {
            std::printf("  compare: mismatch\n");
        } else {
            std::printf("  compare: match\n");
        }
    }
}


} // namespace lux_cpu_diagnostics_detail
