#include "integrator/wavefront_kernels.cuh"

#include "core/constants.h"
#include "core/cuda_check.cuh"
#include "geometry/triangle.h"
#include "integrator/path_transport.cuh"
#include "integrator/wavefront_direct_lighting.cuh"
#include "light/light_sampler.h"
#include "material/bsdf.h"
#include "scene/gpu_scene_bvh.cuh"

namespace {

int grid_size(int count, int block_size = WavefrontRuntime::kBlockSize) {
    return (count + block_size - 1) / block_size;
}

struct DeviceShadowTraceResult {
    vec3 transmittance = vec3(1);
    int passthrough_hits = 0;
    int opaque_primitive_id = -1;
    PrimitiveRef opaque_primitive_ref;
    Float opaque_hit_t = 0;
    Float opaque_hit_u = 0;
    Float opaque_hit_v = 0;
};

LuxDeviceInline DeviceShadowTraceResult trace_shadow(
        GpuBvhView bvh,
        GpuScene scene,
        const Ray& input_ray,
        PrimitiveRef source_primitive_ref) {
    DeviceShadowTraceResult result;
    Ray ray = input_ray;
    PrimitiveRef ignored_primitive = source_primitive_ref;

    while (true) {
        int primitive_id = -1;
        PrimitiveRef primitive_ref;
        Float hit_t = 0;
        Float hit_u = 0;
        Float hit_v = 0;
        bool hit = intersect_scene_bvh_device(
            bvh, scene, ray, primitive_id, primitive_ref, hit_t, hit_u, hit_v,
            ignored_primitive);
        if (!hit) {
            return result;
        }

        int material_id = gpu_scene_material_id(scene, primitive_ref);
        if (material_id < 0 || material_id >= scene.material_count) {
            result.transmittance = vec3(0);
            result.opaque_primitive_id = primitive_id;
            result.opaque_primitive_ref = primitive_ref;
            return result;
        }
        const Material& material = scene.materials[material_id];
        if (!is_passthrough_material(material)) {
            result.transmittance = vec3(0);
            result.opaque_primitive_id = primitive_id;
            result.opaque_primitive_ref = primitive_ref;
            result.opaque_hit_t = hit_t;
            result.opaque_hit_u = hit_u;
            result.opaque_hit_v = hit_v;
            return result;
        }

        result.transmittance = result.transmittance * passthrough_transmittance(material);
        ++result.passthrough_hits;
        if (max_component(result.transmittance) <= 0) {
            return result;
        }

        Float advance = hit_t + kRayEpsilon;
        if (advance >= ray.t_max) {
            return result;
        }
        ray = Ray(ray.at(advance), ray.direction, kRayEpsilon,
                  ray.t_max - advance, ray.time);
        ignored_primitive = invalid_primitive_ref();
    }
    return result;
}

template <bool LogPaths>
LuxDeviceInline void record_surface_vertex(
        PathLogView path_log,
        PathStateView paths,
        int path_id,
        int depth,
        int triangle_id,
        PrimitiveRef primitive_ref,
        int material_id,
        const vec3& position,
        const SurfaceInteraction& interaction,
        const vec3& wi,
        const vec3& throughput_before,
        const vec3& throughput_after,
        Float bsdf_pdf,
        PdfMeasure bsdf_pdf_measure,
        Float guide_pdf,
        PdfMeasure guide_pdf_measure,
        Float light_pdf,
        PdfMeasure light_pdf_measure,
        SampleSource sample_source,
        unsigned bsdf_flags,
        unsigned termination_flags,
        unsigned event_flags) {
    if constexpr (!LogPaths) {
        return;
    }
    if (!path_log_enabled(path_log)) return;

    PathLogRecord record;
    record.pixel_index = paths.pixel_index[path_id];
    record.sample_index = path_log.sample_index;
    record.depth = depth;
    record.triangle_id = triangle_id;
    record.primitive_ref = primitive_ref;
    record.material_id = material_id;
    record.position = position;
    record.ng = interaction.ng;
    record.ns = interaction.ns;
    record.wo = interaction.wo;
    record.wi = wi;
    record.throughput_before = throughput_before;
    record.throughput_after = throughput_after;
    record.bsdf_pdf = bsdf_pdf;
    record.guide_pdf = guide_pdf;
    record.light_pdf = light_pdf;
    record.bsdf_pdf_measure = bsdf_pdf_measure;
    record.guide_pdf_measure = guide_pdf_measure;
    record.light_pdf_measure = light_pdf_measure;
    record.sample_source = sample_source;
    record.bsdf_flags = bsdf_flags;
    record.termination_flags = termination_flags;
    record.event_flags = event_flags;
    record_path_vertex(path_log, path_id, record);
}

struct SurfaceStageLogState {
    vec3 wi = vec3(0);
    Float bsdf_pdf = 0;
    Float guide_pdf = 0;
    Float light_pdf = 0;
    PdfMeasure bsdf_measure = PdfMeasure::None;
    PdfMeasure guide_measure = PdfMeasure::None;
    PdfMeasure light_measure = PdfMeasure::None;
    SampleSource source = SampleSource::None;
    unsigned bsdf_flags = 0;
    unsigned events = PathLogEventSurfaceHit;
};

LuxDeviceInline unsigned apply_surface_emission_stage(
        PathStateView paths,
        GpuScene scene,
        int path_id,
        const HitWorkItem& item,
        const PreviousSample& previous_sample,
        const vec3& geometric_normal) {
    int triangle_id = item.triangle_id;
    GpuSceneTriangle surface = gpu_scene_triangle(scene, item.primitive_ref);
    vec2 emission_uv = gpu_scene_interpolate_triangle_uv(surface, item.u, item.v);
    vec3 Le = surface_light_le(
        scene, item.primitive_ref, triangle_id, emission_uv, geometric_normal, item.wo);
    if (max_component(Le) <= 0) return 0;

    if (emission_uses_unweighted_hit(paths.depth[path_id], previous_sample)) {
        paths.radiance[path_id] += paths.throughput[path_id] * Le;
        return PathLogEventEmission;
    }

    if (emission_uses_mis(previous_sample)) {
        Float light_pdf = area_light_pdf_solid_angle(
            scene, item.primitive_ref, triangle_id, item.position,
            previous_sample.reference_position);
        Float mis_weight = emission_mis_weight(previous_sample, light_pdf);
        paths.radiance[path_id] += paths.throughput[path_id] * Le * mis_weight;
        return PathLogEventEmission;
    }

    return 0;
}

template <bool LogPaths>
LuxDeviceInline void record_surface_stage(
        PathLogView path_log,
        PathStateView paths,
        int path_id,
        int depth,
        int triangle_id,
        int material_id,
        const HitWorkItem& item,
        const SurfaceInteraction& interaction,
        const SurfaceStageLogState& log,
        const vec3& throughput_before,
        const vec3& throughput_after,
        unsigned termination_flags) {
    record_surface_vertex<LogPaths>(
        path_log, paths, path_id, depth, triangle_id, item.primitive_ref, material_id,
        item.position, interaction, log.wi, throughput_before, throughput_after,
        log.bsdf_pdf, log.bsdf_measure, log.guide_pdf, log.guide_measure,
        log.light_pdf, log.light_measure, log.source, log.bsdf_flags,
        termination_flags, log.events);
}

template <bool LogPaths>
LuxDeviceInline void terminate_surface_stage(
        WorkQueue<TerminatedPathItem>& terminate_queue,
        PathLogView path_log,
        PathStateView paths,
        int path_id,
        int depth,
        int triangle_id,
        int material_id,
        const HitWorkItem& item,
        const SurfaceInteraction& interaction,
        const SurfaceStageLogState& log,
        const vec3& throughput_before,
        const vec3& throughput_after,
        unsigned termination_flags) {
    SurfaceStageLogState terminated_log = log;
    terminated_log.events |= PathLogEventTerminated;
    record_surface_stage<LogPaths>(
        path_log, paths, path_id, depth, triangle_id, material_id, item, interaction,
        terminated_log, throughput_before, throughput_after, termination_flags);
    terminate_queue.push(TerminatedPathItem{path_id});
}

} // namespace

__global__ void generate_camera_rays_kernel(PathStateView paths, WorkQueue<RayWorkItem> ray_queue,
                                            const Camera camera,
                                            RenderSettings settings,
                                            int width, int height, int spp, int sample_index,
                                            int batch_pixel_offset, int batch_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) *ray_queue.size = batch_size;
    if (tid >= batch_size) return;

    int global_pixel = batch_pixel_offset + tid;
    if (global_pixel >= width * height) return;

    int px = global_pixel % width;
    int py = global_pixel / width;

    int path_id = tid;
    paths.throughput[path_id] = vec3(1);
    paths.radiance[path_id] = vec3(0);
    paths.pixel_index[path_id] = global_pixel;
    paths.depth[path_id] = 0;
    paths.passthrough_depth[path_id] = 0;
    clear_last_sample(paths, path_id, SampleSource::Camera);
    paths.samplers[path_id] = make_render_sampler(settings, global_pixel, sample_index);

    Float u = (Float(px) + sampler_get_1d(paths.samplers[path_id])) / Float(width);
    Float v = (Float(height - 1 - py) + sampler_get_1d(paths.samplers[path_id]))
        / Float(height);
    CameraSample camera_sample;
    camera_sample.p_film = vec2(u, v);
    camera_sample.p_lens = sampler_get_2d(paths.samplers[path_id]);
    camera_sample.time = sampler_get_1d(paths.samplers[path_id]);
    CameraRay generated = generate_camera_ray(camera, camera_sample);
    Ray ray = generated.ray;

    ray_queue.write(tid, RayWorkItem{path_id, ray});
}

__global__ void intersect_bvh_kernel(WorkQueue<RayWorkItem> ray_queue,
                                     MaterialHitQueueViews hit_queues,
                                     PathStateView paths,
                                     GpuBvhView bvh,
                                     GpuScene scene,
                                     WavefrontDebugViews debug,
                                     int width, int height) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= *ray_queue.size) return;

    RayWorkItem item = ray_queue.read(tid);
    int path_id = item.path_id;
    const Ray& ray = item.ray;

    if ((debug.geometric_normal || debug.shading_normal)
        && paths.depth[path_id] == 0
        && paths.passthrough_depth[path_id] == 0) {
        int debug_primitive_id;
        PrimitiveRef debug_primitive_ref;
        Float debug_hit_t, debug_hit_u, debug_hit_v;
        bool debug_found = intersect_scene_bvh_device(
            bvh, scene, ray, debug_primitive_id, debug_primitive_ref, debug_hit_t,
            debug_hit_u, debug_hit_v);
        int pixel = paths.pixel_index[path_id];
        int debug_id = debug_found ? debug_primitive_id : -1;
        if (debug.geometric_normal) {
            debug.geometric_normal[pixel] += lux_wavefront_debug::primary_normal_debug_color(
                scene, debug_primitive_ref, debug_id, debug_hit_u, debug_hit_v, false);
        }
        if (debug.shading_normal) {
            debug.shading_normal[pixel] += lux_wavefront_debug::primary_normal_debug_color(
                scene, debug_primitive_ref, debug_id, debug_hit_u, debug_hit_v, true);
        }
    }

    int primitive_id = -1;
    PrimitiveRef primitive_ref;
    Float hit_t = 0;
    Float hit_u = 0;
    Float hit_v = 0;
    bool found = intersect_scene_bvh_device(
        bvh, scene, ray, primitive_id, primitive_ref, hit_t, hit_u, hit_v);

    if (found) {
        GpuSceneTriangle surface = gpu_scene_triangle(scene, primitive_ref);
        if (!surface.valid) {
            return;
        }
        Triangle tri = surface.triangle;
        vec3 position = ray.at(hit_t);
        vec3 ng = triangle_normal(tri);
        HitWorkItem hit_item{
            path_id, primitive_id, primitive_ref,
            ray.origin, position, ng, ng, -ray.direction,
            hit_u, hit_v};
        int material_id = surface.material_id;
        if (material_id < 0 || material_id >= scene.material_count) {
            return;
        }
        MaterialType material_type = scene.materials[material_id].type;
        hit_queues.push(material_type, hit_item);
    } else {
        vec3 Le = infinite_lights_le(scene, ray);
        if (max_component(Le) > 0) {
            PreviousSample previous_sample{
                paths.last_sample_pdf[path_id],
                paths.last_sample_measure[path_id],
                paths.last_sample_source[path_id],
                paths.last_sample_flags[path_id],
                paths.last_sample_reference_position[path_id]
            };
            if (emission_uses_unweighted_hit(paths.depth[path_id], previous_sample)) {
                paths.radiance[path_id] += paths.throughput[path_id] * Le;
            } else if (emission_uses_mis(previous_sample)) {
                Float light_pdf = infinite_lights_pdf_li(scene, ray.direction);
                Float mis_weight = emission_mis_weight(previous_sample, light_pdf);
                paths.radiance[path_id] += paths.throughput[path_id] * Le * mis_weight;
            }
        }
    }
}

template <MaterialType MaterialKind, bool LogPaths>
__global__ void evaluate_hit_kernel(WorkQueue<HitWorkItem> hit_queue,
                                    WorkQueue<RayWorkItem> next_ray_queue,
                                    WorkQueue<ShadowRayWorkItem> shadow_queue,
                                    WorkQueue<TerminatedPathItem> terminate_queue,
                                    PathStateView paths,
                                    GpuScene scene,
                                    int width, int height,
                                    int wavefront_depth,
                                    int max_depth,
                                    WavefrontDebugViews debug,
                                    PathLogView path_log) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= *hit_queue.size) return;

    HitWorkItem item = hit_queue.read(tid);
    int path_id = item.path_id;
    int triangle_id = item.triangle_id;

    GpuSceneTriangle surface = gpu_scene_triangle(scene, item.primitive_ref);
    int material_id = surface.material_id;
    if (!surface.valid || material_id < 0 || material_id >= scene.material_count) {
        terminate_queue.push(TerminatedPathItem{path_id});
        return;
    }
    const Material& material = scene.materials[material_id];

    vec3 geometric_normal = item.ng;
    SurfaceInteraction interaction = make_surface_interaction(
        item.position, geometric_normal,
        gpu_scene_interpolate_triangle_normal(surface, item.u, item.v),
        item.wo);
    PreviousSample previous_sample{
        paths.last_sample_pdf[path_id],
        paths.last_sample_measure[path_id],
        paths.last_sample_source[path_id],
        paths.last_sample_flags[path_id],
        paths.last_sample_reference_position[path_id]
    };

    vec3 throughput_before = paths.throughput[path_id];
    vec3 throughput_after = throughput_before;
    SurfaceStageLogState log;
    log.events |= apply_surface_emission_stage(
        paths, scene, path_id, item, previous_sample, geometric_normal);

    if (is_passthrough_material(material)) {
        log.events |= PathLogEventPassthrough;
        vec3 wi = normalize(-interaction.wo);
        log.wi = wi;

        throughput_after = apply_passthrough(throughput_before, material);
        log.bsdf_pdf = Float(1);
        log.bsdf_measure = PdfMeasure::Discrete;
        log.source = SampleSource::BSDF;
        log.bsdf_flags = BsdfFlagDeltaTransmission | BsdfFlagTransmission;
        log.events |= PathLogEventDelta;
        paths.throughput[path_id] = throughput_after;
        paths.passthrough_depth[path_id] += 1;

        if (max_component(throughput_after) <= 0) {
            terminate_surface_stage<LogPaths>(
                terminate_queue, path_log, paths, path_id, paths.depth[path_id],
                triangle_id, material_id, item, interaction, log, throughput_before,
                throughput_after, PathLogTerminationZeroThroughput);
            return;
        }

        if (wavefront_depth >= max_depth) {
            terminate_surface_stage<LogPaths>(
                terminate_queue, path_log, paths, path_id, paths.depth[path_id],
                triangle_id, material_id, item, interaction, log, throughput_before,
                throughput_after, PathLogTerminationMaxDepth);
            return;
        }

        log.events |= PathLogEventSpawnedRay;
        record_surface_stage<LogPaths>(
            path_log, paths, path_id, paths.depth[path_id], triangle_id, material_id,
            item, interaction, log, throughput_before, throughput_after,
            PathLogTerminationNone);
        next_ray_queue.push(RayWorkItem{path_id, spawn_passthrough_ray(item.position, wi)});
        return;
    }

    if (!material.is_scattering()) {
        terminate_surface_stage<LogPaths>(
            terminate_queue, path_log, paths, path_id, paths.depth[path_id],
            triangle_id, material_id, item, interaction, log, throughput_before,
            throughput_after, PathLogTerminationNonScattering);
        return;
    }

    if (wavefront_depth >= max_depth || paths.depth[path_id] >= max_depth) {
        terminate_surface_stage<LogPaths>(
            terminate_queue, path_log, paths, path_id, paths.depth[path_id],
            triangle_id, material_id, item, interaction, log, throughput_before,
            throughput_after, PathLogTerminationMaxDepth);
        return;
    }

    if (has_direct_lighting_lobe(material)
        && scene.generic_light_count > 0 && scene.total_light_power > 0) {
        SamplerState& sampler = paths.samplers[path_id];
        LightLiSample light;
        bool sampled_light = false;
        DirectLightEstimate direct =
            lux_wavefront_direct::sample_direct_light<MaterialKind>(
                scene, material, interaction, sampler, light, sampled_light);
        if (sampled_light) {
            log.events |= PathLogEventDirectLightSampled;
            if (direct.pdf_light_solid_angle > 0) {
                log.light_pdf = direct.pdf_light_solid_angle;
                log.light_measure = PdfMeasure::SolidAngle;
            }
        }

        if (debug.shadow && paths.depth[path_id] == 0
            && !direct.valid) {
            int pixel = paths.pixel_index[path_id];
            if (direct.cos_light <= 0) {
                debug.shadow[pixel] += vec3(1, 0.5f, 0);
            } else {
                debug.shadow[pixel] += vec3(0.5f, 0, 1);
            }
        }

        if (direct.valid) {
            vec3 nee = direct.contribution;
            log.events |= PathLogEventDirectLightValid;
            log.light_pdf = direct.pdf_light_solid_angle;
            log.light_measure = PdfMeasure::SolidAngle;

            Float mis_weight = direct_light_mis_weight(direct);
            nee *= mis_weight;

            vec3 contribution = paths.throughput[path_id] * nee;
            shadow_queue.push(lux_wavefront_direct::make_shadow_work_item_li(
                path_id, triangle_id, item.primitive_ref,
                material_id, paths.depth[path_id], item.position,
                interaction, light, direct, contribution));
        }
    }

    SamplerState& sampler = paths.samplers[path_id];
    BSDFSample sample = sample_bsdf_typed<MaterialKind>(material, interaction,
        sampler_get_2d(sampler));
    log.wi = sample.wi;
    log.bsdf_flags = sample.flags;
    log.events |= PathLogEventBsdfSampled;
    bool sample_delta = is_delta_bsdf(sample.flags);
    if (sample_delta) {
        log.events |= PathLogEventDelta;
    }
    if (sample.pdf > 0) {
        log.bsdf_pdf = sample.pdf;
        log.bsdf_measure = sample.pdf_measure;
    }

    vec3 new_throughput;
    if (sample.pdf <= 0 || max_component(sample.weight) <= 0) {
        terminate_surface_stage<LogPaths>(
            terminate_queue, path_log, paths, path_id, paths.depth[path_id],
            triangle_id, material_id, item, interaction, log, throughput_before,
            throughput_after, PathLogTerminationInvalidBsdf);
        return;
    }
    new_throughput = paths.throughput[path_id] * sample.weight;
    log.source = SampleSource::BSDF;
    throughput_after = new_throughput;

    if (should_apply_russian_roulette(paths.depth[path_id])) {
        Float rr_prob = 1;
        if (!survives_russian_roulette(paths.depth[path_id], new_throughput,
                                       sampler_get_1d(sampler), rr_prob)) {
            terminate_surface_stage<LogPaths>(
                terminate_queue, path_log, paths, path_id, paths.depth[path_id],
                triangle_id, material_id, item, interaction, log, throughput_before,
                throughput_after, PathLogTerminationRussianRoulette);
            return;
        }
        new_throughput /= rr_prob;
        throughput_after = new_throughput;
    }

    paths.throughput[path_id] = new_throughput;
    store_last_sample(paths, path_id, sample.pdf, sample.pdf_measure,
                      SampleSource::BSDF, sample.flags, item.position);
    paths.depth[path_id] += 1;

    log.events |= PathLogEventSpawnedRay;
    record_surface_stage<LogPaths>(
        path_log, paths, path_id, paths.depth[path_id] - 1, triangle_id, material_id,
        item, interaction, log, throughput_before, throughput_after,
        PathLogTerminationNone);
    next_ray_queue.push(RayWorkItem{
        path_id, spawn_scatter_ray(item.position, interaction.ng, sample.wi)});
}

__global__ void flush_batch_paths_kernel(
        PathStateView paths,
        vec3* output_pixels,
        int batch_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size) return;

    int pixel = paths.pixel_index[tid];
    output_pixels[pixel] += paths.radiance[tid];
}

__global__ void trace_shadow_rays_kernel(WorkQueue<ShadowRayWorkItem> shadow_queue,
                                         PathStateView paths,
                                         GpuBvhView bvh,
                                         GpuScene scene,
                                         WavefrontDebugViews debug) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= *shadow_queue.size) return;

    ShadowRayWorkItem item = shadow_queue.read(tid);
    int path_id = item.path_id;
    const Ray& shadow_ray = item.shadow_ray;

    DeviceShadowTraceResult shadow = trace_shadow(
        bvh, scene, shadow_ray, item.source_primitive_ref);

    if (debug.shadow && item.source_depth == 0) {
        int pixel = paths.pixel_index[path_id];
        debug.shadow[pixel] += lux_wavefront_debug::shadow_debug_classification_color(
            item, shadow_ray, scene, shadow.opaque_primitive_id,
            shadow.opaque_primitive_ref, shadow.opaque_hit_t, shadow.opaque_hit_u,
            shadow.opaque_hit_v, shadow.passthrough_hits);
    }

    if (max_component(shadow.transmittance) > 0) {
        paths.radiance[path_id] += item.contribution * shadow.transmittance;
    }
}

__global__ void clear_bounce_queues_kernel(WorkQueue<RayWorkItem> ray_queue,
                                           MaterialHitQueueViews hit_queues,
                                           WorkQueue<ShadowRayWorkItem> shadow_queue,
                                           WorkQueue<TerminatedPathItem> terminate_queue) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *ray_queue.size = 0;
    hit_queues.clear();
    *shadow_queue.size = 0;
    *terminate_queue.size = 0;
}

void launch_generate_camera_rays(WavefrontRuntime& runtime,
                                 PathStateView paths,
                                 const Camera& camera,
                                 const RenderSettings& settings,
                                 int width,
                                 int height,
                                 int spp,
                                 int sample_index,
                                 int batch_pixel_offset,
                                 int batch_size) {
    generate_camera_rays_kernel<<<grid_size(batch_size), WavefrontRuntime::kBlockSize>>>(
        paths, runtime.ray_queue.view(), camera, settings, width, height, spp,
        sample_index, batch_pixel_offset, batch_size);
    check_cuda_or_throw("generate_camera_rays_kernel");
}

void launch_intersect_bvh(WavefrontRuntime& runtime,
                          PathStateView paths,
                          GpuBvhView bvh,
                          GpuScene scene,
                          WavefrontDebugViews debug,
                          int width,
                          int height,
                          int batch_size) {
    intersect_bvh_kernel<<<grid_size(batch_size), WavefrontRuntime::kBlockSize>>>(
        runtime.ray_queue.view(),
        runtime.hit_queues.views(),
        paths, bvh, scene, debug, width, height);
    check_cuda_or_throw("intersect_bvh_kernel");
}

template <bool LogPaths>
void launch_evaluate_hits_impl(WavefrontRuntime& runtime,
                               PathStateView paths,
                               GpuScene scene,
                               WavefrontDebugViews debug,
                               PathLogView path_log,
                               int width,
                               int height,
                               int wavefront_depth,
                               int max_depth,
                               int batch_size) {
#define LUX_LAUNCH_MATERIAL_QUEUE(queue_name, material_type, label) \
    evaluate_hit_kernel<material_type, LogPaths> \
        <<<grid_size(batch_size), WavefrontRuntime::kBlockSize>>>( \
            runtime.hit_queues.storage(MaterialQueueKind::queue_name).view(), \
            runtime.next_ray_queue.view(), runtime.shadow_queue.view(), \
            runtime.terminate_queue.view(), paths, scene, width, height, \
            wavefront_depth, max_depth, debug, path_log); \
    check_cuda_or_throw("evaluate_hit_kernel " label);
    LUX_WAVEFRONT_MATERIAL_QUEUE_LIST(LUX_LAUNCH_MATERIAL_QUEUE)
#undef LUX_LAUNCH_MATERIAL_QUEUE
}

void launch_evaluate_hits(WavefrontRuntime& runtime,
                          PathStateView paths,
                          GpuScene scene,
                          WavefrontDebugViews debug,
                          PathLogView path_log,
                          bool enable_path_log,
                          int width,
                          int height,
                          int wavefront_depth,
                          int max_depth,
                          int batch_size) {
    if (enable_path_log) {
        launch_evaluate_hits_impl<true>(
            runtime, paths, scene, debug, path_log, width, height,
            wavefront_depth, max_depth, batch_size);
    } else {
        launch_evaluate_hits_impl<false>(
            runtime, paths, scene, debug, path_log, width, height,
            wavefront_depth, max_depth, batch_size);
    }
}

void launch_trace_shadow_rays(WavefrontRuntime& runtime,
                              PathStateView paths,
                              GpuBvhView bvh,
                              GpuScene scene,
                              WavefrontDebugViews debug,
                              int batch_size) {
    trace_shadow_rays_kernel<<<grid_size(batch_size), WavefrontRuntime::kBlockSize>>>(
        runtime.shadow_queue.view(), paths, bvh, scene, debug);
    check_cuda_or_throw("trace_shadow_rays_kernel");
}

void launch_flush_batch_paths(WavefrontRuntime& runtime,
                              PathStateView paths,
                              int batch_size) {
    flush_batch_paths_kernel<<<grid_size(batch_size), WavefrontRuntime::kBlockSize>>>(
        paths, runtime.output_pixels_ptr(), batch_size);
    check_cuda_or_throw("flush_batch_paths_kernel");
}

void launch_clear_bounce_queues(WavefrontRuntime& runtime) {
    clear_bounce_queues_kernel<<<1, 1>>>(
        runtime.ray_queue.view(),
        runtime.hit_queues.views(),
        runtime.shadow_queue.view(),
        runtime.terminate_queue.view());
    check_cuda_or_throw("clear_bounce_queues_kernel");
}
