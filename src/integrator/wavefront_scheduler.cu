#include "integrator/wavefront_runtime.cuh"

#include <algorithm>
#include <utility>

#include "core/cuda_check.cuh"
#include "integrator/wavefront_kernels.cuh"
#include "scene/prepared_scene.h"

namespace {

struct WavefrontFrameContext {
    PathStateView paths;
    GpuBvhView bvh;
    GpuScene scene;
    WavefrontDebugViews debug;
    PathLogView path_log;
    int width = 0;
    int height = 0;
    int max_depth = 0;
    int current_batch = 0;
    bool enable_path_log = false;
};

void generate_camera_stage(WavefrontRuntime& runtime,
                           const WavefrontFrameContext& context,
                           const RenderSettings& settings,
                           int spp,
                           int sample_index,
                           int batch_offset) {
    launch_generate_camera_rays(
        runtime, context.paths, context.scene.camera, settings, context.width, context.height,
        spp, sample_index, batch_offset, context.current_batch);
}

void intersect_stage(WavefrontRuntime& runtime,
                     const WavefrontFrameContext& context) {
    launch_intersect_bvh(
        runtime, context.paths, context.bvh, context.scene, context.debug,
        context.width, context.height, context.current_batch);
}

void shade_baseline_stage(WavefrontRuntime& runtime,
                          const WavefrontFrameContext& context,
                          int wavefront_depth) {
    launch_evaluate_hits(
        runtime, context.paths, context.scene, context.debug, context.path_log,
        context.enable_path_log, context.width, context.height,
        wavefront_depth, context.max_depth, context.current_batch);
}

void trace_shadow_stage(WavefrontRuntime& runtime,
                        const WavefrontFrameContext& context) {
    launch_trace_shadow_rays(
        runtime, context.paths, context.bvh, context.scene, context.debug,
        context.current_batch);
}

void flush_batch_stage(WavefrontRuntime& runtime,
                       const WavefrontFrameContext& context) {
    launch_flush_batch_paths(runtime, context.paths, context.current_batch);
}

void clear_and_swap_stage(WavefrontRuntime& runtime) {
    launch_clear_bounce_queues(runtime);
    std::swap(runtime.ray_queue, runtime.next_ray_queue);
}

} // namespace

void WavefrontRuntime::render_baseline_path_tracing(
        const PreparedScene& scene,
        Film& film,
        const RenderSettings& settings,
        RenderOutputs* outputs,
        const WavefrontRuntimeOptions& options) {
    ensure_allocated(options, settings.max_depth);

    PathStateView paths_view = path_state();
    GpuBvhView bvh_view = scene.device_bvh();
    GpuScene scene_view = scene.device_scene();
    WavefrontDebugViews debug = debug_views();

    for (int s = 0; s < film.samples_per_pixel; ++s) {
        for (int batch_offset = 0; batch_offset < options.image_size;
             batch_offset += options.batch_size) {
            int current_batch = std::min(options.batch_size, options.image_size - batch_offset);
            path_log.clear();
            PathLogView path_log_view = path_log.view(s + settings.sample_offset);
            WavefrontFrameContext context{
                paths_view,
                bvh_view,
                scene_view,
                debug,
                path_log_view,
                film.width,
                film.height,
                settings.max_depth,
                current_batch,
                options.enable_path_log
            };

            generate_camera_stage(
                *this, context, settings, film.samples_per_pixel,
                s, batch_offset);

            for (int wavefront_depth = 0; wavefront_depth <= settings.max_depth;
                ++wavefront_depth) {
                intersect_stage(*this, context);
                shade_baseline_stage(*this, context, wavefront_depth);
                trace_shadow_stage(*this, context);
                if (wavefront_depth < settings.max_depth) {
                    clear_and_swap_stage(*this);
                }
            }
            flush_batch_stage(*this, context);

            synchronize_cuda_or_throw("render batch");
            check_queue_overflows();
            append_path_log_to(outputs);
            clear_batch_queues();
        }
    }

    synchronize_cuda_or_throw("render");
    download_outputs(film, outputs);
}
