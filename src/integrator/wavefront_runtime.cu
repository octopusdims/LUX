#include "integrator/wavefront_runtime.cuh"

#include <cstdio>
#include <stdexcept>

#include <thrust/copy.h>
#include <thrust/fill.h>

WavefrontRuntime::~WavefrontRuntime() {
    free_queues();
}

bool WavefrontRuntime::has_compatible_allocation(
        const WavefrontRuntimeOptions& options,
        int max_depth) const {
    return allocated
        && allocated_max_depth >= max_depth
        && allocated_options.batch_size >= options.batch_size
        && allocated_options.enable_path_log == options.enable_path_log;
}

void WavefrontRuntime::ensure_allocated(const WavefrontRuntimeOptions& options,
                                        int max_depth) {
    bool reuse_allocation = has_compatible_allocation(options, max_depth);
    if (!reuse_allocation) {
        free_queues();

        paths.allocate(options.batch_size);
        path_log.allocate(options.batch_size, max_depth, options.enable_path_log);

        ray_queue.allocate(options.batch_size);
        next_ray_queue.allocate(options.batch_size);
        hit_queues.allocate(options.batch_size);
        shadow_queue.allocate(options.batch_size);
        terminate_queue.allocate(options.batch_size);

        allocated_options = options;
        allocated_max_depth = max_depth;
        allocated = true;
    }

    batch_size = options.batch_size;
    path_log.max_depth = options.enable_path_log ? max_depth : 0;
    clear_batch_queues();
    clear_frame_outputs(options);
}

void WavefrontRuntime::clear_frame_outputs(const WavefrontRuntimeOptions& options) {
    output_pixels.resize(options.image_size);
    thrust::fill(output_pixels.begin(), output_pixels.end(), vec3(0));

    if (options.enable_shadow_debug) {
        shadow_debug_pixels.resize(options.image_size);
        thrust::fill(shadow_debug_pixels.begin(), shadow_debug_pixels.end(), vec3(0));
    } else {
        shadow_debug_pixels.clear();
    }

    if (options.enable_normal_debug) {
        normal_debug_pixels.resize(options.image_size);
        thrust::fill(normal_debug_pixels.begin(), normal_debug_pixels.end(), vec3(0));
    } else {
        normal_debug_pixels.clear();
    }

    if (options.enable_shading_normal_debug) {
        shading_normal_debug_pixels.resize(options.image_size);
        thrust::fill(shading_normal_debug_pixels.begin(), shading_normal_debug_pixels.end(),
                     vec3(0));
    } else {
        shading_normal_debug_pixels.clear();
    }
}

void WavefrontRuntime::free_queues() {
    ray_queue.free();
    next_ray_queue.free();
    hit_queues.free();
    shadow_queue.free();
    terminate_queue.free();
    batch_size = 0;
    allocated_options = WavefrontRuntimeOptions{};
    allocated_max_depth = -1;
    allocated = false;
}

namespace {

template <typename T>
void check_queue_overflow(const WorkQueueStorage<T>& queue, const char* name) {
    if (!queue.host_overflowed()) return;

    char message[160];
    std::snprintf(message, sizeof(message),
                  "Work queue overflow: %s (capacity=%d, size=%d)",
                  name, queue.capacity, queue.host_size());
    throw std::runtime_error(message);
}

} // namespace

void WavefrontRuntime::check_queue_overflows() const {
    check_queue_overflow(ray_queue, "camera/current ray queue");
    check_queue_overflow(next_ray_queue, "next ray queue");
    for (int i = 0; i < kMaterialQueueCount; ++i) {
        MaterialQueueKind kind = static_cast<MaterialQueueKind>(i);
        check_queue_overflow(hit_queues.storage(kind), material_queue_label(kind));
    }
    check_queue_overflow(shadow_queue, "shadow ray queue");
    check_queue_overflow(terminate_queue, "terminate queue");
}

void WavefrontRuntime::clear_batch_queues() {
    ray_queue.clear();
    next_ray_queue.clear();
    hit_queues.clear();
    shadow_queue.clear();
    terminate_queue.clear();
}

PathStateView WavefrontRuntime::path_state() {
    return paths.view();
}

vec3* WavefrontRuntime::output_pixels_ptr() {
    return thrust::raw_pointer_cast(output_pixels.data());
}

vec3* WavefrontRuntime::shadow_debug_pixels_ptr() {
    if (shadow_debug_pixels.empty()) return nullptr;
    return thrust::raw_pointer_cast(shadow_debug_pixels.data());
}

vec3* WavefrontRuntime::normal_debug_pixels_ptr() {
    if (normal_debug_pixels.empty()) return nullptr;
    return thrust::raw_pointer_cast(normal_debug_pixels.data());
}

vec3* WavefrontRuntime::shading_normal_debug_pixels_ptr() {
    if (shading_normal_debug_pixels.empty()) return nullptr;
    return thrust::raw_pointer_cast(shading_normal_debug_pixels.data());
}

WavefrontDebugViews WavefrontRuntime::debug_views() {
    return WavefrontDebugViews{
        shadow_debug_pixels_ptr(),
        normal_debug_pixels_ptr(),
        shading_normal_debug_pixels_ptr()
    };
}

void WavefrontRuntime::append_path_log_to(RenderOutputs* outputs) const {
    if (!outputs || !outputs->path_log_records) return;
    path_log.append_to(*outputs->path_log_records);
}

void WavefrontRuntime::download_outputs(Film& film, RenderOutputs* outputs) const {
    thrust::copy(output_pixels.begin(), output_pixels.end(), film.pixels.begin());

    Film* shadow_debug = outputs ? outputs->get(RenderAov::ShadowDebug) : nullptr;
    if (shadow_debug) {
        thrust::copy(shadow_debug_pixels.begin(), shadow_debug_pixels.end(),
                     shadow_debug->pixels.begin());
    }

    Film* normal_debug = outputs ? outputs->get(RenderAov::GeometricNormal) : nullptr;
    if (normal_debug) {
        thrust::copy(normal_debug_pixels.begin(), normal_debug_pixels.end(),
                     normal_debug->pixels.begin());
    }

    Film* shading_normal_debug = outputs ? outputs->get(RenderAov::ShadingNormal) : nullptr;
    if (shading_normal_debug) {
        thrust::copy(shading_normal_debug_pixels.begin(), shading_normal_debug_pixels.end(),
                     shading_normal_debug->pixels.begin());
    }
}
