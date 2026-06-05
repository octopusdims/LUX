#pragma once

#ifndef LUX_INTEGRATOR_PATHSTATE_CUH
#define LUX_INTEGRATOR_PATHSTATE_CUH

#include "core/types.h"
#include "core/vec3.cuh"
#include "integrator/workitems.cuh"
#include "sampler/sample_record.h"
#include "sampler/sampler.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include <stdexcept>
#include <vector>

enum PathLogEventFlags : unsigned {
    PathLogEventNone = 0,
    PathLogEventSurfaceHit = 1u << 0,
    PathLogEventEmission = 1u << 1,
    PathLogEventDirectLightSampled = 1u << 2,
    PathLogEventDirectLightValid = 1u << 3,
    PathLogEventBsdfSampled = 1u << 4,
    PathLogEventTerminated = 1u << 5,
    PathLogEventSpawnedRay = 1u << 6,
    PathLogEventDelta = 1u << 7,
    PathLogEventPassthrough = 1u << 8,
};

enum PathLogTerminationFlags : unsigned {
    PathLogTerminationNone = 0,
    PathLogTerminationNonScattering = 1u << 0,
    PathLogTerminationMaxDepth = 1u << 1,
    PathLogTerminationInvalidBsdf = 1u << 2,
    PathLogTerminationRussianRoulette = 1u << 3,
    PathLogTerminationZeroThroughput = 1u << 4,
};

struct PathLogRecord {
    int pixel_index = -1;
    int sample_index = -1;
    int depth = -1;
    int triangle_id = -1;
    PrimitiveRef primitive_ref;
    int material_id = -1;
    vec3 position = vec3(0);
    vec3 ng = vec3(0);
    vec3 ns = vec3(0);
    vec3 wo = vec3(0);
    vec3 wi = vec3(0);
    vec3 throughput_before = vec3(0);
    vec3 throughput_after = vec3(0);
    Float bsdf_pdf = 0;
    Float guide_pdf = 0;
    Float light_pdf = 0;
    PdfMeasure bsdf_pdf_measure = PdfMeasure::None;
    PdfMeasure guide_pdf_measure = PdfMeasure::None;
    PdfMeasure light_pdf_measure = PdfMeasure::None;
    SampleSource sample_source = SampleSource::None;
    unsigned bsdf_flags = 0;
    unsigned termination_flags = PathLogTerminationNone;
    unsigned event_flags = PathLogEventNone;
};

struct PathLogView {
    PathLogRecord* records = nullptr;
    int* count = nullptr;
    int* overflow = nullptr;
    int capacity = 0;
    int max_depth = 0;
    int sample_index = 0;
};

LuxDeviceInline bool path_log_enabled(PathLogView log) {
    return log.records != nullptr && log.count != nullptr && log.overflow != nullptr
        && log.capacity > 0 && log.max_depth >= 0;
}

LuxDeviceInline void record_path_vertex(PathLogView log, int path_id,
                                        const PathLogRecord& record) {
    if (!path_log_enabled(log)) return;
    if (path_id < 0) return;
    if (record.depth < 0 || record.depth > log.max_depth) return;

    int index = atomicAdd(log.count, 1);
    if (index >= log.capacity) {
        atomicExch(log.overflow, 1);
        return;
    }
    log.records[index] = record;
}

struct PathLogStorage {
    int capacity = 0;
    int max_depth = 0;
    thrust::device_vector<PathLogRecord> records;
    thrust::device_vector<int> count;
    thrust::device_vector<int> overflow;

    void allocate(int paths, int depth, bool enabled) {
        capacity = enabled ? paths * (depth + 1) : 0;
        max_depth = enabled ? depth : 0;
        if (enabled) {
            records.resize(static_cast<size_t>(capacity));
            count.resize(1);
            overflow.resize(1);
        } else {
            records.clear();
            count.clear();
            overflow.clear();
        }
    }

    void clear() {
        if (!count.empty()) {
            thrust::fill(count.begin(), count.end(), 0);
        }
        if (!overflow.empty()) {
            thrust::fill(overflow.begin(), overflow.end(), 0);
        }
    }

    PathLogView view(int sample_index) {
        if (records.empty()) return PathLogView{};
        return PathLogView{
            thrust::raw_pointer_cast(records.data()),
            thrust::raw_pointer_cast(count.data()),
            thrust::raw_pointer_cast(overflow.data()),
            capacity,
            max_depth,
            sample_index
        };
    }

    void append_to(std::vector<PathLogRecord>& out) const {
        if (records.empty()) return;

        int host_count = 0;
        int host_overflow = 0;
        thrust::copy(count.begin(), count.end(), &host_count);
        thrust::copy(overflow.begin(), overflow.end(), &host_overflow);
        if (host_overflow != 0 || host_count > capacity) {
            throw std::runtime_error("PathLogStorage overflow: compact path log capacity exhausted");
        }
        if (host_count <= 0) return;

        size_t record_count = static_cast<size_t>(host_count);
        size_t offset = out.size();
        out.resize(offset + record_count);
        thrust::copy(records.begin(), records.begin() + record_count, out.begin() + offset);
    }
};

// SoA path state — one entry per active path in the current batch.
// Allocated for the full batch size; individual paths are indexed by path_id.
struct PathStateView {
    int max_paths = 0;
    vec3* throughput = nullptr;
    Float* eta_scale = nullptr;
    vec3* radiance = nullptr;
    int* pixel_index = nullptr;
    int* depth = nullptr;
    int* passthrough_depth = nullptr;
    Float* last_sample_pdf = nullptr;
    PdfMeasure* last_sample_measure = nullptr;
    SampleSource* last_sample_source = nullptr;
    unsigned* last_sample_flags = nullptr;
    vec3* last_sample_reference_position = nullptr;
    SamplerState* samplers = nullptr;
};

LuxHDInline void clear_last_sample(PathStateView paths, int path_id,
                                   SampleSource source = SampleSource::Camera) {
    paths.last_sample_pdf[path_id] = 0;
    paths.last_sample_measure[path_id] = PdfMeasure::None;
    paths.last_sample_source[path_id] = source;
    paths.last_sample_flags[path_id] = 0;
    paths.last_sample_reference_position[path_id] = vec3(0);
}

LuxHDInline void store_last_sample(PathStateView paths, int path_id,
                                   Float pdf,
                                   PdfMeasure measure,
                                   SampleSource source,
                                   unsigned flags,
                                   const vec3& reference_position) {
    paths.last_sample_pdf[path_id] = pdf;
    paths.last_sample_measure[path_id] = measure;
    paths.last_sample_source[path_id] = source;
    paths.last_sample_flags[path_id] = flags;
    paths.last_sample_reference_position[path_id] = reference_position;
}

LuxHDInline bool last_sample_is_solid_angle(PathStateView paths, int path_id) {
    return paths.last_sample_measure[path_id] == PdfMeasure::SolidAngle
        && paths.last_sample_pdf[path_id] > 0;
}

struct PathStateStorage {
    thrust::device_vector<vec3> throughput;
    thrust::device_vector<Float> eta_scale;
    thrust::device_vector<vec3> radiance;
    thrust::device_vector<int> pixel_index;
    thrust::device_vector<int> depth;
    thrust::device_vector<int> passthrough_depth;
    thrust::device_vector<Float> last_sample_pdf;
    thrust::device_vector<PdfMeasure> last_sample_measure;
    thrust::device_vector<SampleSource> last_sample_source;
    thrust::device_vector<unsigned> last_sample_flags;
    thrust::device_vector<vec3> last_sample_reference_position;
    thrust::device_vector<SamplerState> samplers;

    void allocate(int max_paths) {
        throughput.resize(max_paths);
        eta_scale.resize(max_paths);
        radiance.resize(max_paths);
        pixel_index.resize(max_paths);
        depth.resize(max_paths);
        passthrough_depth.resize(max_paths);
        last_sample_pdf.resize(max_paths);
        last_sample_measure.resize(max_paths);
        last_sample_source.resize(max_paths);
        last_sample_flags.resize(max_paths);
        last_sample_reference_position.resize(max_paths);
        samplers.resize(max_paths);
    }

    PathStateView view() {
        PathStateView ps;
        ps.max_paths = static_cast<int>(throughput.size());
        ps.throughput = thrust::raw_pointer_cast(throughput.data());
        ps.eta_scale = thrust::raw_pointer_cast(eta_scale.data());
        ps.radiance = thrust::raw_pointer_cast(radiance.data());
        ps.pixel_index = thrust::raw_pointer_cast(pixel_index.data());
        ps.depth = thrust::raw_pointer_cast(depth.data());
        ps.passthrough_depth = thrust::raw_pointer_cast(passthrough_depth.data());
        ps.last_sample_pdf = thrust::raw_pointer_cast(last_sample_pdf.data());
        ps.last_sample_measure = thrust::raw_pointer_cast(last_sample_measure.data());
        ps.last_sample_source = thrust::raw_pointer_cast(last_sample_source.data());
        ps.last_sample_flags = thrust::raw_pointer_cast(last_sample_flags.data());
        ps.last_sample_reference_position =
            thrust::raw_pointer_cast(last_sample_reference_position.data());
        ps.samplers = thrust::raw_pointer_cast(samplers.data());
        return ps;
    }
};

#endif // LUX_INTEGRATOR_PATHSTATE_CUH
