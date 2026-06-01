#pragma once

#ifndef LUX_INTEGRATOR_WAVEFRONT_MATERIAL_QUEUES_CUH
#define LUX_INTEGRATOR_WAVEFRONT_MATERIAL_QUEUES_CUH

#include "integrator/pathstate.cuh"
#include "integrator/wavefront_material_registry.cuh"
#include "integrator/workqueue.cuh"
#include "material/material.h"

enum class MaterialQueueKind : int {
#define LUX_DECLARE_MATERIAL_QUEUE(name, material_type, label) name,
    LUX_WAVEFRONT_MATERIAL_QUEUE_LIST(LUX_DECLARE_MATERIAL_QUEUE)
#undef LUX_DECLARE_MATERIAL_QUEUE
    Count
};

static constexpr int kMaterialQueueCount = static_cast<int>(MaterialQueueKind::Count);

LuxHDInline int material_queue_index(MaterialQueueKind kind) {
    return static_cast<int>(kind);
}

LuxHDInline MaterialQueueKind material_queue_kind(MaterialType type) {
    switch (type) {
#define LUX_MATERIAL_QUEUE_CASE(material_type, queue_kind) \
    case material_type: return queue_kind;
        LUX_WAVEFRONT_MATERIAL_QUEUE_MAP(LUX_MATERIAL_QUEUE_CASE)
#undef LUX_MATERIAL_QUEUE_CASE
    }
    return MaterialQueueKind::Passthrough;
}

LuxInline const char* material_queue_label(MaterialQueueKind kind) {
    switch (kind) {
#define LUX_MATERIAL_QUEUE_LABEL(name, material_type, label) \
    case MaterialQueueKind::name: return label;
        LUX_WAVEFRONT_MATERIAL_QUEUE_LIST(LUX_MATERIAL_QUEUE_LABEL)
#undef LUX_MATERIAL_QUEUE_LABEL
    case MaterialQueueKind::Count:
        break;
    }
    return "unknown material hit queue";
}

struct MaterialHitQueueViews {
    WorkQueue<HitWorkItem> queues[kMaterialQueueCount];

    LuxDeviceInline bool push(MaterialType type, const HitWorkItem& item) {
        return queues[material_queue_index(material_queue_kind(type))].push(item);
    }

    LuxDeviceInline void clear() {
        for (int i = 0; i < kMaterialQueueCount; ++i) {
            *queues[i].size = 0;
        }
    }
};

struct MaterialHitQueues {
    WorkQueueStorage<HitWorkItem> queues[kMaterialQueueCount];

    void allocate(int capacity) {
        for (int i = 0; i < kMaterialQueueCount; ++i) {
            queues[i].allocate(capacity);
        }
    }

    void free() {
        for (int i = 0; i < kMaterialQueueCount; ++i) {
            queues[i].free();
        }
    }

    void clear() {
        for (int i = 0; i < kMaterialQueueCount; ++i) {
            queues[i].clear();
        }
    }

    WorkQueueStorage<HitWorkItem>& storage(MaterialQueueKind kind) {
        return queues[material_queue_index(kind)];
    }

    const WorkQueueStorage<HitWorkItem>& storage(MaterialQueueKind kind) const {
        return queues[material_queue_index(kind)];
    }

    MaterialHitQueueViews views() const {
        MaterialHitQueueViews result;
        for (int i = 0; i < kMaterialQueueCount; ++i) {
            result.queues[i] = queues[i].view();
        }
        return result;
    }
};

#endif // LUX_INTEGRATOR_WAVEFRONT_MATERIAL_QUEUES_CUH
