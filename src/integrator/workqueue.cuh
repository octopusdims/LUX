#pragma once

#ifndef LUX_INTEGRATOR_WORKQUEUE_CUH
#define LUX_INTEGRATOR_WORKQUEUE_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "core/cuda_check.cuh"
#include "core/types.h"
#include "integrator/workitems.cuh"

namespace lux_workqueue_detail {

template <typename T>
void cuda_allocate_array(T*& ptr, int capacity) {
    LUX_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), capacity * sizeof(T)));
}

template <typename T>
void cuda_release_array(T*& ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

template <typename T>
struct WorkQueueFieldsView {
    T* items = nullptr;

    LuxDeviceInline T read(int index) const {
        return items[index];
    }

    LuxDeviceInline void write(int index, const T& item) const {
        items[index] = item;
    }
};

template <typename T>
struct WorkQueueFieldsStorage {
    T* items = nullptr;

    void allocate(int capacity) {
        cuda_allocate_array(items, capacity);
    }

    void free() {
        cuda_release_array(items);
    }

    WorkQueueFieldsView<T> view() const {
        return WorkQueueFieldsView<T>{items};
    }

    void move_from(WorkQueueFieldsStorage& other) noexcept {
        items = other.items;
        other.items = nullptr;
    }
};

template <>
struct WorkQueueFieldsView<RayWorkItem> {
    int* path_id = nullptr;
    Ray* ray = nullptr;

    LuxDeviceInline RayWorkItem read(int index) const {
        return RayWorkItem{path_id[index], ray[index]};
    }

    LuxDeviceInline void write(int index, const RayWorkItem& item) const {
        path_id[index] = item.path_id;
        ray[index] = item.ray;
    }
};

template <>
struct WorkQueueFieldsStorage<RayWorkItem> {
    int* path_id = nullptr;
    Ray* ray = nullptr;

    void allocate(int capacity) {
        cuda_allocate_array(path_id, capacity);
        cuda_allocate_array(ray, capacity);
    }

    void free() {
        cuda_release_array(path_id);
        cuda_release_array(ray);
    }

    WorkQueueFieldsView<RayWorkItem> view() const {
        return WorkQueueFieldsView<RayWorkItem>{path_id, ray};
    }

    void move_from(WorkQueueFieldsStorage& other) noexcept {
        path_id = other.path_id;
        ray = other.ray;
        other.path_id = nullptr;
        other.ray = nullptr;
    }
};

template <>
struct WorkQueueFieldsView<HitWorkItem> {
    int* path_id = nullptr;
    int* triangle_id = nullptr;
    PrimitiveRef* primitive_ref = nullptr;
    vec3* ray_origin = nullptr;
    vec3* position = nullptr;
    vec3* ng = nullptr;
    vec3* ns = nullptr;
    vec3* wo = nullptr;
    Float* u = nullptr;
    Float* v = nullptr;

    LuxDeviceInline HitWorkItem read(int index) const {
        return HitWorkItem{
            path_id[index], triangle_id[index], primitive_ref[index],
            ray_origin[index], position[index], ng[index], ns[index], wo[index],
            u[index], v[index]};
    }

    LuxDeviceInline void write(int index, const HitWorkItem& item) const {
        path_id[index] = item.path_id;
        triangle_id[index] = item.triangle_id;
        primitive_ref[index] = item.primitive_ref;
        ray_origin[index] = item.ray_origin;
        position[index] = item.position;
        ng[index] = item.ng;
        ns[index] = item.ns;
        wo[index] = item.wo;
        u[index] = item.u;
        v[index] = item.v;
    }
};

template <>
struct WorkQueueFieldsStorage<HitWorkItem> {
    int* path_id = nullptr;
    int* triangle_id = nullptr;
    PrimitiveRef* primitive_ref = nullptr;
    vec3* ray_origin = nullptr;
    vec3* position = nullptr;
    vec3* ng = nullptr;
    vec3* ns = nullptr;
    vec3* wo = nullptr;
    Float* u = nullptr;
    Float* v = nullptr;

    void allocate(int capacity) {
        cuda_allocate_array(path_id, capacity);
        cuda_allocate_array(triangle_id, capacity);
        cuda_allocate_array(primitive_ref, capacity);
        cuda_allocate_array(ray_origin, capacity);
        cuda_allocate_array(position, capacity);
        cuda_allocate_array(ng, capacity);
        cuda_allocate_array(ns, capacity);
        cuda_allocate_array(wo, capacity);
        cuda_allocate_array(u, capacity);
        cuda_allocate_array(v, capacity);
    }

    void free() {
        cuda_release_array(path_id);
        cuda_release_array(triangle_id);
        cuda_release_array(primitive_ref);
        cuda_release_array(ray_origin);
        cuda_release_array(position);
        cuda_release_array(ng);
        cuda_release_array(ns);
        cuda_release_array(wo);
        cuda_release_array(u);
        cuda_release_array(v);
    }

    WorkQueueFieldsView<HitWorkItem> view() const {
        return WorkQueueFieldsView<HitWorkItem>{
            path_id, triangle_id, primitive_ref, ray_origin, position,
            ng, ns, wo, u, v};
    }

    void move_from(WorkQueueFieldsStorage& other) noexcept {
        path_id = other.path_id;
        triangle_id = other.triangle_id;
        primitive_ref = other.primitive_ref;
        ray_origin = other.ray_origin;
        position = other.position;
        ng = other.ng;
        ns = other.ns;
        wo = other.wo;
        u = other.u;
        v = other.v;
        other.path_id = nullptr;
        other.triangle_id = nullptr;
        other.primitive_ref = nullptr;
        other.ray_origin = nullptr;
        other.position = nullptr;
        other.ng = nullptr;
        other.ns = nullptr;
        other.wo = nullptr;
        other.u = nullptr;
        other.v = nullptr;
    }
};

template <>
struct WorkQueueFieldsView<ShadowRayWorkItem> {
    int* path_id = nullptr;
    int* source_triangle_id = nullptr;
    PrimitiveRef* source_primitive_ref = nullptr;
    int* source_material_id = nullptr;
    int* source_depth = nullptr;
    Ray* shadow_ray = nullptr;
    vec3* contribution = nullptr;
    vec3* source_geometric_normal = nullptr;
    vec3* source_shading_normal = nullptr;
    Float* pdf_light_solid_angle = nullptr;
    Float* pdf_bsdf_solid_angle = nullptr;

    LuxDeviceInline ShadowRayWorkItem read(int index) const {
        return ShadowRayWorkItem{
            path_id[index], source_triangle_id[index], source_primitive_ref[index],
            source_material_id[index], source_depth[index], shadow_ray[index],
            contribution[index], source_geometric_normal[index],
            source_shading_normal[index], pdf_light_solid_angle[index],
            pdf_bsdf_solid_angle[index]};
    }

    LuxDeviceInline void write(int index, const ShadowRayWorkItem& item) const {
        path_id[index] = item.path_id;
        source_triangle_id[index] = item.source_triangle_id;
        source_primitive_ref[index] = item.source_primitive_ref;
        source_material_id[index] = item.source_material_id;
        source_depth[index] = item.source_depth;
        shadow_ray[index] = item.shadow_ray;
        contribution[index] = item.contribution;
        source_geometric_normal[index] = item.source_geometric_normal;
        source_shading_normal[index] = item.source_shading_normal;
        pdf_light_solid_angle[index] = item.pdf_light_solid_angle;
        pdf_bsdf_solid_angle[index] = item.pdf_bsdf_solid_angle;
    }
};

template <>
struct WorkQueueFieldsStorage<ShadowRayWorkItem> {
    int* path_id = nullptr;
    int* source_triangle_id = nullptr;
    PrimitiveRef* source_primitive_ref = nullptr;
    int* source_material_id = nullptr;
    int* source_depth = nullptr;
    Ray* shadow_ray = nullptr;
    vec3* contribution = nullptr;
    vec3* source_geometric_normal = nullptr;
    vec3* source_shading_normal = nullptr;
    Float* pdf_light_solid_angle = nullptr;
    Float* pdf_bsdf_solid_angle = nullptr;

    void allocate(int capacity) {
        cuda_allocate_array(path_id, capacity);
        cuda_allocate_array(source_triangle_id, capacity);
        cuda_allocate_array(source_primitive_ref, capacity);
        cuda_allocate_array(source_material_id, capacity);
        cuda_allocate_array(source_depth, capacity);
        cuda_allocate_array(shadow_ray, capacity);
        cuda_allocate_array(contribution, capacity);
        cuda_allocate_array(source_geometric_normal, capacity);
        cuda_allocate_array(source_shading_normal, capacity);
        cuda_allocate_array(pdf_light_solid_angle, capacity);
        cuda_allocate_array(pdf_bsdf_solid_angle, capacity);
    }

    void free() {
        cuda_release_array(path_id);
        cuda_release_array(source_triangle_id);
        cuda_release_array(source_primitive_ref);
        cuda_release_array(source_material_id);
        cuda_release_array(source_depth);
        cuda_release_array(shadow_ray);
        cuda_release_array(contribution);
        cuda_release_array(source_geometric_normal);
        cuda_release_array(source_shading_normal);
        cuda_release_array(pdf_light_solid_angle);
        cuda_release_array(pdf_bsdf_solid_angle);
    }

    WorkQueueFieldsView<ShadowRayWorkItem> view() const {
        return WorkQueueFieldsView<ShadowRayWorkItem>{
            path_id, source_triangle_id, source_primitive_ref, source_material_id,
            source_depth, shadow_ray, contribution, source_geometric_normal,
            source_shading_normal, pdf_light_solid_angle, pdf_bsdf_solid_angle};
    }

    void move_from(WorkQueueFieldsStorage& other) noexcept {
        path_id = other.path_id;
        source_triangle_id = other.source_triangle_id;
        source_primitive_ref = other.source_primitive_ref;
        source_material_id = other.source_material_id;
        source_depth = other.source_depth;
        shadow_ray = other.shadow_ray;
        contribution = other.contribution;
        source_geometric_normal = other.source_geometric_normal;
        source_shading_normal = other.source_shading_normal;
        pdf_light_solid_angle = other.pdf_light_solid_angle;
        pdf_bsdf_solid_angle = other.pdf_bsdf_solid_angle;
        other.path_id = nullptr;
        other.source_triangle_id = nullptr;
        other.source_primitive_ref = nullptr;
        other.source_material_id = nullptr;
        other.source_depth = nullptr;
        other.shadow_ray = nullptr;
        other.contribution = nullptr;
        other.source_geometric_normal = nullptr;
        other.source_shading_normal = nullptr;
        other.pdf_light_solid_angle = nullptr;
        other.pdf_bsdf_solid_angle = nullptr;
    }
};

template <>
struct WorkQueueFieldsView<TerminatedPathItem> {
    int* path_id = nullptr;

    LuxDeviceInline TerminatedPathItem read(int index) const {
        return TerminatedPathItem{path_id[index]};
    }

    LuxDeviceInline void write(int index, const TerminatedPathItem& item) const {
        path_id[index] = item.path_id;
    }
};

template <>
struct WorkQueueFieldsStorage<TerminatedPathItem> {
    int* path_id = nullptr;

    void allocate(int capacity) {
        cuda_allocate_array(path_id, capacity);
    }

    void free() {
        cuda_release_array(path_id);
    }

    WorkQueueFieldsView<TerminatedPathItem> view() const {
        return WorkQueueFieldsView<TerminatedPathItem>{path_id};
    }

    void move_from(WorkQueueFieldsStorage& other) noexcept {
        path_id = other.path_id;
        other.path_id = nullptr;
    }
};

} // namespace lux_workqueue_detail

// Device-side work queue view with atomic push. Kernels iterate by tid over [0, *size).
template <typename T>
struct WorkQueueView {
    lux_workqueue_detail::WorkQueueFieldsView<T> fields;
    int* size = nullptr;
    int* overflow = nullptr;
    int capacity = 0;

    __device__ bool push(const T& item) {
        int index = atomicAdd(size, 1);
        if (index >= capacity) {
            atomicExch(overflow, 1);
            atomicMin(size, capacity);
            return false;
        }
        fields.write(index, item);
        return true;
    }

    LuxDeviceInline T read(int index) const {
        return fields.read(index);
    }

    LuxDeviceInline void write(int index, const T& item) const {
        fields.write(index, item);
    }
};

template <typename T>
using WorkQueue = WorkQueueView<T>;

template <typename T>
struct WorkQueueStorage {
    lux_workqueue_detail::WorkQueueFieldsStorage<T> fields;
    int* size = nullptr;
    int* overflow = nullptr;
    int capacity = 0;

    WorkQueueStorage() = default;

    ~WorkQueueStorage() {
        free();
    }

    WorkQueueStorage(const WorkQueueStorage&) = delete;
    WorkQueueStorage& operator=(const WorkQueueStorage&) = delete;

    WorkQueueStorage(WorkQueueStorage&& other) noexcept {
        move_from(other);
    }

    WorkQueueStorage& operator=(WorkQueueStorage&& other) noexcept {
        if (this != &other) {
            free();
            move_from(other);
        }
        return *this;
    }

    void allocate(int cap) {
        free();
        capacity = cap;
        fields.allocate(capacity);
        LUX_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&size), sizeof(int)));
        LUX_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&overflow), sizeof(int)));
        clear();
    }

    void free() {
        fields.free();
        if (size) { cudaFree(size); size = nullptr; }
        if (overflow) { cudaFree(overflow); overflow = nullptr; }
        capacity = 0;
    }

    void clear() {
        if (!size || !overflow) return;
        LUX_CUDA_CHECK(cudaMemsetAsync(size, 0, sizeof(int), 0));
        LUX_CUDA_CHECK(cudaMemsetAsync(overflow, 0, sizeof(int), 0));
    }

    int host_size() const {
        int s = 0;
        if (!size) return s;
        LUX_CUDA_CHECK(cudaMemcpy(&s, size, sizeof(int), cudaMemcpyDeviceToHost));
        return s;
    }

    bool host_overflowed() const {
        int flag = 0;
        if (!overflow) return false;
        LUX_CUDA_CHECK(cudaMemcpy(&flag, overflow, sizeof(int), cudaMemcpyDeviceToHost));
        return flag != 0;
    }

    WorkQueueView<T> view() const {
        return WorkQueueView<T>{fields.view(), size, overflow, capacity};
    }

private:
    void move_from(WorkQueueStorage& other) noexcept {
        fields.move_from(other.fields);
        size = other.size;
        overflow = other.overflow;
        capacity = other.capacity;
        other.size = nullptr;
        other.overflow = nullptr;
        other.capacity = 0;
    }
};

#endif // LUX_INTEGRATOR_WORKQUEUE_CUH
