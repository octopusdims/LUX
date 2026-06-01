#pragma once

#ifndef LUX_BVH_CPU_TRAVERSAL_H
#define LUX_BVH_CPU_TRAVERSAL_H

#include "cpu_bvh.h"
#include "core/ray.cuh"
#include "core/types.h"

struct CpuBvhPrimitiveHit {
    Float t = INFINITY;
    Float u = 0;
    Float v = 0;
    int primitive_id = -1;
};

template <typename IntersectPrimitive>
LuxInline bool intersect_cpu_bvh_nodes(const std::vector<CpuBvhNode>& nodes,
                                       const std::vector<int>& primitive_ids,
                                       int root_node,
                                       Ray ray,
                                       IntersectPrimitive intersect_primitive,
                                       int& hit_primitive_id,
                                       Float& hit_t,
                                       Float& hit_u,
                                       Float& hit_v,
                                       bool* stack_overflow = nullptr) {
    constexpr int kStackCapacity = 64;
    int stack[kStackCapacity];
    int stack_size = 0;

    if (stack_overflow) *stack_overflow = false;
    hit_primitive_id = -1;
    hit_t = ray.t_max;
    hit_u = 0;
    hit_v = 0;

    if (root_node < 0 || nodes.empty()) return false;

    stack[stack_size++] = root_node;
    bool found = false;

    while (stack_size > 0) {
        int node_index = stack[--stack_size];
        const CpuBvhNode& node = nodes[node_index];

        Float t_near, t_far;
        if (!node.bounds.intersect(ray, t_near, t_far)) continue;

        if (node.is_leaf()) {
            int end = node.first_primitive + node.primitive_count;
            for (int i = node.first_primitive; i < end; ++i) {
                int primitive_id = primitive_ids[i];
                CpuBvhPrimitiveHit primitive_hit;
                if (!intersect_primitive(primitive_id, ray, primitive_hit)) continue;
                if (primitive_hit.t >= hit_t) continue;

                found = true;
                hit_t = primitive_hit.t;
                hit_u = primitive_hit.u;
                hit_v = primitive_hit.v;
                hit_primitive_id = primitive_hit.primitive_id >= 0
                    ? primitive_hit.primitive_id
                    : primitive_id;
                ray.t_max = primitive_hit.t;
            }
            continue;
        }

        if (stack_size + 2 > kStackCapacity) {
            if (stack_overflow) *stack_overflow = true;
            hit_primitive_id = -1;
            hit_t = ray.t_max;
            return false;
        }
        stack[stack_size++] = node.left;
        stack[stack_size++] = node.right;
    }

    return found;
}

template <typename IntersectPrimitive>
LuxInline bool intersect_cpu_bvh_tree(const CpuBvhTree& bvh, Ray ray,
                                      IntersectPrimitive intersect_primitive,
                                      int& hit_primitive_id,
                                      Float& hit_t,
                                      Float& hit_u,
                                      Float& hit_v,
                                      bool* stack_overflow = nullptr) {
    return intersect_cpu_bvh_nodes(
        bvh.nodes, bvh.primitive_ids, bvh.root_node, ray, intersect_primitive,
        hit_primitive_id, hit_t, hit_u, hit_v, stack_overflow);
}

#endif // LUX_BVH_CPU_TRAVERSAL_H
