#include "bvh/bvh_overlap.cuh"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "core/cuda_check.cuh"

namespace {

constexpr int kQueryBlockSize = 128;
constexpr int kSharedPairCapacity = 4 * kQueryBlockSize;
}

int grid_size(int count, int block_size) {
    return (count + block_size - 1) / block_size;
}

__global__ void overlap_self_kernel(const GpuBvhNode* nodes,
                                    int node_count,
                                    const AABB* query_bounds,
                                    const int* query_primitive_ids,
                                    int query_count,
                                    PrimitivePair* pairs,
                                    int max_pairs,
                                    int* counter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    bool active = tid < query_count;
    AABB query_bound;
    int query_id = -1;
    int node_index = active ? 0 : -1;
    if (active) {
        query_bound = query_bounds[tid];
        query_id = query_primitive_ids[tid];
    }

    __shared__ PrimitivePair shared_pairs[kSharedPairCapacity];
    __shared__ int shared_count;
    __shared__ int shared_active_count;
    __shared__ int shared_global_index;

    while (true) {
        if (threadIdx.x == 0) {
            shared_count = 0;
            shared_active_count = 0;
        }
        __syncthreads();

        if (active && node_index >= 0 && node_index < node_count) {
            atomicAdd(&shared_active_count, 1);
        }
        __syncthreads();
        int active_count = shared_active_count;

        if (active) {
            while (node_index >= 0 && node_index < node_count) {
                const GpuBvhNode& node = nodes[node_index];
                if (!node.bounds.overlaps(query_bound)) {
                    node_index = node.escape;
                    continue;
                }

                if (node.primitive_id >= 0) {
                    if (node.primitive_id > query_id) {
                        int shared_index = atomicAdd(&shared_count, 1);
                        if (shared_index < kSharedPairCapacity) {
                            shared_pairs[shared_index] =
                                PrimitivePair{query_id, node.primitive_id};
                            node_index = node.escape;
                        }
                        break;
                    }
                    node_index = node.escape;
                } else {
                    node_index += 1;
                }

                if (shared_count >= kSharedPairCapacity) break;
            }
        }

        __syncthreads();
        int write_count = min(shared_count, kSharedPairCapacity);
        if (threadIdx.x == 0 && write_count > 0) {
            shared_global_index = atomicAdd(counter, write_count);
        }
        __syncthreads();

        if (write_count > 0) {
            int global_index = shared_global_index;
            for (int i = threadIdx.x; i < write_count; i += blockDim.x) {
                int output_index = global_index + i;
                if (output_index < max_pairs) pairs[output_index] = shared_pairs[i];
            }
        }

        if (write_count == 0 && active_count == 0) break;
    }
}

__global__ void overlap_cross_kernel(const GpuBvhNode* left_nodes,
                                     int left_node_count,
                                     const AABB* right_bounds,
                                     const int* right_primitive_ids,
                                     const uint32_t* right_query_indices,
                                     int right_count,
                                     PrimitivePair* pairs,
                                     int max_pairs,
                                     int* counter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    bool active = tid < right_count;
    AABB query_bound;
    int right_id = -1;
    int node_index = active ? 0 : -1;
    if (active) {
        int right_input_index = static_cast<int>(right_query_indices[tid]);
        query_bound = right_bounds[right_input_index];
        right_id = right_primitive_ids[right_input_index];
    }

    __shared__ PrimitivePair shared_pairs[kSharedPairCapacity];
    __shared__ int shared_count;
    __shared__ int shared_active_count;
    __shared__ int shared_global_index;

    while (true) {
        if (threadIdx.x == 0) {
            shared_count = 0;
            shared_active_count = 0;
        }
        __syncthreads();

        if (active && node_index >= 0 && node_index < left_node_count) {
            atomicAdd(&shared_active_count, 1);
        }
        __syncthreads();
        int active_count = shared_active_count;

        if (active) {
            while (node_index >= 0 && node_index < left_node_count) {
                const GpuBvhNode& node = left_nodes[node_index];
                if (!node.bounds.overlaps(query_bound)) {
                    node_index = node.escape;
                    continue;
                }

                if (node.primitive_id >= 0) {
                    int shared_index = atomicAdd(&shared_count, 1);
                    if (shared_index < kSharedPairCapacity) {
                        shared_pairs[shared_index] = PrimitivePair{node.primitive_id, right_id};
                        node_index = node.escape;
                    }
                    break;
                }

                node_index += 1;
                if (shared_count >= kSharedPairCapacity) break;
            }
        }

        __syncthreads();
        int write_count = min(shared_count, kSharedPairCapacity);
        if (threadIdx.x == 0 && write_count > 0) {
            shared_global_index = atomicAdd(counter, write_count);
        }
        __syncthreads();

        if (write_count > 0) {
            int global_index = shared_global_index;
            for (int i = threadIdx.x; i < write_count; i += blockDim.x) {
                int output_index = global_index + i;
                if (output_index < max_pairs) pairs[output_index] = shared_pairs[i];
            }
        }

        if (write_count == 0 && active_count == 0) break;
    }
}

OverlapQueryResult query_aabb_overlaps_self(const GpuBvh& bvh, int max_pairs) {
    OverlapQueryResult result;
    if (!bvh.is_valid() || max_pairs <= 0) return result;

    GpuBvhView bvh_view = bvh.device_view();
    const AABB* bounds = bvh.overlap_bounds_device_ptr();
    const int* primitive_ids = bvh.overlap_primitive_ids_device_ptr();

    thrust::device_vector<PrimitivePair> d_pairs(max_pairs);
    thrust::device_vector<int> d_counter(1, 0);

    overlap_self_kernel<<<grid_size(bvh.primitive_count, kQueryBlockSize), kQueryBlockSize>>>(
        bvh_view.nodes,
        bvh_view.node_count,
        bounds,
        primitive_ids,
        bvh.primitive_count,
        thrust::raw_pointer_cast(d_pairs.data()),
        max_pairs,
        thrust::raw_pointer_cast(d_counter.data()));
    check_cuda_or_throw("overlap_self_kernel");
    synchronize_cuda_or_throw("overlap_self_kernel synchronize");

    thrust::host_vector<int> h_counter = d_counter;
    result.found_count = h_counter[0];
    result.truncated = result.found_count > max_pairs;
    int copy_count = std::min(result.found_count, max_pairs);
    result.pairs.resize(copy_count);
    thrust::copy(d_pairs.begin(), d_pairs.begin() + copy_count, result.pairs.begin());
    return result;
}

OverlapQueryResult query_aabb_overlaps_cross(const GpuBvh& left, const GpuBvh& right,
                                             int max_pairs) {
    OverlapQueryResult result;
    if (!left.is_valid() || !right.is_valid() || max_pairs <= 0) return result;

    GpuBvhView left_view = left.device_view();
    const AABB* right_bounds = right.overlap_bounds_device_ptr();
    const int* right_primitive_ids = right.overlap_primitive_ids_device_ptr();
    const uint32_t* right_sorted_indices = right.overlap_sorted_indices_device_ptr();

    thrust::device_vector<PrimitivePair> d_pairs(max_pairs);
    thrust::device_vector<int> d_counter(1, 0);

    overlap_cross_kernel<<<grid_size(right.primitive_count, kQueryBlockSize), kQueryBlockSize>>>(
        left_view.nodes,
        left_view.node_count,
        right_bounds,
        right_primitive_ids,
        right_sorted_indices,
        right.primitive_count,
        thrust::raw_pointer_cast(d_pairs.data()),
        max_pairs,
        thrust::raw_pointer_cast(d_counter.data()));
    check_cuda_or_throw("overlap_cross_kernel");
    synchronize_cuda_or_throw("overlap_cross_kernel synchronize");

    thrust::host_vector<int> h_counter = d_counter;
    result.found_count = h_counter[0];
    result.truncated = result.found_count > max_pairs;
    int copy_count = std::min(result.found_count, max_pairs);
    result.pairs.resize(copy_count);
    thrust::copy(d_pairs.begin(), d_pairs.begin() + copy_count, result.pairs.begin());
    return result;
}
