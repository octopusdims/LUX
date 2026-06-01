#include "bvh/gpu_bvh.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string> 
#include <unordered_map>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace {

constexpr uint32_t kLeafMask = 0x80000000u;
constexpr uint32_t kIndexMask = 0x7fffffffu;
constexpr int kBlockSize = 256;

struct RadixNode {
    AABB bounds[2];
    uint32_t parent = 0;
    uint32_t left = 0;
    uint32_t right = 0;
    uint32_t fence = 0;
};

struct IndexRange {
    int first;
    int last;
};

struct AABBUnion {
    LuxHDInline AABB operator()(const AABB& a, const AABB& b) const {
        AABB out = a;
        out.expand(b);
        return out;
    }
};

LuxHDInline uint32_t expand_bits_10_device(uint32_t value) {
    value &= 0x000003ffu;
    value = (value * 0x00010001u) & 0xff0000ffu;
    value = (value * 0x00000101u) & 0x0f00f00fu;
    value = (value * 0x00000011u) & 0xc30c30c3u;
    value = (value * 0x00000005u) & 0x49249249u;
    return value;
}

LuxHDInline uint32_t morton_encode_3d_device(uint32_t x, uint32_t y, uint32_t z) {
    return (expand_bits_10_device(x) << 2)
         | (expand_bits_10_device(y) << 1)
         | expand_bits_10_device(z);
}

LuxHDInline uint32_t quantize_morton_coord_device(Float value) {
    value = fmaxf(Float(0), fminf(Float(1023), value * Float(1024)));
    return static_cast<uint32_t>(value);
}

LuxHDInline uint64_t merged_key_device(uint32_t morton_code, int sorted_index) {
    return (uint64_t(morton_code) << 32u) | uint64_t(uint32_t(sorted_index));
}

LuxDeviceInline int common_prefix_device(const uint32_t* morton_codes, int lhs, int rhs) {
    uint64_t lhs_key = merged_key_device(morton_codes[lhs], lhs);
    uint64_t rhs_key = merged_key_device(morton_codes[rhs], rhs);
    return __clzll(lhs_key ^ rhs_key);
}

LuxDeviceInline IndexRange determine_range_device(const uint32_t* morton_codes,
                                                   int num_primitives,
                                                   int idx) {
    if (idx == 0) return IndexRange{0, num_primitives - 1};

    int left_delta = common_prefix_device(morton_codes, idx, idx - 1);
    int right_delta = common_prefix_device(morton_codes, idx, idx + 1);
    int direction = (right_delta > left_delta) ? 1 : -1;
    int min_delta = min(left_delta, right_delta);

    int max_length = 2;
    while (true) {
        int candidate = idx + direction * max_length;
        if (candidate < 0 || candidate >= num_primitives) break;
        if (common_prefix_device(morton_codes, idx, candidate) <= min_delta) break;
        max_length <<= 1;
    }

    int length = 0;
    for (int step = max_length >> 1; step > 0; step >>= 1) {
        int candidate = idx + (length + step) * direction;
        if (candidate < 0 || candidate >= num_primitives) continue;
        if (common_prefix_device(morton_codes, idx, candidate) > min_delta) {
            length += step;
        }
    }

    int other = idx + length * direction;
    return direction < 0 ? IndexRange{other, idx} : IndexRange{idx, other};
}

LuxDeviceInline int find_split_device(const uint32_t* morton_codes, int first, int last) {
    int node_delta = common_prefix_device(morton_codes, first, last);
    int split = first;
    int stride = last - first;

    do {
        stride = (stride + 1) >> 1;
        int middle = split + stride;
        if (middle < last && common_prefix_device(morton_codes, first, middle) > node_delta) {
            split = middle;
        }
    } while (stride > 1);

    return split;
}

LuxDeviceInline int subtree_size_device(uint32_t child, const int* subtree_sizes) {
    if (child & kLeafMask) return 1;
    return subtree_sizes[child & kIndexMask];
}

LuxDeviceInline int preorder_of_child_device(uint32_t child,
                                              const int* internal_preorder,
                                              const int* leaf_preorder) {
    uint32_t index = child & kIndexMask;
    return (child & kLeafMask) ? leaf_preorder[index] : internal_preorder[index];
}

__global__ void morton_kernel(const AABB* bounds,
                              uint32_t* morton_codes,
                              uint32_t* input_indices,
                              AABB scene_bounds,
                              int num_primitives) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_primitives) return;

    vec3 offset = scene_bounds.normalized_offset(bounds[tid].centroid());
    morton_codes[tid] = morton_encode_3d_device(
        quantize_morton_coord_device(offset.x),
        quantize_morton_coord_device(offset.y),
        quantize_morton_coord_device(offset.z));
    input_indices[tid] = static_cast<uint32_t>(tid);
}

__global__ void build_internal_kernel(RadixNode* nodes,
                                      uint32_t* leaf_parents,
                                      const uint32_t* morton_codes,
                                      int num_primitives) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_primitives - 1) return;

    IndexRange range = determine_range_device(morton_codes, num_primitives, tid);
    nodes[tid].fence = (tid == range.first)
        ? static_cast<uint32_t>(range.last)
        : static_cast<uint32_t>(range.first);

    int split = find_split_device(morton_codes, range.first, range.last);

    if (range.first == split) {
        nodes[tid].left = kLeafMask | static_cast<uint32_t>(split);
        leaf_parents[split] = static_cast<uint32_t>(tid);
    } else {
        nodes[tid].left = static_cast<uint32_t>(split);
        nodes[split].parent = static_cast<uint32_t>(tid);
    }

    if (range.last == split + 1) {
        nodes[tid].right = kLeafMask | static_cast<uint32_t>(split + 1);
        leaf_parents[split + 1] = static_cast<uint32_t>(tid) | kLeafMask;
    } else {
        nodes[tid].right = static_cast<uint32_t>(split + 1);
        nodes[split + 1].parent = static_cast<uint32_t>(tid) | kLeafMask;
    }
}

__global__ void merge_bounds_kernel(RadixNode* nodes,
                                    const uint32_t* leaf_parents,
                                    const AABB* input_bounds,
                                    const uint32_t* sorted_input_indices,
                                    int* flags,
                                    int num_primitives) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_primitives) return;

    int depth = 1;
    AABB last = input_bounds[sorted_input_indices[tid]];
    int parent = static_cast<int>(leaf_parents[tid]);

    while (true) {
        int is_right = (parent & int(kLeafMask)) != 0;
        parent &= int(kIndexMask);
        nodes[parent].bounds[is_right] = last;
        __threadfence();

        int other_depth = atomicOr(flags + parent, depth);
        if (!other_depth) return;

        depth = is_right ? max(depth + 1, other_depth) : max(depth, other_depth + 1);

        if (parent == 0) {
            flags[0] = depth;
            return;
        }

        last.expand(nodes[parent].bounds[1 - is_right]);
        parent = static_cast<int>(nodes[parent].parent);
    }
}

__global__ void subtree_sizes_kernel(RadixNode* nodes,
                                     const uint32_t* leaf_parents,
                                     int* flags,
                                     int* subtree_sizes,
                                     int num_primitives) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_primitives) return;

    int parent = static_cast<int>(leaf_parents[tid]);
    int size = 1;

    while (true) {
        parent &= int(kIndexMask);
        int sibling_size = atomicExch(flags + parent, size);
        if (sibling_size == 0) return;

        size = size + sibling_size + 1;
        subtree_sizes[parent] = size;
        __threadfence();

        if (parent == 0) return;
        parent = static_cast<int>(nodes[parent].parent);
    }
}

__global__ void internal_layout_kernel(const RadixNode* nodes,
                                       const int* subtree_sizes,
                                       int* internal_preorder,
                                       int* internal_escape_parent,
                                       int num_internal_nodes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_internal_nodes) return;

    int preorder = 0;
    int escape_parent = -1;
    int node_index = tid;

    while (node_index != 0) {
        uint32_t parent_info = nodes[node_index].parent;
        bool is_right = (parent_info & kLeafMask) != 0;
        int parent_index = static_cast<int>(parent_info & kIndexMask);

        if (is_right) {
            preorder += 1 + subtree_size_device(nodes[parent_index].left, subtree_sizes);
        } else {
            preorder += 1;
            if (escape_parent < 0) escape_parent = parent_index;
        }

        node_index = parent_index;
    }

    internal_preorder[tid] = preorder;
    internal_escape_parent[tid] = escape_parent;
}

__global__ void leaf_layout_kernel(const RadixNode* nodes,
                                   const uint32_t* leaf_parents,
                                   const int* subtree_sizes,
                                   int* leaf_preorder,
                                   int* leaf_escape_parent,
                                   int num_primitives) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_primitives) return;

    int preorder = 0;
    int escape_parent = -1;
    uint32_t parent_info = leaf_parents[tid];
    bool is_right = (parent_info & kLeafMask) != 0;
    int parent_index = static_cast<int>(parent_info & kIndexMask);

    if (is_right) {
        preorder += 1 + subtree_size_device(nodes[parent_index].left, subtree_sizes);
    } else {
        preorder += 1;
        escape_parent = parent_index;
    }

    while (parent_index != 0) {
        uint32_t grandparent_info = nodes[parent_index].parent;
        bool grandparent_is_right = (grandparent_info & kLeafMask) != 0;
        int grandparent_index = static_cast<int>(grandparent_info & kIndexMask);

        if (grandparent_is_right) {
            preorder += 1 + subtree_size_device(nodes[grandparent_index].left, subtree_sizes);
        } else {
            preorder += 1;
            if (escape_parent < 0) escape_parent = grandparent_index;
        }

        parent_index = grandparent_index;
    }

    leaf_preorder[tid] = preorder;
    leaf_escape_parent[tid] = escape_parent;
}

__global__ void write_internal_stackless_kernel(const RadixNode* nodes,
                                                GpuBvhNode* stackless_nodes,
                                                const int* internal_preorder,
                                                const int* internal_escape_parent,
                                                const int* leaf_preorder,
                                                int num_internal_nodes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_internal_nodes) return;

    int preorder = internal_preorder[tid];
    int escape = -1;
    int escape_parent = internal_escape_parent[tid];
    if (escape_parent >= 0) {
        escape = preorder_of_child_device(nodes[escape_parent].right,
                                          internal_preorder,
                                          leaf_preorder);
    }

    AABB bounds = nodes[tid].bounds[0];
    bounds.expand(nodes[tid].bounds[1]);
    stackless_nodes[preorder] = GpuBvhNode{bounds, -1, escape};
}

__global__ void write_leaf_stackless_kernel(const RadixNode* nodes,
                                            GpuBvhNode* stackless_nodes,
                                            const AABB* input_bounds,
                                            const int* primitive_ids,
                                            const uint32_t* sorted_input_indices,
                                            const int* internal_preorder,
                                            const int* leaf_preorder,
                                            const int* leaf_escape_parent,
                                            int num_primitives) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_primitives) return;

    int preorder = leaf_preorder[tid];
    int escape = -1;
    int escape_parent = leaf_escape_parent[tid];
    if (escape_parent >= 0) {
        escape = preorder_of_child_device(nodes[escape_parent].right,
                                          internal_preorder,
                                          leaf_preorder);
    }

    uint32_t input_index = sorted_input_indices[tid];
    stackless_nodes[preorder] = GpuBvhNode{
        input_bounds[input_index],
        primitive_ids[input_index],
        escape
    };
}

__global__ void write_single_leaf_kernel(GpuBvhNode* stackless_nodes,
                                         const AABB* input_bounds,
                                         const int* primitive_ids) {
    stackless_nodes[0] = GpuBvhNode{input_bounds[0], primitive_ids[0], -1};
}

__global__ void trace_rays_kernel(GpuBvhView bvh_view,
                                  const Triangle* triangles,
                                  const Ray* rays,
                                  GpuRayHit* hits,
                                  int ray_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= ray_count) return;

    int primitive_id;
    Float hit_t, hit_u, hit_v;
    bool found = intersect_bvh_device(
        bvh_view, triangles, rays[tid], primitive_id, hit_t, hit_u, hit_v);
    hits[tid] = found ? GpuRayHit{primitive_id, hit_t} : GpuRayHit{-1, INFINITY};
}

#include "core/cuda_check.cuh"

int grid_size(int count, int block_size = kBlockSize) {
    return (count + block_size - 1) / block_size;
}

bool aabb_close(const AABB& a, const AABB& b, Float epsilon = 1e-4f) {
    return length2(a.p_min - b.p_min) <= epsilon * epsilon
        && length2(a.p_max - b.p_max) <= epsilon * epsilon;
}

void set_validation_error(std::string* error_message, const std::string& message) {
    if (error_message) *error_message = message;
}

} // namespace

struct GpuBvhStorage {
    thrust::device_vector<AABB> input_bounds;
    thrust::device_vector<int> primitive_ids;
    thrust::device_vector<uint32_t> morton_codes;
    thrust::device_vector<uint32_t> sorted_input_indices;
    thrust::device_vector<uint32_t> leaf_parents;
    thrust::device_vector<RadixNode> radix_nodes;
    thrust::device_vector<int> flags;
    thrust::device_vector<int> subtree_sizes;
    thrust::device_vector<int> internal_preorder;
    thrust::device_vector<int> internal_escape_parent;
    thrust::device_vector<int> leaf_preorder;
    thrust::device_vector<int> leaf_escape_parent;
    thrust::device_vector<GpuBvhNode> stackless_nodes;
};

GpuBvh::GpuBvh() : storage(std::make_unique<GpuBvhStorage>()) {}
GpuBvh::~GpuBvh() = default;
GpuBvh::GpuBvh(GpuBvh&& other) noexcept = default;
GpuBvh& GpuBvh::operator=(GpuBvh&& other) noexcept = default;

bool GpuBvh::is_valid() const {
    if (!storage || primitive_count <= 0) return false;
    const GpuBvhStorage& s = *storage;
    int node_count = primitive_count * 2 - 1;
    return static_cast<int>(s.input_bounds.size()) == primitive_count
        && static_cast<int>(s.primitive_ids.size()) == primitive_count
        && static_cast<int>(s.stackless_nodes.size()) == node_count;
}

GpuBvhView GpuBvh::device_view() const {
    if (!is_valid()) return GpuBvhView{};
    const GpuBvhStorage& s = *storage;
    return GpuBvhView{
        thrust::raw_pointer_cast(s.stackless_nodes.data()),
        static_cast<int>(s.stackless_nodes.size()),
        root_bounds
    };
}

const AABB* GpuBvh::overlap_bounds_device_ptr() const {
    return is_valid() ? thrust::raw_pointer_cast(storage->input_bounds.data()) : nullptr;
}

const int* GpuBvh::overlap_primitive_ids_device_ptr() const {
    return is_valid() ? thrust::raw_pointer_cast(storage->primitive_ids.data()) : nullptr;
}

const uint32_t* GpuBvh::overlap_sorted_indices_device_ptr() const {
    return is_valid() ? thrust::raw_pointer_cast(storage->sorted_input_indices.data()) : nullptr;
}

static bool rebuild_stackless_layout(GpuBvh& bvh) {
    if (!bvh.is_valid()) return true;

    int n = bvh.primitive_count;
    int internal_count = n - 1;
    int node_count = n * 2 - 1;
    GpuBvhStorage& s = *bvh.storage;

    if (n == 1) {
        write_single_leaf_kernel<<<1, 1>>>(
            thrust::raw_pointer_cast(s.stackless_nodes.data()),
            thrust::raw_pointer_cast(s.input_bounds.data()),
            thrust::raw_pointer_cast(s.primitive_ids.data()));
        check_cuda_or_throw("write_single_leaf_kernel");
        synchronize_cuda_or_throw("write_single_leaf_kernel synchronize");
        return true;
    }

    thrust::fill(s.flags.begin(), s.flags.end(), 0);
    merge_bounds_kernel<<<grid_size(n), kBlockSize>>>(
        thrust::raw_pointer_cast(s.radix_nodes.data()),
        thrust::raw_pointer_cast(s.leaf_parents.data()),
        thrust::raw_pointer_cast(s.input_bounds.data()),
        thrust::raw_pointer_cast(s.sorted_input_indices.data()),
        thrust::raw_pointer_cast(s.flags.data()),
        n);
    check_cuda_or_throw("merge_bounds_kernel");

    thrust::fill(s.flags.begin(), s.flags.end(), 0);
    thrust::fill(s.subtree_sizes.begin(), s.subtree_sizes.end(), 0);
    subtree_sizes_kernel<<<grid_size(n), kBlockSize>>>(
        thrust::raw_pointer_cast(s.radix_nodes.data()),
        thrust::raw_pointer_cast(s.leaf_parents.data()),
        thrust::raw_pointer_cast(s.flags.data()),
        thrust::raw_pointer_cast(s.subtree_sizes.data()),
        n);
    check_cuda_or_throw("subtree_sizes_kernel");

    internal_layout_kernel<<<grid_size(internal_count), kBlockSize>>>(
        thrust::raw_pointer_cast(s.radix_nodes.data()),
        thrust::raw_pointer_cast(s.subtree_sizes.data()),
        thrust::raw_pointer_cast(s.internal_preorder.data()),
        thrust::raw_pointer_cast(s.internal_escape_parent.data()),
        internal_count);
    check_cuda_or_throw("internal_layout_kernel");

    leaf_layout_kernel<<<grid_size(n), kBlockSize>>>(
        thrust::raw_pointer_cast(s.radix_nodes.data()),
        thrust::raw_pointer_cast(s.leaf_parents.data()),
        thrust::raw_pointer_cast(s.subtree_sizes.data()),
        thrust::raw_pointer_cast(s.leaf_preorder.data()),
        thrust::raw_pointer_cast(s.leaf_escape_parent.data()),
        n);
    check_cuda_or_throw("leaf_layout_kernel");

    write_internal_stackless_kernel<<<grid_size(internal_count), kBlockSize>>>(
        thrust::raw_pointer_cast(s.radix_nodes.data()),
        thrust::raw_pointer_cast(s.stackless_nodes.data()),
        thrust::raw_pointer_cast(s.internal_preorder.data()),
        thrust::raw_pointer_cast(s.internal_escape_parent.data()),
        thrust::raw_pointer_cast(s.leaf_preorder.data()),
        internal_count);
    check_cuda_or_throw("write_internal_stackless_kernel");

    write_leaf_stackless_kernel<<<grid_size(n), kBlockSize>>>(
        thrust::raw_pointer_cast(s.radix_nodes.data()),
        thrust::raw_pointer_cast(s.stackless_nodes.data()),
        thrust::raw_pointer_cast(s.input_bounds.data()),
        thrust::raw_pointer_cast(s.primitive_ids.data()),
        thrust::raw_pointer_cast(s.sorted_input_indices.data()),
        thrust::raw_pointer_cast(s.internal_preorder.data()),
        thrust::raw_pointer_cast(s.leaf_preorder.data()),
        thrust::raw_pointer_cast(s.leaf_escape_parent.data()),
        n);
    check_cuda_or_throw("write_leaf_stackless_kernel");

    synchronize_cuda_or_throw("rebuild_stackless_layout synchronize");
    return node_count > 0;
}

GpuBvh build_gpu_lbvh(const std::vector<BvhBuildPrimitive>& primitives) {
    GpuBvh bvh;
    if (primitives.empty()) return bvh;

    int n = static_cast<int>(primitives.size());
    int internal_count = n - 1;
    int node_count = n * 2 - 1;

    std::vector<AABB> bounds(n);
    std::vector<int> primitive_ids(n);
    for (int i = 0; i < n; ++i) {
        bounds[i] = primitives[i].bounds;
        primitive_ids[i] = primitives[i].primitive_id;
    }

    GpuBvhStorage& s = *bvh.storage;
    s.input_bounds = bounds;
    s.primitive_ids = primitive_ids;
    s.morton_codes.resize(n);
    s.sorted_input_indices.resize(n);
    s.leaf_parents.resize(n);
    s.radix_nodes.resize(std::max(0, internal_count));
    s.flags.resize(std::max(1, internal_count));
    s.subtree_sizes.resize(std::max(1, internal_count));
    s.internal_preorder.resize(std::max(1, internal_count));
    s.internal_escape_parent.resize(std::max(1, internal_count));
    s.leaf_preorder.resize(n);
    s.leaf_escape_parent.resize(n);
    s.stackless_nodes.resize(node_count);

    bvh.primitive_count = n;
    bvh.root_bounds = thrust::reduce(
        s.input_bounds.begin(), s.input_bounds.end(), AABB{}, AABBUnion{});

    if (n == 1) {
        rebuild_stackless_layout(bvh);
        return bvh;
    }

    morton_kernel<<<grid_size(n), kBlockSize>>>(
        thrust::raw_pointer_cast(s.input_bounds.data()),
        thrust::raw_pointer_cast(s.morton_codes.data()),
        thrust::raw_pointer_cast(s.sorted_input_indices.data()),
        bvh.root_bounds,
        n);
    check_cuda_or_throw("morton_kernel");

    thrust::stable_sort_by_key(s.morton_codes.begin(),
                               s.morton_codes.end(),
                               s.sorted_input_indices.begin());

    build_internal_kernel<<<grid_size(internal_count), kBlockSize>>>(
        thrust::raw_pointer_cast(s.radix_nodes.data()),
        thrust::raw_pointer_cast(s.leaf_parents.data()),
        thrust::raw_pointer_cast(s.morton_codes.data()),
        n);
    check_cuda_or_throw("build_internal_kernel");

    rebuild_stackless_layout(bvh);
    return bvh;
}

bool refit_gpu_lbvh(GpuBvh& bvh, const std::vector<BvhBuildPrimitive>& primitives) {
    if (static_cast<int>(primitives.size()) != bvh.primitive_count) return false;
    if (!bvh.is_valid()) return primitives.empty();

    std::vector<AABB> bounds(primitives.size());
    thrust::host_vector<int> current_ids = bvh.storage->primitive_ids;
    for (int i = 0; i < static_cast<int>(primitives.size()); ++i) {
        if (current_ids[i] != primitives[i].primitive_id) return false;
        bounds[i] = primitives[i].bounds;
    }

    GpuBvhStorage& s = *bvh.storage;
    s.input_bounds = bounds;
    bvh.root_bounds = thrust::reduce(
        s.input_bounds.begin(), s.input_bounds.end(), AABB{}, AABBUnion{});
    return rebuild_stackless_layout(bvh);
}

CpuBvhTree download_bvh_for_validation(const GpuBvh& bvh) {
    CpuBvhTree host_bvh;
    if (!bvh.is_valid()) return host_bvh;

    const GpuBvhStorage& s = *bvh.storage;
    int n = bvh.primitive_count;
    int internal_count = n - 1;

    thrust::host_vector<AABB> bounds = s.input_bounds;
    thrust::host_vector<int> primitive_ids = s.primitive_ids;

    host_bvh.root_node = 0;
    if (n == 1) {
        host_bvh.primitive_ids.push_back(primitive_ids[0]);
        host_bvh.nodes.push_back(CpuBvhNode{bounds[0], -1, -1, 0, 1});
        return host_bvh;
    }

    thrust::host_vector<RadixNode> radix_nodes = s.radix_nodes;
    thrust::host_vector<uint32_t> sorted_input_indices = s.sorted_input_indices;

    host_bvh.nodes.resize(n * 2 - 1);
    host_bvh.primitive_ids.reserve(n);

    for (int node_index = 0; node_index < internal_count; ++node_index) {
        const RadixNode& node = radix_nodes[node_index];
        int left_index = (node.left & kLeafMask)
            ? internal_count + static_cast<int>(node.left & kIndexMask)
            : static_cast<int>(node.left);
        int right_index = (node.right & kLeafMask)
            ? internal_count + static_cast<int>(node.right & kIndexMask)
            : static_cast<int>(node.right);

        AABB node_bounds = node.bounds[0];
        node_bounds.expand(node.bounds[1]);
        host_bvh.nodes[node_index].bounds = node_bounds;
        host_bvh.nodes[node_index].left = left_index;
        host_bvh.nodes[node_index].right = right_index;
    }

    for (int leaf_index = 0; leaf_index < n; ++leaf_index) {
        int input_index = static_cast<int>(sorted_input_indices[leaf_index]);
        int node_index = internal_count + leaf_index;
        host_bvh.primitive_ids.push_back(primitive_ids[input_index]);
        host_bvh.nodes[node_index].bounds = bounds[input_index];
        host_bvh.nodes[node_index].first_primitive = leaf_index;
        host_bvh.nodes[node_index].primitive_count = 1;
    }

    return host_bvh;
}

bool validate_gpu_bvh_structure(const GpuBvh& bvh, std::string* error_message) {
    if (!bvh.is_valid()) {
        set_validation_error(error_message, "BVH is not valid");
        return false;
    }

    const GpuBvhStorage& s = *bvh.storage;
    int n = bvh.primitive_count;
    int internal_count = n - 1;
    int node_count = n * 2 - 1;

    if (static_cast<int>(s.stackless_nodes.size()) != node_count) {
        set_validation_error(error_message, "stackless node count mismatch");
        return false;
    }
    if (static_cast<int>(s.input_bounds.size()) != n
        || static_cast<int>(s.primitive_ids.size()) != n
        || static_cast<int>(s.sorted_input_indices.size()) != n) {
        set_validation_error(error_message, "primitive storage size mismatch");
        return false;
    }

    thrust::host_vector<AABB> bounds = s.input_bounds;
    thrust::host_vector<int> primitive_ids = s.primitive_ids;
    thrust::host_vector<uint32_t> sorted_input_indices = s.sorted_input_indices;
    thrust::host_vector<GpuBvhNode> stackless_nodes = s.stackless_nodes;

    std::vector<int> seen_inputs(n, 0);
    for (int i = 0; i < n; ++i) {
        int input_index = static_cast<int>(sorted_input_indices[i]);
        if (input_index < 0 || input_index >= n) {
            set_validation_error(error_message, "sorted input index out of range");
            return false;
        }
        if (++seen_inputs[input_index] != 1) {
            set_validation_error(error_message, "duplicate sorted input index");
            return false;
        }
    }

    AABB expected_root;
    for (int i = 0; i < n; ++i) expected_root.expand(bounds[i]);
    if (!aabb_close(expected_root, bvh.root_bounds)) {
        set_validation_error(error_message, "root bounds mismatch");
        return false;
    }
    if (!aabb_close(stackless_nodes[0].bounds, bvh.root_bounds)) {
        set_validation_error(error_message, "stackless root bounds mismatch");
        return false;
    }

    int leaf_count = 0;
    for (int i = 0; i < node_count; ++i) {
        const GpuBvhNode& node = stackless_nodes[i];
        if (node.escape < -1 || node.escape >= node_count) {
            set_validation_error(error_message, "stackless escape index out of range");
            return false;
        }
        if (node.primitive_id >= 0) ++leaf_count;
    }
    if (leaf_count != n) {
        set_validation_error(error_message, "stackless leaf count mismatch");
        return false;
    }

    if (n == 1) {
        if (stackless_nodes[0].primitive_id != primitive_ids[0]
            || stackless_nodes[0].escape != -1) {
            set_validation_error(error_message, "single leaf stackless node mismatch");
            return false;
        }
        return true;
    }

    if (static_cast<int>(s.radix_nodes.size()) != internal_count
        || static_cast<int>(s.leaf_parents.size()) != n
        || static_cast<int>(s.morton_codes.size()) != n) {
        set_validation_error(error_message, "radix storage size mismatch");
        return false;
    }

    thrust::host_vector<RadixNode> radix_nodes = s.radix_nodes;
    thrust::host_vector<uint32_t> leaf_parents = s.leaf_parents;
    thrust::host_vector<uint32_t> morton_codes = s.morton_codes;

    for (int i = 1; i < n; ++i) {
        if (morton_codes[i - 1] > morton_codes[i]) {
            set_validation_error(error_message, "Morton codes are not sorted");
            return false;
        }
    }

    for (int child = 1; child < internal_count; ++child) {
        uint32_t parent_info = radix_nodes[child].parent;
        int parent = static_cast<int>(parent_info & kIndexMask);
        bool is_right = (parent_info & kLeafMask) != 0;
        if (parent < 0 || parent >= internal_count) {
            set_validation_error(error_message, "internal parent index out of range");
            return false;
        }
        uint32_t expected_child = is_right ? radix_nodes[parent].right : radix_nodes[parent].left;
        if (expected_child != static_cast<uint32_t>(child)) {
            set_validation_error(error_message, "internal parent/child mismatch");
            return false;
        }
    }

    for (int leaf = 0; leaf < n; ++leaf) {
        uint32_t parent_info = leaf_parents[leaf];
        int parent = static_cast<int>(parent_info & kIndexMask);
        bool is_right = (parent_info & kLeafMask) != 0;
        if (parent < 0 || parent >= internal_count) {
            set_validation_error(error_message, "leaf parent index out of range");
            return false;
        }
        uint32_t expected_leaf = kLeafMask | static_cast<uint32_t>(leaf);
        uint32_t expected_child = is_right ? radix_nodes[parent].right : radix_nodes[parent].left;
        if (expected_child != expected_leaf) {
            set_validation_error(error_message, "leaf parent/child mismatch");
            return false;
        }
    }

    std::vector<uint32_t> stack;
    std::vector<int> visited_internal(internal_count, 0);
    std::vector<int> visited_leaf(n, 0);
    stack.push_back(0);
    while (!stack.empty()) {
        uint32_t child = stack.back();
        stack.pop_back();
        if (child & kLeafMask) {
            int leaf = static_cast<int>(child & kIndexMask);
            if (leaf < 0 || leaf >= n || ++visited_leaf[leaf] != 1) {
                set_validation_error(error_message, "invalid or duplicate leaf traversal");
                return false;
            }
            continue;
        }

        int node = static_cast<int>(child);
        if (node < 0 || node >= internal_count || ++visited_internal[node] != 1) {
            set_validation_error(error_message, "invalid or duplicate internal traversal");
            return false;
        }
        stack.push_back(radix_nodes[node].right);
        stack.push_back(radix_nodes[node].left);
    }

    for (int i = 0; i < internal_count; ++i) {
        if (visited_internal[i] != 1) {
            set_validation_error(error_message, "unvisited internal node");
            return false;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (visited_leaf[i] != 1) {
            set_validation_error(error_message, "unvisited leaf node");
            return false;
        }
    }

    return true;
}

bool validate_gpu_bvh_matches_primitives(
        const GpuBvh& bvh,
        const std::vector<BvhBuildPrimitive>& primitives,
        std::string* error_message) {
    if (primitives.empty()) {
        if (!bvh.is_valid() && bvh.primitive_count == 0) return true;
        set_validation_error(error_message, "BVH contains primitives for an empty scene");
        return false;
    }

    std::string structure_error;
    if (!validate_gpu_bvh_structure(bvh, &structure_error)) {
        set_validation_error(
            error_message,
            structure_error.empty() ? "BVH structure validation failed" : structure_error);
        return false;
    }
    if (bvh.primitive_count != static_cast<int>(primitives.size())) {
        set_validation_error(error_message, "BVH primitive count does not match scene primitive count");
        return false;
    }

    std::unordered_map<int, const BvhBuildPrimitive*> expected;
    expected.reserve(primitives.size());
    for (const BvhBuildPrimitive& primitive : primitives) {
        auto inserted = expected.emplace(primitive.primitive_id, &primitive);
        if (!inserted.second) {
            set_validation_error(error_message, "Scene primitive ids are not unique");
            return false;
        }
    }

    const GpuBvhStorage& s = *bvh.storage;
    thrust::host_vector<AABB> bounds = s.input_bounds;
    thrust::host_vector<int> primitive_ids = s.primitive_ids;
    std::unordered_map<int, int> seen;
    seen.reserve(primitive_ids.size());
    for (int i = 0; i < static_cast<int>(primitive_ids.size()); ++i) {
        int primitive_id = primitive_ids[i];
        auto expected_it = expected.find(primitive_id);
        if (expected_it == expected.end()) {
            set_validation_error(error_message, "BVH contains primitive id not present in scene");
            return false;
        }
        if (++seen[primitive_id] != 1) {
            set_validation_error(error_message, "BVH contains duplicate primitive id");
            return false;
        }
        if (!aabb_close(bounds[i], expected_it->second->bounds)) {
            set_validation_error(error_message, "BVH primitive bounds do not match scene primitive");
            return false;
        }
    }

    if (seen.size() != expected.size()) {
        set_validation_error(error_message, "BVH is missing one or more scene primitives");
        return false;
    }

    return true;
}

std::vector<GpuRayHit> intersect_gpu_bvh_stackless(const GpuBvh& bvh,
                                                   const std::vector<Triangle>& triangles,
                                                   const std::vector<Ray>& rays) {
    std::vector<GpuRayHit> hits(rays.size());
    if (!bvh.is_valid() || rays.empty()) return hits;

    thrust::device_vector<Triangle> d_triangles = triangles;
    thrust::device_vector<Ray> d_rays = rays;
    thrust::device_vector<GpuRayHit> d_hits(rays.size());

    GpuBvhView bvh_view = bvh.device_view();
    trace_rays_kernel<<<grid_size(static_cast<int>(rays.size())), kBlockSize>>>(
        bvh_view,
        thrust::raw_pointer_cast(d_triangles.data()),
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_hits.data()),
        static_cast<int>(rays.size()));
    check_cuda_or_throw("trace_rays_kernel");
    synchronize_cuda_or_throw("trace_rays_kernel synchronize");

    thrust::copy(d_hits.begin(), d_hits.end(), hits.begin());
    return hits;
}
