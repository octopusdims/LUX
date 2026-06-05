#pragma once

#ifndef LUX_LIGHT_DISTRIBUTION_H
#define LUX_LIGHT_DISTRIBUTION_H

#include <cstdint>
#include <limits>
#include <vector>

#include "core/aabb.cuh"
#include "core/types.h"
#include "core/vec3.cuh"
#include "light/light.h"
#include "scene/scene_stamp.h"

struct AreaLightPrimitive {
    PrimitiveRef primitive_ref;
    Float area = 0;
};

struct AliasTableEntry {
    Float probability = 1;
    int alias = 0;
    Float pmf = 0;
};

enum class LightSamplerKind : unsigned char {
    Uniform,
    Power,
    Bvh,
};

struct LightBvhNode {
    AABB bounds;
    int left = -1;
    int right = -1;
    int parent = -1;
    int light_index = -1;
    Float power = 0;
};

struct LightBvhBuildItem {
    AABB bounds;
    vec3 centroid = vec3(0);
    int light_index = -1;
    Float power = 0;
};

struct DiscreteSamplingTable {
    std::vector<Float> cdf;
    std::vector<AliasTableEntry> alias_table;
    Float total_weight = 0;

    bool empty() const {
        return cdf.empty() || total_weight <= 0;
    }
};

struct AreaLightSamplingTable {
    std::vector<PrimitiveRef> primitive_refs;
    std::vector<Float> primitive_areas;
    DiscreteSamplingTable area_distribution;

    bool empty() const {
        return primitive_refs.empty() || area_distribution.total_weight <= 0;
    }
};

struct LightSelectionTable {
    std::vector<Light> lights;
    DiscreteSamplingTable power_distribution;

    bool empty() const {
        return lights.empty() || power_distribution.total_weight <= 0;
    }
};

struct LightBvhTable {
    std::vector<LightBvhNode> nodes;
    int root = -1;
    std::vector<int> leaf_nodes_by_light_index;
    std::vector<int> non_bvh_light_indices;
    DiscreteSamplingTable non_bvh_light_distribution;

    bool valid() const {
        return root >= 0 && !nodes.empty() && !leaf_nodes_by_light_index.empty();
    }
};

struct SurfaceLightLookupTable {
    std::vector<int> primitive_light_indices;
};

struct PreparedLightSampling {
    SceneStamp source_scene;
    AreaLightSamplingTable area_lights;
    LightSelectionTable light_selection;
    LightBvhTable light_bvh;
    SurfaceLightLookupTable surface_lookup;

    bool empty() const {
        return area_lights.empty() && light_selection.empty();
    }
};

AreaLightSamplingTable build_area_light_sampling_table(
    const std::vector<AreaLightPrimitive>& lights);
void append_light_selection_entry(LightSelectionTable& selection,
                                  const Light& light,
                                  Float power);
std::vector<AliasTableEntry> build_alias_table_from_cdf(
    const std::vector<Float>& cdf, Float total_weight);
void finalize_discrete_sampling_table(DiscreteSamplingTable& table);
void finalize_area_light_sampling_table(AreaLightSamplingTable& table);
void finalize_light_selection_table(LightSelectionTable& table);
void finalize_prepared_light_sampling_tables(PreparedLightSampling& sampling);
void build_light_bvh(LightBvhTable& bvh,
                     const LightSelectionTable& selection,
                     const std::vector<LightBvhBuildItem>& build_items);

#endif // LUX_LIGHT_DISTRIBUTION_H
