#include "light/light_distribution.h"

#include <algorithm>
#include <utility>

namespace {

std::vector<AliasTableEntry> build_alias_table_from_weights(
        const std::vector<Float>& weights) {
    std::vector<AliasTableEntry> table(weights.size());
    int count = static_cast<int>(weights.size());
    if (count <= 0) return table;

    Float total = 0;
    for (Float weight : weights) {
        total += fmaxf(Float(0), weight);
    }
    if (total <= 0) return {};

    std::vector<Float> scaled(count, 0);
    std::vector<int> small;
    std::vector<int> large;
    small.reserve(weights.size());
    large.reserve(weights.size());

    for (int i = 0; i < count; ++i) {
        Float pmf = fmaxf(Float(0), weights[i]) / total;
        table[i].pmf = pmf;
        table[i].alias = i;
        scaled[i] = pmf * Float(count);
        if (scaled[i] < Float(1)) {
            small.push_back(i);
        } else {
            large.push_back(i);
        }
    }

    while (!small.empty() && !large.empty()) {
        int less = small.back();
        small.pop_back();
        int more = large.back();
        large.pop_back();

        table[less].probability = scaled[less];
        table[less].alias = more;
        scaled[more] = (scaled[more] + scaled[less]) - Float(1);
        if (scaled[more] < Float(1)) {
            small.push_back(more);
        } else {
            large.push_back(more);
        }
    }

    for (int index : large) {
        table[index].probability = Float(1);
        table[index].alias = index;
    }
    for (int index : small) {
        table[index].probability = Float(1);
        table[index].alias = index;
    }

    return table;
}

} // namespace

void build_light_bvh(LightBvhTable& bvh,
                     const LightSelectionTable& selection,
                     const std::vector<LightBvhBuildItem>& build_items) {
    bvh.nodes.clear();
    bvh.root = -1;
    bvh.leaf_nodes_by_light_index.clear();
    bvh.non_bvh_light_indices.clear();
    bvh.non_bvh_light_distribution = DiscreteSamplingTable{};
    if (build_items.empty() || selection.lights.empty()) return;

    std::vector<unsigned char> seen(selection.lights.size(), 0);
    for (const LightBvhBuildItem& item : build_items) {
        if (item.bounds.is_empty()
            || item.light_index < 0
            || item.light_index >= static_cast<int>(selection.lights.size())
            || item.power <= 0
            || seen[static_cast<std::size_t>(item.light_index)] != 0) {
            bvh.nodes.clear();
            return;
        }
        seen[static_cast<std::size_t>(item.light_index)] = 1;
    }

    Float non_bvh_weight = 0;
    bvh.non_bvh_light_indices.reserve(selection.lights.size() - build_items.size());
    bvh.non_bvh_light_distribution.cdf.reserve(
        selection.lights.size() - build_items.size());
    for (int light_index = 0;
         light_index < static_cast<int>(selection.lights.size());
         ++light_index) {
        if (seen[static_cast<std::size_t>(light_index)] != 0) continue;
        Float previous =
            light_index == 0 ? 0 : selection.power_distribution.cdf[light_index - 1];
        Float weight = selection.power_distribution.cdf[light_index] - previous;
        if (weight <= 0) continue;
        non_bvh_weight += weight;
        bvh.non_bvh_light_indices.push_back(light_index);
        bvh.non_bvh_light_distribution.cdf.push_back(non_bvh_weight);
    }
    bvh.non_bvh_light_distribution.total_weight = non_bvh_weight;
    finalize_discrete_sampling_table(bvh.non_bvh_light_distribution);

    std::vector<LightBvhBuildItem> items = build_items;
    bvh.nodes.reserve(items.size() * 2 - 1);
    bvh.leaf_nodes_by_light_index.assign(selection.lights.size(), -1);

    auto build_range = [&](auto&& self, int begin, int end, int parent) -> int {
        int node_index = static_cast<int>(bvh.nodes.size());
        bvh.nodes.push_back(LightBvhNode{});

        AABB bounds;
        AABB centroid_bounds;
        Float power = 0;
        for (int i = begin; i < end; ++i) {
            bounds.expand(items[i].bounds);
            centroid_bounds.expand(items[i].centroid);
            power += items[i].power;
        }

        bvh.nodes[node_index].bounds = bounds;
        bvh.nodes[node_index].parent = parent;
        bvh.nodes[node_index].power = power;

        if (end - begin == 1) {
            bvh.nodes[node_index].light_index = items[begin].light_index;
            bvh.leaf_nodes_by_light_index[
                static_cast<std::size_t>(items[begin].light_index)] = node_index;
            return node_index;
        }

        int axis = centroid_bounds.max_extent_axis();
        int mid = begin + (end - begin) / 2;
        std::nth_element(
            items.begin() + begin, items.begin() + mid, items.begin() + end,
            [axis](const LightBvhBuildItem& a, const LightBvhBuildItem& b) {
                return a.centroid[axis] < b.centroid[axis];
            });

        int left = self(self, begin, mid, node_index);
        int right = self(self, mid, end, node_index);
        bvh.nodes[node_index].left = left;
        bvh.nodes[node_index].right = right;
        bvh.nodes[node_index].light_index = -1;
        bvh.nodes[node_index].power =
            bvh.nodes[left].power
            + bvh.nodes[right].power;
        bvh.nodes[node_index].bounds = union_aabb(
            bvh.nodes[left].bounds,
            bvh.nodes[right].bounds);
        return node_index;
    };

    bvh.root = build_range(build_range, 0, static_cast<int>(items.size()), -1);
}

std::vector<AliasTableEntry> build_alias_table_from_cdf(
        const std::vector<Float>& cdf, Float total_weight) {
    if (cdf.empty() || total_weight <= 0) return {};
    std::vector<Float> weights;
    weights.reserve(cdf.size());
    Float previous = 0;
    for (Float value : cdf) {
        weights.push_back(value - previous);
        previous = value;
    }
    return build_alias_table_from_weights(weights);
}

void finalize_discrete_sampling_table(DiscreteSamplingTable& table) {
    table.alias_table = build_alias_table_from_cdf(table.cdf, table.total_weight);
}

void finalize_area_light_sampling_table(AreaLightSamplingTable& table) {
    finalize_discrete_sampling_table(table.area_distribution);
}

void finalize_light_selection_table(LightSelectionTable& table) {
    finalize_discrete_sampling_table(table.power_distribution);
}

void finalize_prepared_light_sampling_tables(PreparedLightSampling& sampling) {
    finalize_area_light_sampling_table(sampling.area_lights);
    finalize_light_selection_table(sampling.light_selection);
    finalize_discrete_sampling_table(
        sampling.light_bvh.non_bvh_light_distribution);
}

AreaLightSamplingTable build_area_light_sampling_table(
        const std::vector<AreaLightPrimitive>& lights) {
    AreaLightSamplingTable table;
    table.primitive_refs.reserve(lights.size());
    table.primitive_areas.reserve(lights.size());
    table.area_distribution.cdf.reserve(lights.size());

    Float accumulated_area = 0;
    for (const AreaLightPrimitive& light : lights) {
        if (!primitive_ref_valid(light.primitive_ref) || light.area <= 0) {
            continue;
        }

        accumulated_area += light.area;
        table.primitive_refs.push_back(light.primitive_ref);
        table.primitive_areas.push_back(light.area);
        table.area_distribution.cdf.push_back(accumulated_area);
    }

    table.area_distribution.total_weight = accumulated_area;
    finalize_area_light_sampling_table(table);
    return table;
}

void append_light_selection_entry(LightSelectionTable& selection,
                                  const Light& light,
                                  Float power) {
    if (power <= 0) return;
    selection.power_distribution.total_weight += power;
    selection.lights.push_back(light);
    selection.power_distribution.cdf.push_back(selection.power_distribution.total_weight);
}
