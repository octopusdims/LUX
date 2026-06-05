#pragma once

#ifndef LUX_LIGHT_SAMPLER_H
#define LUX_LIGHT_SAMPLER_H

#include "core/constants.h"
#include "core/vec2.cuh"
#include "geometry/triangle_mesh_view.h"
#include "light/light_distribution.h"
#include "light/light.h"
#include "material/bsdf.h"
#include "sampler/sample_record.h"
#include "scene/primitive_ref.h"

struct LightSample {
    int triangle_id = -1;
    PrimitiveRef primitive_ref;
    vec3 position = vec3(0);
    vec3 normal = vec3(0);
    vec2 uv = vec2(0);
    // Raw material emission. Use eval_emission() for directional contribution.
    vec3 emission = vec3(0);
    EmissionSidedness emission_sidedness = EmissionSidedness::SingleSided;
    Float pdf_area = 0;
    Float area = 0;
};

struct DirectLightEstimate {
    vec3 wi = vec3(0);
    vec3 contribution = vec3(0);
    Float distance2 = 0;
    Float cos_surface = 0;
    Float cos_light = 0;
    Float pdf_light_area = 0;
    Float pdf_light_solid_angle = 0;
    Float pdf_bsdf_solid_angle = 0;
    SampleRecord light_sample;
    bool valid = false;
};

LuxHDInline Float clamp_cdf_sample(Float u) {
    return fminf(u, Float(0.99999994));
}

LuxHDInline int sample_cdf_index(const Float* cdf, int count,
                                 Float total_weight, Float u) {
    if (cdf == nullptr || count <= 0 || total_weight <= 0) return -1;
    Float target = clamp_cdf_sample(u) * total_weight;
    int index = count - 1;
    for (int i = 0; i < count; ++i) {
        if (target <= cdf[i]) {
            index = i;
            break;
        }
    }
    return index;
}

LuxHDInline Float cdf_interval_weight(const Float* cdf, int count, int index) {
    if (cdf == nullptr || index < 0 || index >= count) return 0;
    Float prev = index == 0 ? 0 : cdf[index - 1];
    return cdf[index] - prev;
}

LuxHDInline Float cdf_interval_pmf(const Float* cdf, int count,
                                   Float total_weight, int index) {
    if (total_weight <= 0) return 0;
    return cdf_interval_weight(cdf, count, index) / total_weight;
}

LuxHDInline int sample_alias_table_index(const AliasTableEntry* table,
                                         int count,
                                         Float u) {
    if (table == nullptr || count <= 0) return -1;
    Float scaled = clamp_cdf_sample(u) * Float(count);
    int column = static_cast<int>(scaled);
    if (column >= count) column = count - 1;
    Float threshold = scaled - Float(column);
    const AliasTableEntry& entry = table[column];
    int index = threshold < entry.probability ? column : entry.alias;
    return index >= 0 && index < count ? index : -1;
}

LuxHDInline Float alias_table_pmf(const AliasTableEntry* table,
                                  int count,
                                  int index) {
    if (table == nullptr || index < 0 || index >= count) return 0;
    return table[index].pmf;
}

LuxHDInline Float squared_distance_to_bounds(const AABB& bounds, const vec3& point) {
    Float distance2 = 0;
    for (int axis = 0; axis < 3; ++axis) {
        Float delta = 0;
        if (point[axis] < bounds.p_min[axis]) {
            delta = bounds.p_min[axis] - point[axis];
        } else if (point[axis] > bounds.p_max[axis]) {
            delta = point[axis] - bounds.p_max[axis];
        }
        distance2 += delta * delta;
    }
    return distance2;
}

LuxHDInline Float light_bvh_node_importance(const LightBvhNode* nodes,
                                            int node_count,
                                            int node_index,
                                            const LightSampleContext& ctx) {
    if (nodes == nullptr || node_index < 0 || node_index >= node_count) return 0;
    const LightBvhNode& node = nodes[node_index];
    if (node.power <= 0 || node.bounds.is_empty()) return 0;
    Float distance2 = squared_distance_to_bounds(node.bounds, ctx.position);
    distance2 = fmaxf(distance2, kRayEpsilon * kRayEpsilon);
    return node.power / distance2;
}

LuxHDInline Float light_bvh_child_probability(const LightBvhNode* nodes,
                                              int node_count,
                                              int parent_index,
                                              int child_index,
                                              const LightSampleContext& ctx) {
    if (nodes == nullptr
        || parent_index < 0 || parent_index >= node_count
        || child_index < 0 || child_index >= node_count) {
        return 0;
    }
    const LightBvhNode& parent = nodes[parent_index];
    if (child_index != parent.left && child_index != parent.right) return 0;

    Float left_importance = light_bvh_node_importance(
        nodes, node_count, parent.left, ctx);
    Float right_importance = light_bvh_node_importance(
        nodes, node_count, parent.right, ctx);
    Float total_importance = left_importance + right_importance;
    if (total_importance <= 0) return 0;

    Float left_probability = left_importance / total_importance;
    return child_index == parent.left ? left_probability
                                      : Float(1) - left_probability;
}

LuxHDInline int sample_light_bvh(const LightBvhNode* nodes,
                                 int node_count,
                                 int root_index,
                                 const LightSampleContext& ctx,
                                 Float select_u,
                                 Float& pmf) {
    pmf = 0;
    if (nodes == nullptr || root_index < 0 || root_index >= node_count) return -1;
    int node_index = root_index;
    Float u = clamp_cdf_sample(select_u);
    pmf = 1;

    for (int depth = 0; depth < node_count; ++depth) {
        const LightBvhNode& node = nodes[node_index];
        if (node.light_index >= 0) return node.light_index;

        Float left_probability = light_bvh_child_probability(
            nodes, node_count, node_index, node.left, ctx);
        if (left_probability <= 0) {
            node_index = node.right;
            if (node_index < 0 || node_index >= node_count) {
                pmf = 0;
                return -1;
            }
            continue;
        }
        if (left_probability >= 1) {
            node_index = node.left;
            if (node_index < 0 || node_index >= node_count) {
                pmf = 0;
                return -1;
            }
            continue;
        }

        if (u < left_probability) {
            pmf *= left_probability;
            u /= left_probability;
            node_index = node.left;
        } else {
            Float right_probability = Float(1) - left_probability;
            pmf *= right_probability;
            u = (u - left_probability) / right_probability;
            node_index = node.right;
        }
        if (node_index < 0 || node_index >= node_count) {
            pmf = 0;
            return -1;
        }
    }

    pmf = 0;
    return -1;
}

LuxHDInline Float light_bvh_pmf(const LightBvhNode* nodes,
                                int node_count,
                                const int* leaf_nodes,
                                int leaf_count,
                                const LightSampleContext& ctx,
                                int light_index) {
    if (nodes == nullptr || leaf_nodes == nullptr
        || light_index < 0 || light_index >= leaf_count) {
        return 0;
    }

    int node_index = leaf_nodes[light_index];
    if (node_index < 0 || node_index >= node_count) return 0;

    Float pmf = 1;
    for (int depth = 0; depth < node_count; ++depth) {
        const LightBvhNode& node = nodes[node_index];
        if (node.parent < 0) return pmf;
        Float probability = light_bvh_child_probability(
            nodes, node_count, node.parent, node_index, ctx);
        if (probability <= 0) return 0;
        pmf *= probability;
        node_index = node.parent;
        if (node_index < 0 || node_index >= node_count) return 0;
    }
    return 0;
}

LuxHDInline Float discrete_sampling_table_pmf(const Float* cdf,
                                              const AliasTableEntry* alias_table,
                                              int count,
                                              Float total_weight,
                                              int index) {
    if (alias_table) return alias_table_pmf(alias_table, count, index);
    return cdf_interval_pmf(cdf, count, total_weight, index);
}

LuxHDInline int find_light_index_slot(const int* light_indices,
                                      int count,
                                      int light_index) {
    if (light_indices == nullptr || light_index < 0) return -1;
    for (int i = 0; i < count; ++i) {
        if (light_indices[i] == light_index) return i;
    }
    return -1;
}

LuxHDInline int sample_indexed_discrete_light(
        const int* light_indices,
        const Float* cdf,
        const AliasTableEntry* alias_table,
        int count,
        Float total_weight,
        Float select_u,
        Float& pmf) {
    pmf = 0;
    if (light_indices == nullptr || count <= 0 || total_weight <= 0) return -1;
    int slot = alias_table
        ? sample_alias_table_index(alias_table, count, select_u)
        : sample_cdf_index(cdf, count, total_weight, select_u);
    if (slot < 0 || slot >= count) return -1;
    pmf = discrete_sampling_table_pmf(
        cdf, alias_table, count, total_weight, slot);
    return pmf > 0 ? light_indices[slot] : -1;
}

LuxHDInline int sample_light_bvh_mixture(
        const LightBvhNode* nodes,
        int node_count,
        int root_index,
        const int* non_bvh_light_indices,
        const Float* non_bvh_light_cdf,
        const AliasTableEntry* non_bvh_light_alias_table,
        int non_bvh_light_count,
        Float non_bvh_light_power,
        const LightSampleContext& ctx,
        Float select_u,
        Float& pmf) {
    pmf = 0;
    if (nodes == nullptr || root_index < 0 || root_index >= node_count) return -1;
    Float bvh_power = nodes[root_index].power;
    if (bvh_power <= 0) return -1;

    Float non_bvh_power =
        non_bvh_light_count > 0 ? fmaxf(Float(0), non_bvh_light_power) : Float(0);
    Float total_power = bvh_power + non_bvh_power;
    if (total_power <= 0) return -1;

    Float bvh_group_p = bvh_power / total_power;
    if (non_bvh_power <= 0 || bvh_group_p >= 1) {
        return sample_light_bvh(
            nodes, node_count, root_index, ctx, select_u, pmf);
    }
    if (bvh_group_p <= 0) {
        return sample_indexed_discrete_light(
            non_bvh_light_indices, non_bvh_light_cdf,
            non_bvh_light_alias_table, non_bvh_light_count,
            non_bvh_power, select_u, pmf);
    }

    Float u = clamp_cdf_sample(select_u);
    if (u < bvh_group_p) {
        Float local_u = u / bvh_group_p;
        Float bvh_pmf = 0;
        int light_index = sample_light_bvh(
            nodes, node_count, root_index, ctx, local_u, bvh_pmf);
        pmf = bvh_group_p * bvh_pmf;
        return pmf > 0 ? light_index : -1;
    }

    Float non_bvh_group_p = Float(1) - bvh_group_p;
    Float local_u = (u - bvh_group_p) / non_bvh_group_p;
    Float non_bvh_pmf = 0;
    int light_index = sample_indexed_discrete_light(
        non_bvh_light_indices, non_bvh_light_cdf,
        non_bvh_light_alias_table, non_bvh_light_count,
        non_bvh_power, local_u, non_bvh_pmf);
    pmf = non_bvh_group_p * non_bvh_pmf;
    return pmf > 0 ? light_index : -1;
}

LuxHDInline Float light_bvh_mixture_pmf(
        const LightBvhNode* nodes,
        int node_count,
        int root_index,
        const int* leaf_nodes,
        int leaf_count,
        const int* non_bvh_light_indices,
        const Float* non_bvh_light_cdf,
        const AliasTableEntry* non_bvh_light_alias_table,
        int non_bvh_light_count,
        Float non_bvh_light_power,
        const LightSampleContext& ctx,
        int light_index) {
    if (nodes == nullptr || root_index < 0 || root_index >= node_count
        || light_index < 0) {
        return 0;
    }
    Float bvh_power = nodes[root_index].power;
    if (bvh_power <= 0) return 0;
    Float non_bvh_power =
        non_bvh_light_count > 0 ? fmaxf(Float(0), non_bvh_light_power) : Float(0);
    Float total_power = bvh_power + non_bvh_power;
    if (total_power <= 0) return 0;

    Float bvh_group_p = bvh_power / total_power;
    if (leaf_nodes && light_index < leaf_count
        && leaf_nodes[light_index] >= 0) {
        return bvh_group_p
            * light_bvh_pmf(
                nodes, node_count, leaf_nodes, leaf_count, ctx, light_index);
    }

    int non_bvh_slot = find_light_index_slot(
        non_bvh_light_indices, non_bvh_light_count, light_index);
    if (non_bvh_slot < 0 || non_bvh_power <= 0) return 0;

    Float non_bvh_group_p = Float(1) - bvh_group_p;
    return non_bvh_group_p
        * discrete_sampling_table_pmf(
            non_bvh_light_cdf, non_bvh_light_alias_table,
            non_bvh_light_count, non_bvh_power, non_bvh_slot);
}

LuxHDInline vec3 sample_triangle_area(const Triangle& triangle, const vec2& u) {
    Float sqrt_u0 = sqrtf(u.x);
    Float b0 = Float(1) - sqrt_u0;
    Float b1 = sqrt_u0 * (Float(1) - u.y);
    Float b2 = sqrt_u0 * u.y;
    return b0 * triangle.v0 + b1 * triangle.v1 + b2 * triangle.v2;
}

LuxHDInline vec2 sample_triangle_barycentric_uv(const vec2& u) {
    Float sqrt_u0 = sqrtf(u.x);
    Float b1 = sqrt_u0 * (Float(1) - u.y);
    Float b2 = sqrt_u0 * u.y;
    return vec2(b1, b2);
}

LuxHDInline Float pdf_solid_angle(const LightSample& light, const vec3& reference_position) {
    vec3 to_light = light.position - reference_position;
    Float distance2 = length2(to_light);
    if (distance2 <= kRayEpsilon * kRayEpsilon || light.pdf_area <= 0) return 0;

    vec3 wi = to_light / sqrtf(distance2);
    Float cos_light = emission_projected_cosine(
        light.normal, -wi, light.emission_sidedness);
    if (cos_light <= 0) return 0;
    return light.pdf_area * distance2 / cos_light;
}

LuxHDInline DirectLightEstimate estimate_unoccluded_direct_light(
        const Material& material,
        const SurfaceInteraction& interaction,
        const LightSample& light) {
    DirectLightEstimate estimate;
    vec3 to_light = light.position - interaction.position;
    estimate.distance2 = length2(to_light);
    if (estimate.distance2 <= kRayEpsilon * kRayEpsilon || light.pdf_area <= 0) {
        return estimate;
    }

    estimate.wi = to_light / sqrtf(estimate.distance2);
    vec3 f = eval_bsdf(material, interaction, estimate.wi);
    vec3 Le = eval_emission(
        light.emission, light.emission_sidedness, light.normal, -estimate.wi);
    estimate.cos_surface = abs_cosine(interaction.ns, estimate.wi);
    estimate.cos_light = emission_projected_cosine(
        light.normal, -estimate.wi, light.emission_sidedness);
    estimate.pdf_light_area = light.pdf_area;
    estimate.pdf_light_solid_angle = pdf_solid_angle(light, interaction.position);
    estimate.pdf_bsdf_solid_angle = pdf_bsdf(material, interaction, estimate.wi);
    estimate.light_sample = SampleRecord{
        SampleSource::Light,
        solid_angle_pdf(estimate.pdf_light_solid_angle),
        0
    };
    estimate.valid = max_component(f) > 0
        && max_component(Le) > 0
        && estimate.cos_surface > 0
        && estimate.cos_light > 0
        && estimate.pdf_light_solid_angle > 0;
    if (!estimate.valid) return estimate;

    estimate.contribution = f * Le
        * (estimate.cos_surface / estimate.pdf_light_solid_angle);
    return estimate;
}

LuxHDInline DirectLightEstimate estimate_unoccluded_direct_light_li(
        const Material& material,
        const SurfaceInteraction& interaction,
        const LightLiSample& light) {
    DirectLightEstimate estimate;
    if (!light.valid || light.pdf <= 0) return estimate;
    estimate.wi = light.wi;
    estimate.distance2 = light.distance < INFINITY
        ? light.distance * light.distance
        : INFINITY;
    vec3 f = eval_bsdf(material, interaction, estimate.wi);
    estimate.cos_surface = abs_cosine(interaction.ns, estimate.wi);
    estimate.cos_light = 1;
    estimate.pdf_light_area = 0;
    estimate.pdf_light_solid_angle = light.pdf;
    estimate.pdf_bsdf_solid_angle = pdf_bsdf(material, interaction, estimate.wi);
    estimate.light_sample = SampleRecord{
        SampleSource::Light,
        light.pdf_measure == PdfMeasure::Discrete ? discrete_pdf(light.pdf)
                                                  : solid_angle_pdf(light.pdf),
        0
    };
    estimate.valid = max_component(f) > 0
        && max_component(light.radiance) > 0
        && estimate.cos_surface > 0
        && estimate.pdf_light_solid_angle > 0;
    if (!estimate.valid) return estimate;
    estimate.contribution = f * light.radiance
        * (estimate.cos_surface / estimate.pdf_light_solid_angle);
    return estimate;
}

template <MaterialType Type>
LuxHDInline DirectLightEstimate estimate_unoccluded_direct_light_li_typed(
        const Material& material,
        const SurfaceInteraction& interaction,
        const LightLiSample& light) {
    DirectLightEstimate estimate;
    if (!light.valid || light.pdf <= 0) return estimate;
    estimate.wi = light.wi;
    estimate.distance2 = light.distance < INFINITY
        ? light.distance * light.distance
        : INFINITY;
    vec3 f = eval_bsdf_typed<Type>(material, interaction, estimate.wi);
    estimate.cos_surface = abs_cosine(interaction.ns, estimate.wi);
    estimate.cos_light = 1;
    estimate.pdf_light_area = 0;
    estimate.pdf_light_solid_angle = light.pdf;
    estimate.pdf_bsdf_solid_angle = pdf_bsdf_typed<Type>(material, interaction, estimate.wi);
    estimate.light_sample = SampleRecord{
        SampleSource::Light,
        light.pdf_measure == PdfMeasure::Discrete ? discrete_pdf(light.pdf)
                                                  : solid_angle_pdf(light.pdf),
        0
    };
    estimate.valid = max_component(f) > 0
        && max_component(light.radiance) > 0
        && estimate.cos_surface > 0
        && estimate.pdf_light_solid_angle > 0;
    if (!estimate.valid) return estimate;
    estimate.contribution = f * light.radiance
        * (estimate.cos_surface / estimate.pdf_light_solid_angle);
    return estimate;
}

template <MaterialType Type>
LuxHDInline DirectLightEstimate estimate_unoccluded_direct_light_typed(
        const Material& material,
        const SurfaceInteraction& interaction,
        const LightSample& light) {
    DirectLightEstimate estimate;
    vec3 to_light = light.position - interaction.position;
    estimate.distance2 = length2(to_light);
    if (estimate.distance2 <= kRayEpsilon * kRayEpsilon || light.pdf_area <= 0) {
        return estimate;
    }

    estimate.wi = to_light / sqrtf(estimate.distance2);
    vec3 f = eval_bsdf_typed<Type>(material, interaction, estimate.wi);
    vec3 Le = eval_emission(
        light.emission, light.emission_sidedness, light.normal, -estimate.wi);
    estimate.cos_surface = abs_cosine(interaction.ns, estimate.wi);
    estimate.cos_light = emission_projected_cosine(
        light.normal, -estimate.wi, light.emission_sidedness);
    estimate.pdf_light_area = light.pdf_area;
    estimate.pdf_light_solid_angle = pdf_solid_angle(light, interaction.position);
    estimate.pdf_bsdf_solid_angle = pdf_bsdf_typed<Type>(material, interaction, estimate.wi);
    estimate.light_sample = SampleRecord{
        SampleSource::Light,
        solid_angle_pdf(estimate.pdf_light_solid_angle),
        0
    };
    estimate.valid = max_component(f) > 0
        && max_component(Le) > 0
        && estimate.cos_surface > 0
        && estimate.cos_light > 0
        && estimate.pdf_light_solid_angle > 0;
    if (!estimate.valid) return estimate;

    estimate.contribution = f * Le
        * (estimate.cos_surface / estimate.pdf_light_solid_angle);
    return estimate;
}

#endif // LUX_LIGHT_SAMPLER_H
