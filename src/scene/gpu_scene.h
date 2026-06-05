#pragma once

#ifndef LUX_SCENE_GPU_H
#define LUX_SCENE_GPU_H

#include <vector>

#include <thrust/device_vector.h>

#include "bvh/bvh_device.cuh"
#include "light/light_distribution.h"
#include "material/material.h"
#include "scene.h"
#include "scene/gpu_scene_view.h"
#include "scene/scene_prepare.h"

// Host-side owner of device scene data. Creates GpuScene POD on demand.
struct GpuSceneData {
    thrust::device_vector<Material> d_materials;
    thrust::device_vector<PrimitiveRef> d_light_primitive_refs;
    thrust::device_vector<Float> d_light_primitive_areas;
    thrust::device_vector<Float> d_light_area_cdf;
    thrust::device_vector<AliasTableEntry> d_light_area_alias_table;
    thrust::device_vector<Light> d_lights;
    thrust::device_vector<Float> d_light_power_cdf;
    thrust::device_vector<AliasTableEntry> d_light_power_alias_table;
    thrust::device_vector<LightBvhNode> d_light_bvh_nodes;
    thrust::device_vector<int> d_light_bvh_leaf_nodes;
    thrust::device_vector<int> d_non_bvh_light_indices;
    thrust::device_vector<Float> d_non_bvh_light_cdf;
    thrust::device_vector<AliasTableEntry> d_non_bvh_light_alias_table;
    thrust::device_vector<int> d_primitive_light_indices;
    thrust::device_vector<vec3> d_image_light_pixels;
    thrust::device_vector<Float> d_image_light_cdfs;
    thrust::device_vector<GpuImageAsset> d_image_assets;
    thrust::device_vector<vec3> d_mesh_vertices;
    thrust::device_vector<vec3> d_mesh_normals;
    thrust::device_vector<vec2> d_mesh_uvs;
    thrust::device_vector<TriangleIndices> d_mesh_indices;
    thrust::device_vector<int> d_mesh_material_ids;
    thrust::device_vector<unsigned char> d_mesh_boundary_edges;
    thrust::device_vector<GpuSceneMesh> d_meshes;
    thrust::device_vector<GpuSceneInstance> d_instances;
    thrust::device_vector<PrimitiveRef> d_scene_primitive_refs;
    thrust::device_vector<GpuBvhView> d_blas_views;
    Float total_light_area = 0;
    Float total_light_power = 0;
    Float non_bvh_light_power = 0;
    int light_bvh_root = -1;
    bool two_level_enabled = false;
    Camera scene_camera;

    GpuSceneData() = default;

    GpuSceneData(const Scene& scene, const PreparedLightSampling& light_sampling) {
        upload(scene, light_sampling);
    }

    void upload(const Scene& scene, const PreparedLightSampling& light_sampling) {
        require_prepared_light_sampling_current(
            scene, light_sampling,
            "GpuSceneData requires PreparedLightSampling built from the current Scene");
        const AreaLightSamplingTable& area_lights = light_sampling.area_lights;
        const LightSelectionTable& light_selection = light_sampling.light_selection;
        const DiscreteSamplingTable& area_distribution = area_lights.area_distribution;
        const DiscreteSamplingTable& power_distribution = light_selection.power_distribution;
        const LightBvhTable& light_bvh = light_sampling.light_bvh;
        const SurfaceLightLookupTable& surface_lookup = light_sampling.surface_lookup;
        scene_camera = scene.camera;
        d_materials = scene.materials;
        d_light_primitive_refs = area_lights.primitive_refs;
        d_light_primitive_areas = area_lights.primitive_areas;
        d_light_area_cdf = area_distribution.cdf;
        d_light_area_alias_table = area_distribution.alias_table;
        two_level_enabled = false;
        d_blas_views.clear();

        std::vector<vec3> mesh_vertices;
        std::vector<vec3> mesh_normals;
        std::vector<vec2> mesh_uvs;
        std::vector<TriangleIndices> mesh_indices;
        std::vector<int> mesh_material_ids;
        std::vector<unsigned char> mesh_boundary_edges;
        std::vector<int> mesh_vertex_offsets(scene.mesh_assets.size(), 0);
        std::vector<int> mesh_triangle_offsets(scene.mesh_assets.size(), 0);
        std::vector<PrimitiveRef> scene_primitive_refs;

        for (int mesh_id = 0; mesh_id < static_cast<int>(scene.mesh_assets.size()); ++mesh_id) {
            const SceneMesh& mesh = scene.mesh_assets[mesh_id];
            mesh_vertex_offsets[mesh_id] = static_cast<int>(mesh_vertices.size());
            mesh_triangle_offsets[mesh_id] = static_cast<int>(mesh_indices.size());
            mesh_vertices.insert(mesh_vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
            mesh_normals.insert(mesh_normals.end(), mesh.normals.begin(), mesh.normals.end());
            mesh_uvs.insert(mesh_uvs.end(), mesh.uvs.begin(), mesh.uvs.end());
            mesh_indices.insert(mesh_indices.end(), mesh.indices.begin(), mesh.indices.end());
            mesh_material_ids.insert(
                mesh_material_ids.end(), mesh.material_ids.begin(), mesh.material_ids.end());
            mesh_boundary_edges.insert(
                mesh_boundary_edges.end(), mesh.boundary_edges.begin(), mesh.boundary_edges.end());
        }

        d_mesh_vertices = mesh_vertices;
        d_mesh_normals = mesh_normals;
        d_mesh_uvs = mesh_uvs;
        d_mesh_indices = mesh_indices;
        d_mesh_material_ids = mesh_material_ids;
        d_mesh_boundary_edges = mesh_boundary_edges;

        const vec3* mesh_vertices_device = d_mesh_vertices.empty()
            ? nullptr
            : thrust::raw_pointer_cast(d_mesh_vertices.data());
        const vec3* mesh_normals_device = d_mesh_normals.empty()
            ? nullptr
            : thrust::raw_pointer_cast(d_mesh_normals.data());
        const vec2* mesh_uvs_device = d_mesh_uvs.empty()
            ? nullptr
            : thrust::raw_pointer_cast(d_mesh_uvs.data());
        const TriangleIndices* mesh_indices_device = d_mesh_indices.empty()
            ? nullptr
            : thrust::raw_pointer_cast(d_mesh_indices.data());
        const int* mesh_material_ids_device = d_mesh_material_ids.empty()
            ? nullptr
            : thrust::raw_pointer_cast(d_mesh_material_ids.data());
        const unsigned char* mesh_boundary_edges_device = d_mesh_boundary_edges.empty()
            ? nullptr
            : thrust::raw_pointer_cast(d_mesh_boundary_edges.data());

        std::vector<GpuSceneMesh> gpu_meshes;
        gpu_meshes.reserve(scene.mesh_assets.size());
        for (int mesh_id = 0; mesh_id < static_cast<int>(scene.mesh_assets.size()); ++mesh_id) {
            const SceneMesh& mesh = scene.mesh_assets[mesh_id];
            int vertex_offset = mesh_vertex_offsets[mesh_id];
            int triangle_offset = mesh_triangle_offsets[mesh_id];
            gpu_meshes.push_back(GpuSceneMesh{
                TriangleMeshView{
                    mesh.vertices.empty() ? nullptr : mesh_vertices_device + vertex_offset,
                    mesh.normals.empty() ? nullptr : mesh_normals_device + vertex_offset,
                    mesh.uvs.empty() ? nullptr : mesh_uvs_device + vertex_offset,
                    mesh.indices.empty() ? nullptr : mesh_indices_device + triangle_offset,
                    static_cast<int>(mesh.indices.size())
                },
                mesh.material_ids.empty() ? nullptr : mesh_material_ids_device + triangle_offset,
                mesh.boundary_edges.empty() ? nullptr : mesh_boundary_edges_device + triangle_offset,
                static_cast<int>(mesh.indices.size())
            });
        }
        d_meshes = gpu_meshes;

        std::vector<GpuSceneInstance> gpu_instances;
        gpu_instances.reserve(scene.instances.size());
        for (int instance_id = 0; instance_id < static_cast<int>(scene.instances.size());
             ++instance_id) {
            const SceneInstance& instance = scene.instances[instance_id];
            int first_primitive = -1;
            if (instance.mesh_id >= 0
                && instance.mesh_id < static_cast<int>(scene.mesh_assets.size())) {
                const SceneMesh& mesh = scene.mesh_assets[instance.mesh_id];
                first_primitive = static_cast<int>(scene_primitive_refs.size());
                for (int local_id = 0; local_id < static_cast<int>(mesh.indices.size());
                     ++local_id) {
                    scene_primitive_refs.push_back(PrimitiveRef{
                        instance_id, instance.mesh_id, local_id, PrimitiveKind::Triangle});
                }
            }
            gpu_instances.push_back(GpuSceneInstance{
                instance.mesh_id,
                instance.mesh_id,
                instance.material_override,
                first_primitive,
                instance.object_to_world
            });
        }
        d_scene_primitive_refs = scene_primitive_refs;
        d_instances = gpu_instances;

        std::vector<vec3> image_pixels;
        std::vector<Float> image_cdfs;
        std::vector<int> image_pixel_offsets(scene.image_light_assets.size(), -1);
        std::vector<int> image_cdf_offsets(scene.image_light_assets.size(), -1);
        for (int image_id = 0; image_id < static_cast<int>(scene.image_light_assets.size());
             ++image_id) {
            const SceneImageLightAsset& asset = scene.image_light_assets[image_id];
            image_pixel_offsets[image_id] = static_cast<int>(image_pixels.size());
            image_cdf_offsets[image_id] = static_cast<int>(image_cdfs.size());
            image_pixels.insert(image_pixels.end(), asset.pixels.begin(), asset.pixels.end());
            image_cdfs.insert(image_cdfs.end(), asset.cdf.begin(), asset.cdf.end());
        }
        d_image_light_pixels = image_pixels;
        d_image_light_cdfs = image_cdfs;
        const vec3* image_pixels_device = d_image_light_pixels.empty()
            ? nullptr
            : thrust::raw_pointer_cast(d_image_light_pixels.data());
        const Float* image_cdfs_device = d_image_light_cdfs.empty()
            ? nullptr
            : thrust::raw_pointer_cast(d_image_light_cdfs.data());
        std::vector<GpuImageAsset> gpu_image_assets;
        gpu_image_assets.reserve(scene.image_light_assets.size());
        for (int image_id = 0; image_id < static_cast<int>(scene.image_light_assets.size());
             ++image_id) {
            const SceneImageLightAsset& asset = scene.image_light_assets[image_id];
            gpu_image_assets.push_back(GpuImageAsset{
                image_pixel_offsets[image_id] < 0 ? nullptr
                                                  : image_pixels_device + image_pixel_offsets[image_id],
                asset.width,
                asset.height
            });
        }
        d_image_assets = gpu_image_assets;

        std::vector<Light> gpu_lights;
        std::vector<Float> gpu_light_power_cdf;
        Float gpu_light_power = 0;
        gpu_lights.reserve(light_selection.lights.size());
        gpu_light_power_cdf.reserve(power_distribution.cdf.size());
        for (int i = 0; i < static_cast<int>(light_selection.lights.size()); ++i) {
            Light light = light_selection.lights[i];
            if (light.kind == LightKind::ImageInfinite) {
                int image_id = light.image_infinite.image_id;
                if (image_id < 0
                    || image_id >= static_cast<int>(image_pixel_offsets.size())
                    || image_pixel_offsets[image_id] < 0
                    || image_cdf_offsets[image_id] < 0) {
                    continue;
                }
                light.image_infinite.pixels =
                    image_pixels_device + image_pixel_offsets[image_id];
                light.image_infinite.cdf =
                    image_cdfs_device + image_cdf_offsets[image_id];
            }
            Float previous = i == 0 ? 0 : power_distribution.cdf[i - 1];
            Float power = power_distribution.cdf[i] - previous;
            if (power <= 0) continue;
            gpu_light_power += power;
            gpu_lights.push_back(light);
            gpu_light_power_cdf.push_back(gpu_light_power);
        }
        d_lights = gpu_lights;
        d_light_power_cdf = gpu_light_power_cdf;
        d_light_power_alias_table =
            build_alias_table_from_cdf(gpu_light_power_cdf, gpu_light_power);
        if (gpu_lights.size() == light_selection.lights.size()) {
            d_light_bvh_nodes = light_bvh.nodes;
            d_light_bvh_leaf_nodes = light_bvh.leaf_nodes_by_light_index;
            d_non_bvh_light_indices = light_bvh.non_bvh_light_indices;
            d_non_bvh_light_cdf = light_bvh.non_bvh_light_distribution.cdf;
            d_non_bvh_light_alias_table =
                light_bvh.non_bvh_light_distribution.alias_table;
            d_primitive_light_indices = surface_lookup.primitive_light_indices;
            light_bvh_root = light_bvh.root;
            non_bvh_light_power =
                light_bvh.non_bvh_light_distribution.total_weight;
        } else {
            d_light_bvh_nodes.clear();
            d_light_bvh_leaf_nodes.clear();
            d_non_bvh_light_indices.clear();
            d_non_bvh_light_cdf.clear();
            d_non_bvh_light_alias_table.clear();
            d_primitive_light_indices.clear();
            light_bvh_root = -1;
            non_bvh_light_power = 0;
        }
        total_light_area = area_distribution.total_weight;
        total_light_power = gpu_light_power;
    }

    void upload_blas_views(const std::vector<GpuBvhView>& blas_views, bool enable_two_level) {
        d_blas_views = blas_views;
        two_level_enabled = enable_two_level
            && !d_blas_views.empty()
            && !d_meshes.empty()
            && !d_instances.empty();
    }

    GpuScene view() const {
        return GpuScene{
            scene_camera,
            d_materials.empty() ? nullptr : thrust::raw_pointer_cast(d_materials.data()),
            static_cast<int>(d_materials.size()),
            d_light_primitive_refs.empty() ? nullptr : thrust::raw_pointer_cast(d_light_primitive_refs.data()),
            d_light_primitive_areas.empty() ? nullptr : thrust::raw_pointer_cast(d_light_primitive_areas.data()),
            d_light_area_cdf.empty() ? nullptr : thrust::raw_pointer_cast(d_light_area_cdf.data()),
            d_light_area_alias_table.empty() ? nullptr : thrust::raw_pointer_cast(d_light_area_alias_table.data()),
            static_cast<int>(d_light_primitive_refs.size()),
            total_light_area,
            d_lights.empty() ? nullptr : thrust::raw_pointer_cast(d_lights.data()),
            d_light_power_cdf.empty() ? nullptr : thrust::raw_pointer_cast(d_light_power_cdf.data()),
            d_light_power_alias_table.empty() ? nullptr : thrust::raw_pointer_cast(d_light_power_alias_table.data()),
            static_cast<int>(d_lights.size()),
            total_light_power,
            d_light_bvh_nodes.empty() ? nullptr : thrust::raw_pointer_cast(d_light_bvh_nodes.data()),
            d_light_bvh_leaf_nodes.empty() ? nullptr : thrust::raw_pointer_cast(d_light_bvh_leaf_nodes.data()),
            static_cast<int>(d_light_bvh_nodes.size()),
            light_bvh_root,
            d_non_bvh_light_indices.empty() ? nullptr : thrust::raw_pointer_cast(d_non_bvh_light_indices.data()),
            d_non_bvh_light_cdf.empty() ? nullptr : thrust::raw_pointer_cast(d_non_bvh_light_cdf.data()),
            d_non_bvh_light_alias_table.empty() ? nullptr : thrust::raw_pointer_cast(d_non_bvh_light_alias_table.data()),
            static_cast<int>(d_non_bvh_light_indices.size()),
            non_bvh_light_power,
            d_primitive_light_indices.empty()
                ? nullptr
                : thrust::raw_pointer_cast(d_primitive_light_indices.data()),
            static_cast<int>(d_primitive_light_indices.size()),
            d_meshes.empty() ? nullptr : thrust::raw_pointer_cast(d_meshes.data()),
            static_cast<int>(d_meshes.size()),
            d_instances.empty() ? nullptr : thrust::raw_pointer_cast(d_instances.data()),
            static_cast<int>(d_instances.size()),
            d_scene_primitive_refs.empty()
                ? nullptr
                : thrust::raw_pointer_cast(d_scene_primitive_refs.data()),
            static_cast<int>(d_scene_primitive_refs.size()),
            d_blas_views.empty() ? nullptr : thrust::raw_pointer_cast(d_blas_views.data()),
            static_cast<int>(d_blas_views.size()),
            two_level_enabled,
            d_image_assets.empty() ? nullptr : thrust::raw_pointer_cast(d_image_assets.data()),
            static_cast<int>(d_image_assets.size())
        };
    }
};

#endif // LUX_SCENE_GPU_H
