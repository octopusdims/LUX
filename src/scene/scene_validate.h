#pragma once

#ifndef LUX_SCENE_VALIDATE_H
#define LUX_SCENE_VALIDATE_H

#include "core/validation.cuh"
#include "scene/scene_data.h"

LuxInline bool vec3_channels_in_range(const vec3& value, Float lo, Float hi) {
    return value.x >= lo && value.x <= hi
        && value.y >= lo && value.y <= hi
        && value.z >= lo && value.z <= hi;
}

LuxInline void validate_scene_shape(const Scene& scene) {
    for (const Material& material : scene.materials) {
        if (!finite_vec3(material.albedo) || !finite_vec3(material.emission)
            || !finite_vec3(material.emission_scale)
            || !std::isfinite(material.roughness) || !std::isfinite(material.metallic)
            || !std::isfinite(material.eta)) {
            throw std::runtime_error("Scene validation failed: material contains non-finite value");
        }
        if (!vec3_channels_in_range(material.albedo, Float(0), Float(1))) {
            throw std::runtime_error("Scene validation failed: material albedo must be in [0, 1]");
        }
        if (!vec3_channels_in_range(material.emission, Float(0),
                                    std::numeric_limits<Float>::infinity())) {
            throw std::runtime_error("Scene validation failed: material emission must be non-negative");
        }
        if (material.eta <= 0) {
            throw std::runtime_error("Scene validation failed: material eta must be > 0");
        }
        if (material.emission_source == EmissionSourceKind::Image
            && (material.emission_image_id < 0
                || material.emission_image_id >= static_cast<int>(scene.image_light_assets.size()))) {
            throw std::runtime_error("Scene validation failed: material emission image id out of range");
        }
        switch (material.type) {
            case MaterialType::Lambert:
            case MaterialType::ThinLambert:
            case MaterialType::GGX:
            case MaterialType::Dielectric:
            case MaterialType::Transparent:
                break;
            case MaterialType::Interface:
                if (fabsf(material.albedo.x - Float(1)) > Float(1e-6)
                    || fabsf(material.albedo.y - Float(1)) > Float(1e-6)
                    || fabsf(material.albedo.z - Float(1)) > Float(1e-6)) {
                    throw std::runtime_error("Scene validation failed: Interface material must use unit transmittance");
                }
                break;
        }
    }

    for (const SceneImageLightAsset& asset : scene.image_light_assets) {
        if (asset.width <= 0 || asset.height <= 0
            || asset.pixels.size() != static_cast<size_t>(asset.width * asset.height)
            || asset.cdf.size() != asset.pixels.size()
            || !std::isfinite(static_cast<double>(asset.total_weight))
            || asset.total_weight < 0) {
            throw std::runtime_error("Scene validation failed: image light asset contains invalid value");
        }
        for (const vec3& pixel : asset.pixels) {
            if (!finite_vec3(pixel)) {
                throw std::runtime_error("Scene validation failed: image light asset pixel contains non-finite value");
            }
        }
        Float previous_cdf = Float(0);
        for (Float cdf_value : asset.cdf) {
            if (!std::isfinite(static_cast<double>(cdf_value))
                || cdf_value < previous_cdf) {
                throw std::runtime_error("Scene validation failed: image light asset cdf is invalid");
            }
            previous_cdf = cdf_value;
        }
        Float cdf_tolerance =
            Float(1e-5) * fmaxf(Float(1e-6), fabsf(asset.total_weight));
        if (fabsf(previous_cdf - asset.total_weight) > cdf_tolerance) {
            throw std::runtime_error("Scene validation failed: image light asset cdf total is inconsistent");
        }
    }

    for (const SceneMesh& mesh : scene.mesh_assets) {
        size_t vertex_count = mesh.vertices.size();
        size_t triangle_count = mesh.indices.size();
        if (mesh.normals.size() != mesh.vertices.size()) {
            throw std::runtime_error("Scene validation failed: mesh normals size must match vertices size");
        }
        if (mesh.uvs.size() != mesh.vertices.size()) {
            throw std::runtime_error("Scene validation failed: mesh uvs size must match vertices size");
        }
        if (mesh.material_ids.size() != mesh.indices.size()) {
            throw std::runtime_error("Scene validation failed: mesh material_ids size must match triangle count");
        }
        if (mesh.boundary_edges.size() != mesh.indices.size()) {
            throw std::runtime_error("Scene validation failed: mesh boundary_edges size must match triangle count");
        }
        for (size_t i = 0; i < vertex_count; ++i) {
            if (!finite_vec3(mesh.vertices[i])) {
                throw std::runtime_error("Scene validation failed: mesh vertex contains non-finite value");
            }
            if (!finite_vec3(mesh.normals[i])) {
                throw std::runtime_error("Scene validation failed: mesh normal contains non-finite value");
            }
            if (!finite_vec2(mesh.uvs[i])) {
                throw std::runtime_error("Scene validation failed: mesh uv contains non-finite value");
            }
        }
        for (const TriangleIndices& idx : mesh.indices) {
            if (idx.v0 >= vertex_count || idx.v1 >= vertex_count || idx.v2 >= vertex_count) {
                throw std::runtime_error("Scene validation failed: mesh triangle index out of range");
            }
        }
        for (int material_id : mesh.material_ids) {
            if (material_id < 0 || material_id >= static_cast<int>(scene.materials.size())) {
                throw std::runtime_error("Scene validation failed: mesh material id out of range");
            }
        }
        for (unsigned char boundary_edges : mesh.boundary_edges) {
            if ((boundary_edges & ~7u) != 0) {
                throw std::runtime_error("Scene validation failed: mesh boundary edge mask out of range");
            }
        }
        (void)triangle_count;
    }

    for (const SceneInstance& instance : scene.instances) {
        if (instance.mesh_id < 0
            || instance.mesh_id >= static_cast<int>(scene.mesh_assets.size())) {
            throw std::runtime_error("Scene validation failed: instance mesh id out of range");
        }
        if (instance.material_override < -1
            || instance.material_override >= static_cast<int>(scene.materials.size())) {
            throw std::runtime_error("Scene validation failed: instance material override out of range");
        }
        if (instance_transform_valid(instance.object_to_world)) {
            continue;
        }
        if (!instance_transform_matrices_finite(instance.object_to_world)) {
            throw std::runtime_error("Scene validation failed: instance transform contains non-finite value");
        }
        if (!instance_transform_matrices_affine_invertible(instance.object_to_world)) {
            throw std::runtime_error("Scene validation failed: instance transform must be affine-invertible");
        }
        if (!transform_inverse_pair_consistent(instance.object_to_world)) {
            throw std::runtime_error("Scene validation failed: instance transform inverse is inconsistent");
        }
    }
    if (scene.default_mesh_id < -1
        || scene.default_mesh_id >= static_cast<int>(scene.mesh_assets.size())) {
        throw std::runtime_error("Scene validation failed: default mesh id out of range");
    }
    if (scene.default_instance_id < -1
        || scene.default_instance_id >= static_cast<int>(scene.instances.size())) {
        throw std::runtime_error("Scene validation failed: default instance id out of range");
    }

    for (const Light& light : scene.lights) {
        switch (light.kind) {
            case LightKind::Point:
                if (!finite_vec3(light.point.position) || !finite_vec3(light.point.intensity)
                    || !vec3_channels_in_range(light.point.intensity, Float(0),
                                               std::numeric_limits<Float>::infinity())) {
                    throw std::runtime_error("Scene validation failed: point light contains invalid value");
                }
                break;
            case LightKind::Distant:
                if (!finite_vec3(light.distant.direction_to_light)
                    || length2(light.distant.direction_to_light) == 0
                    || !finite_vec3(light.distant.radiance)
                    || !vec3_channels_in_range(light.distant.radiance, Float(0),
                                               std::numeric_limits<Float>::infinity())) {
                    throw std::runtime_error("Scene validation failed: distant light contains invalid value");
                }
                break;
            case LightKind::UniformInfinite:
                if (!finite_vec3(light.uniform_infinite.radiance)
                    || !vec3_channels_in_range(light.uniform_infinite.radiance, Float(0),
                                               std::numeric_limits<Float>::infinity())) {
                    throw std::runtime_error("Scene validation failed: uniform infinite light contains invalid value");
                }
                break;
            case LightKind::ImageInfinite:
                if (light.image_infinite.image_id < 0
                    || light.image_infinite.image_id >= static_cast<int>(scene.image_light_assets.size())
                    || light.image_infinite.width <= 0 || light.image_infinite.height <= 0
                    || !std::isfinite(static_cast<double>(light.image_infinite.total_weight))
                    || light.image_infinite.total_weight < 0
                    || !finite_vec3(light.image_infinite.scale)) {
                    throw std::runtime_error("Scene validation failed: image infinite light contains invalid value");
                }
                break;
            case LightKind::DiffuseArea:
                if (light.area.primitive_id < -1 || light.area.area < 0
                    || !finite_vec3(light.area.emission)
                    || !finite_vec3(light.area.emission_scale)
                    || !vec3_channels_in_range(light.area.emission, Float(0),
                                               std::numeric_limits<Float>::infinity())) {
                    throw std::runtime_error("Scene validation failed: diffuse area light contains invalid value");
                }
                if (primitive_ref_valid(light.area.primitive_ref)
                    && !scene_primitive_ref_exists(scene, light.area.primitive_ref)) {
                    throw std::runtime_error("Scene validation failed: diffuse area light primitive ref out of range");
                }
                if (!primitive_ref_valid(light.area.primitive_ref)
                    && light.area.primitive_id >= scene_triangle_count(scene)) {
                    throw std::runtime_error("Scene validation failed: diffuse area light primitive id out of range");
                }
                if (light.area.emission_source == EmissionSourceKind::Image
                    && (light.area.emission_image_id < 0
                        || light.area.emission_image_id
                            >= static_cast<int>(scene.image_light_assets.size()))) {
                    throw std::runtime_error("Scene validation failed: diffuse area light image id out of range");
                }
                break;
        }
    }
}


#endif // LUX_SCENE_VALIDATE_H
