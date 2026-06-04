#pragma once

#ifndef LUX_UTIL_OBJ_LOADER_H
#define LUX_UTIL_OBJ_LOADER_H

#include <string>
#include <unordered_map>
#include <vector>
#include "scene/scene.h"

enum class ObjMaterialSource {
    None,
    DefaultMaterial,
    UsemtlBinding,
};

enum class ObjBoundaryTopology {
    SourceVertexIndices,
    WeldedPositions,
};

enum class ObjNormalMode {
    SourceOrGeometry,
    GenerateMissingSmooth,
    FaceOnly,
};

enum class ObjPlacementScaleMode {
    None,
    MaxExtent,
    XExtent,
};

struct ObjPlacementOptions {
    ObjPlacementScaleMode scale_mode = ObjPlacementScaleMode::None;
    Float target_extent = Float(1);
    bool center_xz = false;
    bool ground_to_y = false;
    Float target_y = Float(0);
};

struct ObjMeshGroupRange {
    int mesh_id = -1;
    std::string shape_name;
    std::string material_name;
    int material_id = -1;
    int first_triangle = 0;
    int triangle_count = 0;
};

struct ObjLoadOptions {
    bool cap_boundary_holes = false;
    // Compatibility alias for normal_mode = GenerateMissingSmooth.
    bool generate_missing_normals = false;
    ObjNormalMode normal_mode = ObjNormalMode::SourceOrGeometry;
    int default_material_id = -1;
    bool create_instance = true;
    bool flip_texcoord_v = false;
    bool reverse_orientation = false;
    Transform object_to_world;
    ObjPlacementOptions placement;
    ObjBoundaryTopology boundary_topology = ObjBoundaryTopology::SourceVertexIndices;
    ObjBoundaryTopology cap_boundary_topology = ObjBoundaryTopology::WeldedPositions;
    Float position_weld_epsilon = Float(1e-6);
    std::unordered_map<std::string, int> material_bindings;
};

struct ObjLoadResult {
    SceneMeshAppendRange appended_range;
    int mesh_id = -1;
    int instance_id = -1;
    int first_material_id = -1;
    int material_count = 0;
    AABB source_bounds;
    AABB mesh_bounds;
    AABB world_bounds;
    int source_vertex_count = 0;
    int source_triangle_count = 0;
    int loaded_triangle_count = 0;
    int cap_triangle_count = 0;
    int mesh_vertex_count = 0;
    int mesh_triangle_count = 0;
    int source_boundary_edge_count = 0;
    int cap_boundary_edge_count = 0;
    int cap_non_simple_boundary_vertex_count = 0;
    int cap_incomplete_loop_count = 0;
    int cap_degenerate_loop_count = 0;
    int cap_degenerate_triangle_count = 0;
    int nonfinite_vertex_count = 0;
    int skipped_nonfinite_triangle_count = 0;
    int skipped_invalid_triangle_count = 0;
    int skipped_degenerate_triangle_count = 0;
    int skipped_non_triangle_face_count = 0;
    int invalid_normal_reference_count = 0;
    int nonfinite_normal_reference_count = 0;
    int zero_normal_reference_count = 0;
    int invalid_texcoord_reference_count = 0;
    int nonfinite_texcoord_reference_count = 0;
    int unused_material_binding_count = 0;
    ObjMaterialSource material_source = ObjMaterialSource::None;
    std::unordered_map<std::string, int> material_name_to_id;
    std::vector<std::string> material_names;
    std::vector<ObjMeshGroupRange> groups;
    std::vector<std::string> warnings;
};

bool load_obj_into_scene(const std::string& obj_path, Scene& scene,
                         const ObjLoadOptions& options = {},
                         SceneMeshAppendRange* appended_range = nullptr);

bool load_obj_into_scene_with_result(const std::string& obj_path, Scene& scene,
                                     const ObjLoadOptions& options,
                                     ObjLoadResult& result);

#endif // LUX_UTIL_OBJ_LOADER_H
