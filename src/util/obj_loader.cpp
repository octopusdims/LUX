#include "obj_loader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/validation.cuh"
#include "geometry/triangle.h"
#include "material/material.h"

namespace {

std::string trim_ascii_whitespace(const std::string& text) {
    size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

bool read_text_file(const std::string& path, std::string& text) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    std::ostringstream out;
    out << file.rdbuf();
    text = out.str();
    return true;
}

void record_obj_warning(ObjLoadResult* result, const std::string& warning) {
    if (warning.empty()) return;
    if (result) {
        result->warnings.push_back(warning);
    }
    std::fprintf(stderr, "OBJ warning: %s\n", warning.c_str());
}

bool is_ignored_tinyobj_material_warning(const std::string& warning) {
    return warning.find("material [") != std::string::npos
        && warning.find("not found in .mtl") != std::string::npos;
}

void record_tinyobj_warnings(ObjLoadResult* result, const std::string& warnings) {
    std::istringstream input(warnings);
    std::string line;
    while (std::getline(input, line)) {
        std::string warning = trim_ascii_whitespace(line);
        if (warning.empty() || is_ignored_tinyobj_material_warning(warning)) {
            continue;
        }
        record_obj_warning(result, warning);
    }
}

void record_obj_count_warning(ObjLoadResult* result,
                              int count,
                              const std::string& warning) {
    if (count <= 0) return;
    record_obj_warning(result, warning + ": " + std::to_string(count));
}

bool record_obj_error(ObjLoadResult* result, const std::string& error) {
    if (result) {
        result->warnings.push_back(error);
    }
    std::fprintf(stderr, "OBJ error: %s\n", error.c_str());
    return false;
}

bool obj_keyword_matches(const std::string& trimmed, const char* keyword) {
    size_t length = std::char_traits<char>::length(keyword);
    if (trimmed.size() < length || trimmed.compare(0, length, keyword) != 0) {
        return false;
    }
    return trimmed.size() == length
        || trimmed[length] == ' '
        || trimmed[length] == '\t';
}

std::string obj_keyword_payload(const std::string& trimmed, const char* keyword) {
    size_t length = std::char_traits<char>::length(keyword);
    if (trimmed.size() <= length) return {};
    return trim_ascii_whitespace(trimmed.substr(length));
}

std::string strip_obj_comment(const std::string& line) {
    size_t comment = line.find('#');
    if (comment == std::string::npos) return line;
    return line.substr(0, comment);
}

int count_obj_face_vertices(const std::string& payload) {
    int count = 0;
    std::istringstream input(payload);
    std::string token;
    while (input >> token) {
        ++count;
    }
    return count;
}

struct ObjMaterialLabelScan {
    std::vector<std::string> triangle_material_names;
    std::vector<std::string> material_names;
    bool references_mtllib = false;
};

ObjMaterialLabelScan scan_obj_material_labels(const std::string& obj_text) {
    ObjMaterialLabelScan scan;
    std::unordered_set<std::string> seen_material_names;
    std::string current_material_name;

    std::istringstream input(obj_text);
    std::string line;
    while (std::getline(input, line)) {
        std::string trimmed = trim_ascii_whitespace(strip_obj_comment(line));
        if (trimmed.empty()) continue;

        if (obj_keyword_matches(trimmed, "mtllib")) {
            scan.references_mtllib = true;
            continue;
        }
        if (obj_keyword_matches(trimmed, "usemtl")) {
            current_material_name = obj_keyword_payload(trimmed, "usemtl");
            if (!current_material_name.empty()
                && seen_material_names.insert(current_material_name).second) {
                scan.material_names.push_back(current_material_name);
            }
            continue;
        }
        if (!obj_keyword_matches(trimmed, "f")) {
            continue;
        }

        int vertex_count = count_obj_face_vertices(obj_keyword_payload(trimmed, "f"));
        int triangle_count = vertex_count >= 3 ? vertex_count - 2 : 0;
        for (int i = 0; i < triangle_count; ++i) {
            scan.triangle_material_names.push_back(current_material_name);
        }
    }

    return scan;
}

bool load_tinyobj_without_mtl(const std::string& obj_text,
                              tinyobj::attrib_t& attrib,
                              std::vector<tinyobj::shape_t>& shapes,
                              std::vector<tinyobj::material_t>& materials,
                              std::string& warn,
                              std::string& err) {
    std::istringstream obj_stream(obj_text);
    return tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                            &obj_stream, nullptr, true, true);
}

size_t count_tinyobj_face_entries(const std::vector<tinyobj::shape_t>& shapes) {
    size_t count = 0;
    for (const tinyobj::shape_t& shape : shapes) {
        count += shape.mesh.num_face_vertices.size();
    }
    return count;
}

bool scene_material_id_valid(const Scene& scene, int material_id) {
    return material_id >= 0 && material_id < static_cast<int>(scene.materials.size());
}

bool validate_obj_material_bindings(const Scene& scene,
                                    const ObjLoadOptions& options,
                                    ObjLoadResult* result) {
    if (options.default_material_id >= 0
        && !scene_material_id_valid(scene, options.default_material_id)) {
        return record_obj_error(
            result,
            "OBJ default_material_id is outside the scene material array");
    }

    for (const auto& binding : options.material_bindings) {
        if (!scene_material_id_valid(scene, binding.second)) {
            return record_obj_error(
                result,
                "OBJ material binding '" + binding.first
                    + "' references an invalid scene material id");
        }
    }
    return true;
}

int resolve_default_obj_material_id(const Scene& scene,
                                    const ObjLoadOptions& options,
                                    bool* append_material) {
    *append_material = false;
    if (options.default_material_id >= 0) {
        return options.default_material_id;
    }
    if (!options.material_bindings.empty()) {
        return -1;
    }

    int material_id = static_cast<int>(scene.materials.size());
    *append_material = true;
    return material_id;
}

bool resolve_obj_face_material(const Scene& scene,
                               const ObjLoadOptions& options,
                               const std::string& material_name,
                               int default_material_id,
                               int* material_id,
                               ObjLoadResult* result) {
    if (!options.material_bindings.empty()) {
        if (!material_name.empty()) {
            auto found = options.material_bindings.find(material_name);
            if (found == options.material_bindings.end()) {
                return record_obj_error(
                    result,
                    "OBJ usemtl '" + material_name
                        + "' has no material binding");
            }
            *material_id = found->second;
            if (result) {
                result->material_name_to_id[material_name] = found->second;
            }
            return true;
        }

        if (default_material_id >= 0) {
            *material_id = default_material_id;
            return true;
        }

        return record_obj_error(
            result,
            "OBJ face has no usemtl tag while material bindings are required");
    }

    *material_id = default_material_id;
    return true;
}

bool validate_obj_material_labels(const ObjLoadOptions& options,
                                  const ObjMaterialLabelScan& material_scan,
                                  int default_material_id,
                                  ObjLoadResult* result) {
    if (options.material_bindings.empty()) {
        return true;
    }

    for (const std::string& material_name : material_scan.material_names) {
        if (options.material_bindings.find(material_name)
            == options.material_bindings.end()) {
            return record_obj_error(
                result,
                "OBJ usemtl '" + material_name
                    + "' has no material binding");
        }
    }

    int unused_binding_count = 0;
    for (const auto& binding : options.material_bindings) {
        if (std::find(material_scan.material_names.begin(),
                      material_scan.material_names.end(),
                      binding.first) == material_scan.material_names.end()) {
            ++unused_binding_count;
            record_obj_warning(
                result,
                "OBJ material binding '" + binding.first
                    + "' did not match any usemtl tag");
        }
    }
    if (result) {
        result->unused_material_binding_count = unused_binding_count;
    }

    if (default_material_id >= 0) {
        return true;
    }

    for (const std::string& material_name : material_scan.triangle_material_names) {
        if (material_name.empty()) {
            return record_obj_error(
                result,
                "OBJ face has no usemtl tag while material bindings are required");
        }
    }
    return true;
}

ObjNormalMode effective_obj_normal_mode(const ObjLoadOptions& options) {
    if (options.normal_mode != ObjNormalMode::SourceOrGeometry) {
        return options.normal_mode;
    }
    return options.generate_missing_normals
        ? ObjNormalMode::GenerateMissingSmooth
        : ObjNormalMode::SourceOrGeometry;
}

bool validate_obj_instance_transform_options(const ObjLoadOptions& options,
                                             ObjLoadResult* result) {
    if (!options.create_instance) {
        return true;
    }
    if (instance_transform_valid(options.object_to_world)) {
        return true;
    }
    if (!instance_transform_matrices_finite(options.object_to_world)) {
        return record_obj_error(
            result,
            "OBJ object_to_world transform must contain only finite values");
    }
    if (!instance_transform_matrices_affine_invertible(options.object_to_world)) {
        return record_obj_error(
            result,
            "OBJ object_to_world transform must be affine-invertible");
    }
    if (!transform_inverse_pair_consistent(options.object_to_world)) {
        return record_obj_error(
            result,
            "OBJ object_to_world transform and inverse must be consistent");
    }
    return true;
}

bool validate_obj_placement_options(const ObjLoadOptions& options,
                                    ObjLoadResult* result) {
    if (!options.create_instance) {
        return true;
    }
    if (options.placement.scale_mode != ObjPlacementScaleMode::None
        && (!std::isfinite(static_cast<double>(options.placement.target_extent))
            || options.placement.target_extent <= Float(0))) {
        return record_obj_error(
            result,
            "OBJ placement target_extent must be finite and positive when scaling is enabled");
    }
    if (options.placement.ground_to_y
        && !std::isfinite(static_cast<double>(options.placement.target_y))) {
        return record_obj_error(
            result,
            "OBJ placement target_y must be finite when grounding is enabled");
    }
    return true;
}

bool validate_obj_weld_options(const ObjLoadOptions& options,
                               ObjLoadResult* result) {
    bool uses_welded_positions =
        options.boundary_topology == ObjBoundaryTopology::WeldedPositions
        || (options.cap_boundary_holes
            && options.cap_boundary_topology == ObjBoundaryTopology::WeldedPositions);
    if (!uses_welded_positions) {
        return true;
    }
    if (!std::isfinite(static_cast<double>(options.position_weld_epsilon))
        || options.position_weld_epsilon <= Float(0)) {
        return record_obj_error(
            result,
            "OBJ position_weld_epsilon must be finite and positive when welded topology is enabled");
    }
    return true;
}

Float obj_placement_source_extent(const AABB& bounds,
                                  ObjPlacementScaleMode mode) {
    vec3 extent = bounds.diagonal();
    switch (mode) {
    case ObjPlacementScaleMode::MaxExtent:
        return std::max({extent.x, extent.y, extent.z});
    case ObjPlacementScaleMode::XExtent:
        return extent.x;
    case ObjPlacementScaleMode::None:
    default:
        return Float(0);
    }
}

Transform make_obj_placement_transform(const AABB& bounds,
                                       const ObjPlacementOptions& placement,
                                       ObjLoadResult* result) {
    if (bounds.is_empty()) {
        return Transform{};
    }

    Float scale_factor = Float(1);
    if (placement.scale_mode != ObjPlacementScaleMode::None) {
        Float source_extent =
            obj_placement_source_extent(bounds, placement.scale_mode);
        if (source_extent > Float(0)) {
            scale_factor = placement.target_extent / source_extent;
        } else {
            record_obj_warning(
                result,
                "OBJ placement scaling skipped because selected source extent is zero");
        }
    }

    vec3 anchor(0);
    vec3 target(0);
    if (placement.center_xz) {
        vec3 center = bounds.center();
        anchor.x = center.x;
        anchor.z = center.z;
    }
    if (placement.ground_to_y) {
        anchor.y = bounds.p_min.y;
        target.y = placement.target_y;
    }

    mat4 placement_matrix =
        translate(target) * scale(vec3(scale_factor)) * translate(-anchor);
    return make_transform(placement_matrix);
}

struct PositionKey {
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    bool operator==(const PositionKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct PositionKeyHash {
    size_t operator()(const PositionKey& key) const {
        std::uint64_t h = static_cast<std::uint64_t>(key.x);
        h = h * 1099511628211ULL ^ static_cast<std::uint64_t>(key.y);
        h = h * 1099511628211ULL ^ static_cast<std::uint64_t>(key.z);
        return static_cast<size_t>(h);
    }
};

PositionKey make_position_key(const vec3& position, Float epsilon) {
    double inv_epsilon = 1.0 / static_cast<double>(epsilon > 0 ? epsilon : Float(1e-6));
    return PositionKey{
        static_cast<std::int64_t>(std::llround(static_cast<double>(position.x) * inv_epsilon)),
        static_cast<std::int64_t>(std::llround(static_cast<double>(position.y) * inv_epsilon)),
        static_cast<std::int64_t>(std::llround(static_cast<double>(position.z) * inv_epsilon))};
}

std::uint64_t edge_key(int a, int b) {
    std::uint32_t lo = static_cast<std::uint32_t>(std::min(a, b));
    std::uint32_t hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32) | hi;
}

struct LoadedFace {
    tinyobj::index_t idx[3];
    int material_id = -1;
    std::string material_name;
    std::string shape_name;
};

struct BoundaryEdge {
    int from = -1;
    int to = -1;
    int material_id = -1;
};

struct TopologyVertices {
    std::vector<int> ids_by_obj_vertex;
    std::vector<vec3> positions;
};

TopologyVertices make_topology_vertices(
        const tinyobj::attrib_t& attrib,
        ObjBoundaryTopology topology,
        Float weld_epsilon,
        const std::vector<unsigned char>& vertex_finite) {
    int vertex_count = static_cast<int>(attrib.vertices.size() / 3);
    TopologyVertices result;
    result.ids_by_obj_vertex.resize(vertex_count, -1);

    auto vertex_position = [&](int vertex_index) -> vec3 {
        return vec3(attrib.vertices[3 * vertex_index + 0],
                    attrib.vertices[3 * vertex_index + 1],
                    attrib.vertices[3 * vertex_index + 2]);
    };

    if (topology == ObjBoundaryTopology::SourceVertexIndices) {
        result.positions.resize(vertex_count, vec3(0));
        for (int i = 0; i < vertex_count; ++i) {
            if (!vertex_finite.empty() && !vertex_finite[i]) continue;
            result.ids_by_obj_vertex[i] = i;
            result.positions[i] = vertex_position(i);
        }
        return result;
    }

    std::unordered_map<PositionKey, int, PositionKeyHash> welded_ids;
    welded_ids.reserve(vertex_count);
    for (int i = 0; i < vertex_count; ++i) {
        if (!vertex_finite.empty() && !vertex_finite[i]) continue;
        vec3 position = vertex_position(i);
        PositionKey key = make_position_key(position, weld_epsilon);
        auto found = welded_ids.find(key);
        if (found != welded_ids.end()) {
            result.ids_by_obj_vertex[i] = found->second;
            continue;
        }

        int id = static_cast<int>(result.positions.size());
        welded_ids.emplace(key, id);
        result.ids_by_obj_vertex[i] = id;
        result.positions.push_back(position);
    }
    return result;
}

std::unordered_map<std::uint64_t, int> count_topology_edges(
        const std::vector<LoadedFace>& faces,
        const std::vector<int>& topology_ids) {
    std::unordered_map<std::uint64_t, int> counts;
    counts.reserve(faces.size() * 3);
    for (const LoadedFace& face : faces) {
        int ids[3] = {
            topology_ids[face.idx[0].vertex_index],
            topology_ids[face.idx[1].vertex_index],
            topology_ids[face.idx[2].vertex_index]
        };
        if (ids[0] != ids[1]) ++counts[edge_key(ids[0], ids[1])];
        if (ids[1] != ids[2]) ++counts[edge_key(ids[1], ids[2])];
        if (ids[2] != ids[0]) ++counts[edge_key(ids[2], ids[0])];
    }
    return counts;
}

int count_topology_boundary_edges(
        const std::unordered_map<std::uint64_t, int>& counts) {
    int boundary_edges = 0;
    for (const auto& entry : counts) {
        if (entry.second == 1) {
            ++boundary_edges;
        }
    }
    return boundary_edges;
}

vec3 topology_center_for_faces(const std::vector<LoadedFace>& faces,
                               const TopologyVertices& topology) {
    vec3 center(0);
    int count = 0;
    std::unordered_set<int> seen;
    seen.reserve(faces.size() * 3);
    for (const LoadedFace& face : faces) {
        for (int i = 0; i < 3; ++i) {
            int obj_vertex = face.idx[i].vertex_index;
            if (obj_vertex < 0
                || obj_vertex >= static_cast<int>(topology.ids_by_obj_vertex.size())) {
                continue;
            }
            int topology_id = topology.ids_by_obj_vertex[obj_vertex];
            if (topology_id < 0
                || topology_id >= static_cast<int>(topology.positions.size())) {
                continue;
            }
            if (!seen.insert(topology_id).second) {
                continue;
            }
            center += topology.positions[topology_id];
            ++count;
        }
    }
    if (count == 0) return vec3(0);
    return center / static_cast<Float>(count);
}

bool edge_is_boundary(const std::unordered_map<std::uint64_t, int>& counts,
                      int a, int b) {
    if (a < 0 || b < 0 || a == b) return false;
    auto found = counts.find(edge_key(a, b));
    return found != counts.end() && found->second == 1;
}

void append_or_extend_group(std::vector<ObjMeshGroupRange>& groups,
                            const std::string& shape_name,
                            const std::string& material_name,
                            int material_id,
                            int triangle_id) {
    if (!groups.empty()) {
        ObjMeshGroupRange& last = groups.back();
        if (last.shape_name == shape_name
            && last.material_name == material_name
            && last.material_id == material_id
            && last.first_triangle + last.triangle_count == triangle_id) {
            ++last.triangle_count;
            return;
        }
    }

    ObjMeshGroupRange group;
    group.shape_name = shape_name;
    group.material_name = material_name;
    group.material_id = material_id;
    group.first_triangle = triangle_id;
    group.triangle_count = 1;
    groups.push_back(std::move(group));
}

} // namespace

static bool load_obj_into_scene_impl(const std::string& obj_path, Scene& scene,
                                     const ObjLoadOptions& options,
                                     ObjLoadResult* result) {
    if (result) {
        *result = ObjLoadResult{};
    }

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    std::string obj_text;
    if (!read_text_file(obj_path, obj_text)) {
        return record_obj_error(
            result,
            "Cannot open OBJ file [" + obj_path + "]");
    }

    ObjMaterialLabelScan material_scan = scan_obj_material_labels(obj_text);
    if (!validate_obj_material_bindings(scene, options, result)) {
        return false;
    }
    if (!validate_obj_placement_options(options, result)) {
        return false;
    }
    if (!validate_obj_instance_transform_options(options, result)) {
        return false;
    }
    if (!validate_obj_weld_options(options, result)) {
        return false;
    }

    bool ret = load_tinyobj_without_mtl(
        obj_text, attrib, shapes, materials, warn, err);

    record_tinyobj_warnings(result, warn);
    if (!err.empty()) {
        std::fprintf(stderr, "OBJ error: %s\n", err.c_str());
        if (result) result->warnings.push_back(err);
    }
    if (!ret) {
        return false;
    }
    if (material_scan.references_mtllib) {
        record_obj_warning(
            result,
            "OBJ mtllib ignored; LUX does not import MTL yet");
    }
    size_t parsed_face_count = count_tinyobj_face_entries(shapes);
    if (parsed_face_count != material_scan.triangle_material_names.size()) {
        return record_obj_error(
            result,
            "OBJ usemtl label count does not match parsed triangle count");
    }

    bool append_default_material = false;
    int default_material_id = resolve_default_obj_material_id(
        scene, options, &append_default_material);
    if (!validate_obj_material_labels(
            options, material_scan, default_material_id, result)) {
        return false;
    }
    if (result) {
        result->first_material_id = default_material_id;
        result->material_count = 0;
        result->material_source = options.material_bindings.empty()
            ? ObjMaterialSource::DefaultMaterial
            : ObjMaterialSource::UsemtlBinding;
        result->material_names = material_scan.material_names;
        result->material_name_to_id.clear();
    }

    auto vertex_position = [&](int vertex_index) -> vec3 {
        return vec3(attrib.vertices[3 * vertex_index + 0],
                    attrib.vertices[3 * vertex_index + 1],
                    attrib.vertices[3 * vertex_index + 2]);
    };
    int obj_vertex_count = static_cast<int>(attrib.vertices.size() / 3);
    int obj_normal_count = static_cast<int>(attrib.normals.size() / 3);
    int obj_texcoord_count = static_cast<int>(attrib.texcoords.size() / 2);
    std::vector<unsigned char> vertex_finite(obj_vertex_count, 1);
    std::vector<unsigned char> normal_finite(obj_normal_count, 1);
    std::vector<unsigned char> texcoord_finite(obj_texcoord_count, 1);
    AABB source_bounds;
    for (int i = 0; i < obj_vertex_count; ++i) {
        vec3 position = vertex_position(i);
        vertex_finite[i] = finite_vec3(position) ? 1u : 0u;
        if (vertex_finite[i]) {
            source_bounds.expand(position);
        }
    }
    for (int i = 0; i < obj_normal_count; ++i) {
        vec3 normal(attrib.normals[3 * i + 0],
                    attrib.normals[3 * i + 1],
                    attrib.normals[3 * i + 2]);
        normal_finite[i] = finite_vec3(normal) ? 1u : 0u;
    }
    for (int i = 0; i < obj_texcoord_count; ++i) {
        vec2 texcoord(attrib.texcoords[2 * i + 0],
                      attrib.texcoords[2 * i + 1]);
        texcoord_finite[i] = finite_vec2(texcoord) ? 1u : 0u;
    }
    auto vertex_index_valid = [&](const tinyobj::index_t& idx) -> bool {
        return idx.vertex_index >= 0 && idx.vertex_index < obj_vertex_count;
    };
    auto vertex_index_finite = [&](const tinyobj::index_t& idx) -> bool {
        return vertex_index_valid(idx) && vertex_finite[idx.vertex_index];
    };

    ObjNormalMode normal_mode = effective_obj_normal_mode(options);
    std::vector<vec3> generated_normal_sums(attrib.vertices.size() / 3, vec3(0));
    std::vector<LoadedFace> loaded_faces;
    size_t material_label_cursor = 0;
    int nonfinite_vertex_count = 0;
    for (unsigned char finite : vertex_finite) {
        if (!finite) ++nonfinite_vertex_count;
    }
    int skipped_nonfinite_triangle_count = 0;
    int skipped_invalid_triangle_count = 0;
    int skipped_degenerate_triangle_count = 0;
    int skipped_non_triangle_face_count = 0;
    int invalid_normal_reference_count = 0;
    int nonfinite_normal_reference_count = 0;
    int zero_normal_reference_count = 0;
    int invalid_texcoord_reference_count = 0;
    int nonfinite_texcoord_reference_count = 0;
    if (result) {
        result->source_triangle_count =
            static_cast<int>(material_scan.triangle_material_names.size());
        result->source_vertex_count = obj_vertex_count;
        result->source_bounds = source_bounds;
        result->nonfinite_vertex_count = nonfinite_vertex_count;
    }

    auto normal_index_invalid = [&](const tinyobj::index_t& idx) -> bool {
        int ni = idx.normal_index;
        return ni >= obj_normal_count;
    };
    auto normal_index_nonfinite = [&](const tinyobj::index_t& idx) -> bool {
        int ni = idx.normal_index;
        return ni >= 0 && ni < obj_normal_count && !normal_finite[ni];
    };
    auto normal_index_nonzero = [&](const tinyobj::index_t& idx) -> bool {
        int ni = idx.normal_index;
        if (ni < 0 || ni >= obj_normal_count || !normal_finite[ni]) {
            return false;
        }
        vec3 normal(attrib.normals[3 * ni + 0],
                    attrib.normals[3 * ni + 1],
                    attrib.normals[3 * ni + 2]);
        return length2(normal) > Float(1e-20);
    };
    auto normal_index_zero = [&](const tinyobj::index_t& idx) -> bool {
        int ni = idx.normal_index;
        if (ni < 0 || ni >= obj_normal_count || !normal_finite[ni]) {
            return false;
        }
        vec3 normal(attrib.normals[3 * ni + 0],
                    attrib.normals[3 * ni + 1],
                    attrib.normals[3 * ni + 2]);
        return length2(normal) <= Float(1e-20);
    };
    auto texcoord_index_invalid = [&](const tinyobj::index_t& idx) -> bool {
        int ti = idx.texcoord_index;
        return ti >= obj_texcoord_count;
    };
    auto texcoord_index_nonfinite = [&](const tinyobj::index_t& idx) -> bool {
        int ti = idx.texcoord_index;
        return ti >= 0 && ti < obj_texcoord_count && !texcoord_finite[ti];
    };

    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f];
            std::string material_name;
            if (material_label_cursor < material_scan.triangle_material_names.size()) {
                material_name = material_scan.triangle_material_names[material_label_cursor];
            }
            ++material_label_cursor;

            if (fv != 3) {
                ++skipped_non_triangle_face_count;
                index_offset += fv;
                continue;
            }

            tinyobj::index_t idx0 = shape.mesh.indices[index_offset + 0];
            tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];
            if (!vertex_index_valid(idx0) || !vertex_index_valid(idx1)
                || !vertex_index_valid(idx2)) {
                ++skipped_invalid_triangle_count;
                index_offset += fv;
                continue;
            }
            if (!vertex_index_finite(idx0) || !vertex_index_finite(idx1)
                || !vertex_index_finite(idx2)) {
                ++skipped_nonfinite_triangle_count;
                index_offset += fv;
                continue;
            }

            vec3 v0 = vertex_position(idx0.vertex_index);
            vec3 v1 = vertex_position(idx1.vertex_index);
            vec3 v2 = vertex_position(idx2.vertex_index);
            Triangle tri{v0, v1, v2};
            if (triangle_area(tri) <= Float(1e-12)) {
                ++skipped_degenerate_triangle_count;
                index_offset += fv;
                continue;
            }

            int mat_id = -1;
            if (!resolve_obj_face_material(
                    scene, options, material_name, default_material_id, &mat_id,
                    result)) {
                return false;
            }

            LoadedFace face;
            face.idx[0] = idx0;
            face.idx[1] = options.reverse_orientation ? idx2 : idx1;
            face.idx[2] = options.reverse_orientation ? idx1 : idx2;
            face.material_id = mat_id;
            face.material_name = material_name;
            face.shape_name = shape.name;
            loaded_faces.push_back(std::move(face));

            if (normal_index_invalid(idx0)) ++invalid_normal_reference_count;
            if (normal_index_invalid(idx1)) ++invalid_normal_reference_count;
            if (normal_index_invalid(idx2)) ++invalid_normal_reference_count;
            if (normal_index_nonfinite(idx0)) ++nonfinite_normal_reference_count;
            if (normal_index_nonfinite(idx1)) ++nonfinite_normal_reference_count;
            if (normal_index_nonfinite(idx2)) ++nonfinite_normal_reference_count;
            if (normal_index_zero(idx0)) ++zero_normal_reference_count;
            if (normal_index_zero(idx1)) ++zero_normal_reference_count;
            if (normal_index_zero(idx2)) ++zero_normal_reference_count;
            if (texcoord_index_invalid(idx0)) ++invalid_texcoord_reference_count;
            if (texcoord_index_invalid(idx1)) ++invalid_texcoord_reference_count;
            if (texcoord_index_invalid(idx2)) ++invalid_texcoord_reference_count;
            if (texcoord_index_nonfinite(idx0)) ++nonfinite_texcoord_reference_count;
            if (texcoord_index_nonfinite(idx1)) ++nonfinite_texcoord_reference_count;
            if (texcoord_index_nonfinite(idx2)) ++nonfinite_texcoord_reference_count;

            if (normal_mode == ObjNormalMode::GenerateMissingSmooth) {
                vec3 area_weighted_normal = cross(v1 - v0, v2 - v0);
                generated_normal_sums[idx0.vertex_index] += area_weighted_normal;
                generated_normal_sums[idx1.vertex_index] += area_weighted_normal;
                generated_normal_sums[idx2.vertex_index] += area_weighted_normal;
            }

            index_offset += fv;
        }
    }
    if (material_label_cursor != material_scan.triangle_material_names.size()) {
        return record_obj_error(
            result,
            "OBJ usemtl label count changed while loading parsed triangles");
    }
    if (result) {
        result->loaded_triangle_count = static_cast<int>(loaded_faces.size());
        result->skipped_nonfinite_triangle_count = skipped_nonfinite_triangle_count;
        result->skipped_invalid_triangle_count = skipped_invalid_triangle_count;
        result->skipped_degenerate_triangle_count = skipped_degenerate_triangle_count;
        result->skipped_non_triangle_face_count = skipped_non_triangle_face_count;
        result->invalid_normal_reference_count = invalid_normal_reference_count;
        result->nonfinite_normal_reference_count = nonfinite_normal_reference_count;
        result->zero_normal_reference_count = zero_normal_reference_count;
        result->invalid_texcoord_reference_count = invalid_texcoord_reference_count;
        result->nonfinite_texcoord_reference_count = nonfinite_texcoord_reference_count;
    }
    record_obj_count_warning(
        result,
        nonfinite_vertex_count,
        "OBJ encountered non-finite vertex positions");
    record_obj_count_warning(
        result,
        skipped_invalid_triangle_count,
        "OBJ skipped triangles with invalid vertex indices");
    record_obj_count_warning(
        result,
        skipped_nonfinite_triangle_count,
        "OBJ skipped triangles with non-finite vertex positions");
    record_obj_count_warning(
        result,
        skipped_degenerate_triangle_count,
        "OBJ skipped degenerate triangles");
    record_obj_count_warning(
        result,
        skipped_non_triangle_face_count,
        "OBJ skipped non-triangle faces after triangulation");
    record_obj_count_warning(
        result,
        invalid_normal_reference_count,
        "OBJ encountered invalid normal references");
    record_obj_count_warning(
        result,
        nonfinite_normal_reference_count,
        "OBJ encountered non-finite normal references");
    record_obj_count_warning(
        result,
        zero_normal_reference_count,
        "OBJ encountered zero-length normal references");
    record_obj_count_warning(
        result,
        invalid_texcoord_reference_count,
        "OBJ encountered invalid texcoord references");
    record_obj_count_warning(
        result,
        nonfinite_texcoord_reference_count,
        "OBJ encountered non-finite texcoord references");

    if (loaded_faces.empty()) {
        if (result) {
            result->first_material_id = -1;
            result->material_count = 0;
            result->material_source = ObjMaterialSource::None;
            result->material_name_to_id.clear();
        }
        return true;
    }
    if (append_default_material && !loaded_faces.empty()) {
        scene.materials.push_back(Material::lambert(vec3(0.8f)));
        if (result) result->material_count = 1;
    }

    TopologyVertices boundary_topology = make_topology_vertices(
        attrib, options.boundary_topology, options.position_weld_epsilon,
        vertex_finite);
    auto boundary_edge_counts = count_topology_edges(
        loaded_faces, boundary_topology.ids_by_obj_vertex);
    int source_boundary_edge_count = count_topology_boundary_edges(boundary_edge_counts);
    TopologyVertices cap_topology;
    std::unordered_map<std::uint64_t, int> cap_edge_counts;
    int cap_boundary_edge_count = 0;
    if (options.cap_boundary_holes) {
        cap_topology = make_topology_vertices(
            attrib, options.cap_boundary_topology, options.position_weld_epsilon,
            vertex_finite);
        cap_edge_counts = count_topology_edges(
            loaded_faces, cap_topology.ids_by_obj_vertex);
        cap_boundary_edge_count = count_topology_boundary_edges(cap_edge_counts);
    }

    struct ObjVertexKey {
        int vertex_index;
        int normal_index;
        int texcoord_index;

        bool operator==(const ObjVertexKey& other) const {
            return vertex_index == other.vertex_index
                && normal_index == other.normal_index
                && texcoord_index == other.texcoord_index;
        }
    };

    struct ObjVertexKeyHash {
        size_t operator()(const ObjVertexKey& key) const {
            std::uint64_t h = static_cast<std::uint32_t>(key.vertex_index);
            h = h * 1099511628211ULL ^ static_cast<std::uint32_t>(key.normal_index);
            h = h * 1099511628211ULL ^ static_cast<std::uint32_t>(key.texcoord_index);
            return static_cast<size_t>(h);
        }
    };

    std::vector<BoundaryEdge> boundary_edges_to_cap;
    std::vector<vec3> mesh_vertices;
    std::vector<vec3> mesh_normals;
    std::vector<vec2> mesh_uvs;
    std::vector<TriangleIndices> mesh_indices;
    std::vector<int> mesh_material_ids;
    std::vector<unsigned char> mesh_boundary_edges;
    std::unordered_map<ObjVertexKey, std::uint32_t, ObjVertexKeyHash> vertex_map;
    std::vector<ObjMeshGroupRange> groups;

    mesh_vertices.reserve(loaded_faces.size() * 3);
    mesh_normals.reserve(loaded_faces.size() * 3);
    mesh_uvs.reserve(loaded_faces.size() * 3);
    mesh_indices.reserve(loaded_faces.size());
    mesh_material_ids.reserve(loaded_faces.size());
    mesh_boundary_edges.reserve(loaded_faces.size());
    vertex_map.reserve(loaded_faces.size() * 3);

    auto get_normal = [&](const tinyobj::index_t& idx) -> vec3 {
        if (normal_mode == ObjNormalMode::FaceOnly) {
            return vec3(0);
        }

        int ni = idx.normal_index;
        if (normal_index_nonzero(idx)) {
            vec3 normal = normalize(vec3(attrib.normals[3 * ni + 0],
                                         attrib.normals[3 * ni + 1],
                                         attrib.normals[3 * ni + 2]));
            return options.reverse_orientation ? -normal : normal;
        }
        if (normal_mode == ObjNormalMode::GenerateMissingSmooth && idx.vertex_index >= 0) {
            vec3 normal = generated_normal_sums[idx.vertex_index];
            if (length2(normal) > 0) {
                normal = normalize(normal);
                return options.reverse_orientation ? -normal : normal;
            }
        }
        return vec3(0);
    };

    auto normal_index_used = [&](const tinyobj::index_t& idx) -> bool {
        if (normal_mode == ObjNormalMode::FaceOnly) {
            return false;
        }
        int ni = idx.normal_index;
        return ni >= 0 && normal_index_nonzero(idx);
    };

    auto get_texcoord = [&](const tinyobj::index_t& idx) -> vec2 {
        int ti = idx.texcoord_index;
        if (ti < 0 || ti >= obj_texcoord_count || !texcoord_finite[ti]) {
            return vec2(0);
        }
        Float u = attrib.texcoords[2 * ti + 0];
        Float v = attrib.texcoords[2 * ti + 1];
        if (options.flip_texcoord_v) {
            v = Float(1) - v;
        }
        return vec2(u, v);
    };

    auto texcoord_index_used = [&](const tinyobj::index_t& idx) -> bool {
        int ti = idx.texcoord_index;
        return ti >= 0 && ti < obj_texcoord_count && texcoord_finite[ti];
    };

    auto make_vertex_key = [&](const tinyobj::index_t& idx) -> ObjVertexKey {
        int normal_index = normal_index_used(idx) ? idx.normal_index : -1;
        int texcoord_index = texcoord_index_used(idx) ? idx.texcoord_index : -1;
        return ObjVertexKey{idx.vertex_index, normal_index, texcoord_index};
    };

    auto get_or_add_vertex = [&](const tinyobj::index_t& idx) -> std::uint32_t {
        ObjVertexKey key = make_vertex_key(idx);
        auto found = vertex_map.find(key);
        if (found != vertex_map.end()) return found->second;

        std::uint32_t local_index = static_cast<std::uint32_t>(mesh_vertices.size());
        vertex_map.emplace(key, local_index);
        mesh_vertices.push_back(vertex_position(idx.vertex_index));
        mesh_normals.push_back(get_normal(idx));
        mesh_uvs.push_back(get_texcoord(idx));
        return local_index;
    };

    for (const LoadedFace& loaded : loaded_faces) {
        int obj_vertices[3] = {
            loaded.idx[0].vertex_index,
            loaded.idx[1].vertex_index,
            loaded.idx[2].vertex_index
        };
        int boundary_ids[3] = {
            boundary_topology.ids_by_obj_vertex[obj_vertices[0]],
            boundary_topology.ids_by_obj_vertex[obj_vertices[1]],
            boundary_topology.ids_by_obj_vertex[obj_vertices[2]]
        };
        unsigned char boundary_edges = 0;
        if (edge_is_boundary(boundary_edge_counts, boundary_ids[0], boundary_ids[1])) {
            boundary_edges |= 1u;
        }
        if (edge_is_boundary(boundary_edge_counts, boundary_ids[1], boundary_ids[2])) {
            boundary_edges |= 2u;
        }
        if (edge_is_boundary(boundary_edge_counts, boundary_ids[2], boundary_ids[0])) {
            boundary_edges |= 4u;
        }

        if (options.cap_boundary_holes) {
            int cap_ids[3] = {
                cap_topology.ids_by_obj_vertex[obj_vertices[0]],
                cap_topology.ids_by_obj_vertex[obj_vertices[1]],
                cap_topology.ids_by_obj_vertex[obj_vertices[2]]
            };
            if (edge_is_boundary(cap_edge_counts, cap_ids[0], cap_ids[1])) {
                boundary_edges_to_cap.push_back(
                    BoundaryEdge{cap_ids[0], cap_ids[1], loaded.material_id});
            }
            if (edge_is_boundary(cap_edge_counts, cap_ids[1], cap_ids[2])) {
                boundary_edges_to_cap.push_back(
                    BoundaryEdge{cap_ids[1], cap_ids[2], loaded.material_id});
            }
            if (edge_is_boundary(cap_edge_counts, cap_ids[2], cap_ids[0])) {
                boundary_edges_to_cap.push_back(
                    BoundaryEdge{cap_ids[2], cap_ids[0], loaded.material_id});
            }
        }

        std::uint32_t v0 = get_or_add_vertex(loaded.idx[0]);
        std::uint32_t v1 = get_or_add_vertex(loaded.idx[1]);
        std::uint32_t v2 = get_or_add_vertex(loaded.idx[2]);
        int triangle_id = static_cast<int>(mesh_indices.size());
        mesh_indices.push_back(TriangleIndices{v0, v1, v2});
        mesh_material_ids.push_back(loaded.material_id);
        mesh_boundary_edges.push_back(boundary_edges);
        append_or_extend_group(
            groups, loaded.shape_name, loaded.material_name, loaded.material_id,
            triangle_id);
    }

    int cap_count = 0;
    int cap_non_simple_boundary_vertex_count = 0;
    int cap_incomplete_loop_count = 0;
    int cap_degenerate_loop_count = 0;
    int cap_degenerate_triangle_count = 0;
    if (options.cap_boundary_holes && !boundary_edges_to_cap.empty()) {
        vec3 mesh_center = topology_center_for_faces(loaded_faces, cap_topology);

        std::unordered_map<int, std::vector<BoundaryEdge>> outgoing_edges;
        std::unordered_map<int, int> incoming_edge_counts;
        outgoing_edges.reserve(boundary_edges_to_cap.size());
        incoming_edge_counts.reserve(boundary_edges_to_cap.size());
        for (const BoundaryEdge& edge : boundary_edges_to_cap) {
            outgoing_edges[edge.from].push_back(edge);
            ++incoming_edge_counts[edge.to];
        }

        for (const auto& entry : outgoing_edges) {
            int vertex_id = entry.first;
            int incoming_count = 0;
            auto incoming = incoming_edge_counts.find(vertex_id);
            if (incoming != incoming_edge_counts.end()) {
                incoming_count = incoming->second;
            }
            if (entry.second.size() != 1 || incoming_count != 1) {
                ++cap_non_simple_boundary_vertex_count;
            }
        }
        for (const auto& entry : incoming_edge_counts) {
            if (outgoing_edges.find(entry.first) == outgoing_edges.end()) {
                ++cap_non_simple_boundary_vertex_count;
            }
        }

        std::unordered_map<int, BoundaryEdge> next_edge;
        if (cap_non_simple_boundary_vertex_count == 0) {
            next_edge.reserve(boundary_edges_to_cap.size());
            for (const BoundaryEdge& edge : boundary_edges_to_cap) {
                next_edge.emplace(edge.from, edge);
            }
        } else {
            record_obj_warning(
                result,
                "OBJ boundary capping skipped non-simple boundary topology; "
                "preserving source boundary edge masks");
        }

        std::unordered_set<int> visited_starts;
        visited_starts.reserve(boundary_edges_to_cap.size());
        for (const BoundaryEdge& first_edge : boundary_edges_to_cap) {
            if (cap_non_simple_boundary_vertex_count != 0) break;
            if (visited_starts.find(first_edge.from) != visited_starts.end()) continue;

            std::vector<int> loop;
            int material_id = first_edge.material_id;
            int start = first_edge.from;
            int current = start;
            while (visited_starts.find(current) == visited_starts.end()) {
                auto found = next_edge.find(current);
                if (found == next_edge.end()) break;
                visited_starts.insert(current);
                loop.push_back(current);
                material_id = found->second.material_id;
                current = found->second.to;
                if (current == start) break;
            }

            if (current != start) {
                ++cap_incomplete_loop_count;
                continue;
            }
            if (loop.size() < 3) {
                ++cap_degenerate_loop_count;
                continue;
            }

            vec3 cap_center(0);
            for (int vertex_id : loop) {
                cap_center += cap_topology.positions[vertex_id];
            }
            cap_center /= static_cast<Float>(loop.size());

            for (size_t i = 0; i < loop.size(); ++i) {
                vec3 v0 = cap_center;
                vec3 v1 = cap_topology.positions[loop[i]];
                vec3 v2 = cap_topology.positions[loop[(i + 1) % loop.size()]];
                Triangle cap_tri{v0, v1, v2};
                vec3 cap_normal = triangle_normal(cap_tri);
                if (triangle_area(cap_tri) <= Float(1e-12)) {
                    ++cap_degenerate_triangle_count;
                    continue;
                }
                if (dot(cap_normal, cap_center - mesh_center) < 0) {
                    cap_tri = Triangle{v0, v2, v1};
                    cap_normal = -cap_normal;
                }
                if (options.reverse_orientation) {
                    cap_tri = Triangle{cap_tri.v0, cap_tri.v2, cap_tri.v1};
                    cap_normal = -cap_normal;
                }

                std::uint32_t first_vertex =
                    static_cast<std::uint32_t>(mesh_vertices.size());
                mesh_vertices.push_back(cap_tri.v0);
                mesh_vertices.push_back(cap_tri.v1);
                mesh_vertices.push_back(cap_tri.v2);
                mesh_normals.push_back(cap_normal);
                mesh_normals.push_back(cap_normal);
                mesh_normals.push_back(cap_normal);
                mesh_uvs.push_back(vec2(0));
                mesh_uvs.push_back(vec2(0));
                mesh_uvs.push_back(vec2(0));
                int triangle_id = static_cast<int>(mesh_indices.size());
                mesh_indices.push_back(TriangleIndices{
                    first_vertex, first_vertex + 1, first_vertex + 2});
                mesh_material_ids.push_back(material_id);
                mesh_boundary_edges.push_back(0);
                append_or_extend_group(groups, "__caps__", {}, material_id, triangle_id);
                ++cap_count;
            }
        }

        if (cap_count > 0) {
            std::fprintf(stderr, "OBJ capped %d boundary triangles\n", cap_count);
        }
        if (result) {
            result->cap_triangle_count = cap_count;
        }
        if (cap_incomplete_loop_count > 0) {
            record_obj_warning(
                result,
                "OBJ boundary capping skipped open boundary loops");
        }
        if (cap_degenerate_loop_count > 0 || cap_degenerate_triangle_count > 0) {
            record_obj_warning(
                result,
                "OBJ boundary capping skipped degenerate loops or cap triangles");
        }
    }

    if (mesh_indices.empty()) {
        return true;
    }

    SceneMesh mesh;
    mesh.vertices = std::move(mesh_vertices);
    mesh.normals = std::move(mesh_normals);
    mesh.uvs = std::move(mesh_uvs);
    mesh.indices = std::move(mesh_indices);
    mesh.material_ids = std::move(mesh_material_ids);
    mesh.boundary_edges = std::move(mesh_boundary_edges);

    int mesh_id = add_scene_mesh(scene, std::move(mesh));
    const SceneMesh& stored_mesh = scene.mesh_assets[mesh_id];
    int instance_id = -1;
    AABB world_bounds;
    if (options.create_instance) {
        Transform placement_transform =
            make_obj_placement_transform(stored_mesh.bounds, options.placement, result);
        Transform instance_transform = options.object_to_world * placement_transform;
        instance_id = add_scene_instance(scene, mesh_id, instance_transform);
        world_bounds = scene.instances[instance_id].world_bounds;
    }

    if (result) {
        result->mesh_id = mesh_id;
        result->instance_id = instance_id;
        result->appended_range = SceneMeshAppendRange{
            mesh_id,
            0,
            static_cast<int>(stored_mesh.vertices.size()),
            0,
            static_cast<int>(stored_mesh.indices.size())
        };
        result->mesh_vertex_count = static_cast<int>(stored_mesh.vertices.size());
        result->mesh_triangle_count = static_cast<int>(stored_mesh.indices.size());
        result->mesh_bounds = stored_mesh.bounds;
        result->world_bounds = world_bounds;
        result->source_boundary_edge_count = source_boundary_edge_count;
        result->cap_boundary_edge_count = cap_boundary_edge_count;
        result->cap_non_simple_boundary_vertex_count =
            cap_non_simple_boundary_vertex_count;
        result->cap_incomplete_loop_count = cap_incomplete_loop_count;
        result->cap_degenerate_loop_count = cap_degenerate_loop_count;
        result->cap_degenerate_triangle_count = cap_degenerate_triangle_count;
        result->groups = std::move(groups);
        for (ObjMeshGroupRange& group : result->groups) {
            group.mesh_id = mesh_id;
        }
    }

    return true;
}

bool load_obj_into_scene_with_result(const std::string& obj_path, Scene& scene,
                                     const ObjLoadOptions& options,
                                     ObjLoadResult& result) {
    return load_obj_into_scene_impl(obj_path, scene, options, &result);
}

bool load_obj_into_scene(const std::string& obj_path, Scene& scene,
                         const ObjLoadOptions& options,
                         SceneMeshAppendRange* appended_range) {
    ObjLoadResult result;
    bool ok = load_obj_into_scene_impl(obj_path, scene, options, &result);
    if (ok && appended_range) {
        *appended_range = result.appended_range;
    }
    return ok;
}
