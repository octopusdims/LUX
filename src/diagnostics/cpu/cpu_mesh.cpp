#include "cpu_diagnostics_internal.h"

namespace lux_cpu_diagnostics_detail {

void print_winding_probe(const Scene& scene, int triangle_id) {
    if (triangle_id < 0 || triangle_id >= scene_triangle_count(scene)) {
        throw std::runtime_error("-debug-triangle triangle id is outside the scene");
    }

    SceneTriangle source = scene_triangle_view(scene, triangle_id);
    vec3 source_ng = triangle_normal(source.triangle);
    vec3 source_center = triangle_centroid(source.triangle);
    vec3 material_center = material_area_centroid(scene, source.material_id);
    vec3 source_radial = source_center - material_center;
    Float source_radial_dot = length2(source_radial) > 0
        ? dot(source_ng, normalize(source_radial)) : 0;

    std::printf("Winding probe tri=%d mat=%d has_vn=%d boundary=0x%x area=%.8g\n",
                triangle_id, source.material_id, has_vertex_normals(source) ? 1 : 0,
                static_cast<unsigned>(source.boundary_edges),
                double(triangle_area(source.triangle)));
    std::printf("  center=(%.8g,%.8g,%.8g) ng=(%.8g,%.8g,%.8g) "
                "ng_material_radial=%.8g\n",
                double(source_center.x), double(source_center.y),
                double(source_center.z), double(source_ng.x),
                double(source_ng.y), double(source_ng.z),
                double(source_radial_dot));

    const char* edge_names[3] = {"v0-v1", "v1-v2", "v2-v0"};
    int total_neighbors = 0;
    int consistent_edges = 0;
    int inconsistent_edges = 0;
    int boundary_edges = 0;
    int nonmanifold_edges = 0;

    for (int edge = 0; edge < 3; ++edge) {
        int edge_neighbors = 0;
        int edge_consistent = 0;
        int edge_inconsistent = 0;
        std::printf("  edge %d (%s):\n", edge, edge_names[edge]);

        for (int i = 0; i < scene_triangle_count(scene); ++i) {
            if (i == triangle_id) continue;

            int neighbor_from = -1;
            int neighbor_to = -1;
            SceneTriangle candidate = scene_triangle_view(scene, i);
            int orientation = shared_edge_orientation(
                source.triangle, edge, candidate.triangle,
                neighbor_from, neighbor_to);
            if (orientation == 0) continue;

            SceneTriangle neighbor = candidate;
            vec3 neighbor_ng = triangle_normal(neighbor.triangle);
            vec3 neighbor_center = triangle_centroid(neighbor.triangle);
            vec3 neighbor_radial = neighbor_center
                - material_area_centroid(scene, neighbor.material_id);
            Float neighbor_radial_dot = length2(neighbor_radial) > 0
                ? dot(neighbor_ng, normalize(neighbor_radial)) : 0;
            bool consistent = orientation < 0;

            ++edge_neighbors;
            ++total_neighbors;
            if (consistent) {
                ++edge_consistent;
                ++consistent_edges;
            } else {
                ++edge_inconsistent;
                ++inconsistent_edges;
            }

            std::printf("    neighbor tri=%d mat=%d shared_edge_neighbor=%d->%d "
                        "orientation=%s winding=%s dot_n=%.8g "
                        "ng=(%.8g,%.8g,%.8g) ng_material_radial=%.8g "
                        "boundary=0x%x\n",
                        i, neighbor.material_id, neighbor_from, neighbor_to,
                        orientation < 0 ? "opposite" : "same",
                        consistent ? "consistent" : "BROKEN",
                        double(dot(source_ng, neighbor_ng)),
                        double(neighbor_ng.x), double(neighbor_ng.y),
                        double(neighbor_ng.z), double(neighbor_radial_dot),
                        static_cast<unsigned>(neighbor.boundary_edges));
        }

        if (edge_neighbors == 0) {
            ++boundary_edges;
            std::printf("    no edge neighbor: boundary/open edge\n");
        } else if (edge_neighbors > 1) {
            ++nonmanifold_edges;
            std::printf("    nonmanifold: %d neighbors share this edge\n",
                        edge_neighbors);
        }
        std::printf("    edge_summary: neighbors=%d consistent=%d broken=%d\n",
                    edge_neighbors, edge_consistent, edge_inconsistent);
    }

    std::printf("  summary: total_edge_neighbors=%d consistent=%d broken=%d "
                "boundary_edges=%d nonmanifold_edges=%d\n",
                total_neighbors, consistent_edges, inconsistent_edges,
                boundary_edges, nonmanifold_edges);
}

struct OrientationComponent {
    int id = -1;
    int triangle_count = 0;
    Float area = 0;
    Float score_sum = 0;
    Float min_score = std::numeric_limits<Float>::infinity();
    Float max_score = -std::numeric_limits<Float>::infinity();
    int patch_boundary_edges = 0;
    int mesh_boundary_edges = 0;
    int nonmanifold_edges = 0;
    vec3 p_min = vec3(std::numeric_limits<Float>::infinity());
    vec3 p_max = vec3(-std::numeric_limits<Float>::infinity());
};

std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash>
build_material_edge_map(const Scene& scene, int material_id) {
    std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edge_map;
    for (int i = 0; i < scene_triangle_count(scene); ++i) {
        SceneTriangle scene_triangle = scene_triangle_view(scene, i);
        if (scene_triangle.material_id != material_id) continue;
        const Triangle& tri = scene_triangle.triangle;
        edge_map[make_edge_key(tri.v0, tri.v1)].push_back(i);
        edge_map[make_edge_key(tri.v1, tri.v2)].push_back(i);
        edge_map[make_edge_key(tri.v2, tri.v0)].push_back(i);
    }
    return edge_map;
}

void expand_component_bounds(OrientationComponent& component, const Triangle& tri) {
    component.p_min.x = std::min(component.p_min.x, std::min(tri.v0.x, std::min(tri.v1.x, tri.v2.x)));
    component.p_min.y = std::min(component.p_min.y, std::min(tri.v0.y, std::min(tri.v1.y, tri.v2.y)));
    component.p_min.z = std::min(component.p_min.z, std::min(tri.v0.z, std::min(tri.v1.z, tri.v2.z)));
    component.p_max.x = std::max(component.p_max.x, std::max(tri.v0.x, std::max(tri.v1.x, tri.v2.x)));
    component.p_max.y = std::max(component.p_max.y, std::max(tri.v0.y, std::max(tri.v1.y, tri.v2.y)));
    component.p_max.z = std::max(component.p_max.z, std::max(tri.v0.z, std::max(tri.v1.z, tri.v2.z)));
}

void print_orientation_audit(const Scene& scene) {
    std::vector<vec3> material_centers = compute_material_area_centroids(scene);
    std::vector<Float> scores = compute_radial_orientation_scores(scene, material_centers);
    int target_material = largest_non_emissive_material(scene);

    std::printf("Orientation audit: target_material=%d (largest non-emissive)\n",
                target_material);

    for (int material_id = 0; material_id < static_cast<int>(scene.materials.size());
         ++material_id) {
        int count = 0;
        int negative = 0;
        int positive = 0;
        int near_zero = 0;
        Float area = 0;
        Float score_sum = 0;
        Float min_score = std::numeric_limits<Float>::infinity();
        Float max_score = -std::numeric_limits<Float>::infinity();

        for (int i = 0; i < scene_triangle_count(scene); ++i) {
            SceneTriangle scene_triangle = scene_triangle_view(scene, i);
            if (scene_triangle.material_id != material_id) continue;
            ++count;
            area += triangle_area(scene_triangle.triangle);
            Float score = scores[i];
            score_sum += score;
            min_score = std::min(min_score, score);
            max_score = std::max(max_score, score);
            if (score < Float(0)) {
                ++negative;
            } else if (score > Float(0)) {
                ++positive;
            } else {
                ++near_zero;
            }
        }

        if (count == 0) continue;
        std::printf("  material %d: triangles=%d emissive=%d area=%.8g "
                    "radial_negative=%d radial_positive=%d near_zero=%d "
                    "score[min,avg,max]=[%.8g,%.8g,%.8g] center=(%.8g,%.8g,%.8g)\n",
                    material_id, count, scene.materials[material_id].is_emissive() ? 1 : 0,
                    double(area), negative, positive, near_zero,
                    double(min_score), double(score_sum / Float(count)),
                    double(max_score), double(material_centers[material_id].x),
                    double(material_centers[material_id].y),
                    double(material_centers[material_id].z));
    }

    if (target_material < 0) {
        std::printf("No non-emissive target material to component-audit\n");
        return;
    }

    auto edge_map = build_material_edge_map(scene, target_material);
    std::vector<char> visited(scene_triangle_count(scene), 0);
    std::vector<int> component_id(scene_triangle_count(scene), -1);
    std::vector<OrientationComponent> components;

    for (int seed = 0; seed < scene_triangle_count(scene); ++seed) {
        SceneTriangle seed_triangle = scene_triangle_view(scene, seed);
        if (visited[seed] || seed_triangle.material_id != target_material
            || scores[seed] >= Float(0)) {
            continue;
        }

        OrientationComponent component;
        component.id = static_cast<int>(components.size());
        std::queue<int> queue;
        queue.push(seed);
        visited[seed] = 1;
        component_id[seed] = component.id;

        while (!queue.empty()) {
            int tri_id = queue.front();
            queue.pop();
            Triangle tri = scene_triangle_geometry(scene, tri_id);
            Float area = triangle_area(tri);
            Float score = scores[tri_id];

            ++component.triangle_count;
            component.area += area;
            component.score_sum += score;
            component.min_score = std::min(component.min_score, score);
            component.max_score = std::max(component.max_score, score);
            expand_component_bounds(component, tri);

            std::array<EdgeKey, 3> edges = {
                make_edge_key(tri.v0, tri.v1),
                make_edge_key(tri.v1, tri.v2),
                make_edge_key(tri.v2, tri.v0)
            };

            for (const EdgeKey& edge : edges) {
                const std::vector<int>& edge_tris = edge_map[edge];
                if (edge_tris.size() == 1) {
                    ++component.mesh_boundary_edges;
                }
                if (edge_tris.size() > 2) {
                    ++component.nonmanifold_edges;
                }

                bool has_suspect_neighbor = false;
                for (int neighbor_id : edge_tris) {
                    if (neighbor_id == tri_id) continue;
                    if (scene_triangle_view(scene, neighbor_id).material_id
                        != target_material) continue;
                    if (scores[neighbor_id] < Float(0)) {
                        has_suspect_neighbor = true;
                        if (!visited[neighbor_id]) {
                            visited[neighbor_id] = 1;
                            component_id[neighbor_id] = component.id;
                            queue.push(neighbor_id);
                        }
                    }
                }
                if (!has_suspect_neighbor) {
                    ++component.patch_boundary_edges;
                }
            }
        }

        components.push_back(component);
    }

    std::sort(components.begin(), components.end(),
              [](const OrientationComponent& a, const OrientationComponent& b) {
                  if (a.triangle_count != b.triangle_count) {
                      return a.triangle_count > b.triangle_count;
                  }
                  return a.area > b.area;
              });

    std::printf("  suspect_components=%zu (radial score < 0 on target material)\n",
                components.size());
    int printed = std::min<int>(10, static_cast<int>(components.size()));
    for (int i = 0; i < printed; ++i) {
        const OrientationComponent& component = components[i];
        Float avg_score = component.triangle_count > 0
            ? component.score_sum / Float(component.triangle_count) : 0;
        std::printf("    component %d: original_id=%d triangles=%d area=%.8g "
                    "score[min,avg,max]=[%.8g,%.8g,%.8g] patch_boundary=%d "
                    "mesh_boundary=%d nonmanifold=%d bbox=[(%.8g,%.8g,%.8g),"
                    "(%.8g,%.8g,%.8g)]\n",
                    i, component.id, component.triangle_count, double(component.area),
                    double(component.min_score), double(avg_score),
                    double(component.max_score), component.patch_boundary_edges,
                    component.mesh_boundary_edges, component.nonmanifold_edges,
                    double(component.p_min.x), double(component.p_min.y),
                    double(component.p_min.z), double(component.p_max.x),
                    double(component.p_max.y), double(component.p_max.z));
    }
}


} // namespace lux_cpu_diagnostics_detail
