#pragma once

#ifndef LUX_INTEGRATOR_WORKITEMS_CUH
#define LUX_INTEGRATOR_WORKITEMS_CUH

#include "core/ray.cuh"
#include "core/types.h"
#include "core/vec3.cuh"
#include "scene/primitive_ref.h"

// Work items exchanged between wavefront kernels.
struct RayWorkItem {
    int path_id;
    Ray ray;
};

struct HitWorkItem {
    int path_id;
    int triangle_id;
    PrimitiveRef primitive_ref;
    vec3 ray_origin;
    vec3 position;
    vec3 ng;
    vec3 ns;
    vec3 wo;  // outgoing direction (=-ray.direction) for BSDF evaluation
    Float u = 0;  // barycentric
    Float v = 0;  // barycentric
};

struct ShadowRayWorkItem {
    int path_id;
    int source_triangle_id;
    PrimitiveRef source_primitive_ref;
    int source_material_id;
    int source_depth;
    Ray shadow_ray;
    vec3 contribution;
    vec3 source_geometric_normal;
    vec3 source_shading_normal;
    Float pdf_light_solid_angle = 0;
    Float pdf_bsdf_solid_angle = 0;
};

// Terminated paths waiting for shadow-ray results before accumulation.
struct TerminatedPathItem {
    int path_id;
};

#endif // LUX_INTEGRATOR_WORKITEMS_CUH
