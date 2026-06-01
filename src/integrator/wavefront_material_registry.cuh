#pragma once

#ifndef LUX_INTEGRATOR_WAVEFRONT_MATERIAL_REGISTRY_CUH
#define LUX_INTEGRATOR_WAVEFRONT_MATERIAL_REGISTRY_CUH

#include "material/material.h"

#define LUX_WAVEFRONT_MATERIAL_QUEUE_LIST(X) \
    X(Lambert,      MaterialType::Lambert,      "Lambert hit queue") \
    X(ThinLambert,  MaterialType::ThinLambert,  "ThinLambert hit queue") \
    X(GGX,          MaterialType::GGX,          "GGX hit queue") \
    X(Dielectric,   MaterialType::Dielectric,   "Dielectric hit queue") \
    X(Passthrough,  MaterialType::Transparent,  "passthrough hit queue")

#define LUX_WAVEFRONT_MATERIAL_QUEUE_MAP(X) \
    X(MaterialType::Lambert,      MaterialQueueKind::Lambert) \
    X(MaterialType::ThinLambert,  MaterialQueueKind::ThinLambert) \
    X(MaterialType::GGX,          MaterialQueueKind::GGX) \
    X(MaterialType::Dielectric,   MaterialQueueKind::Dielectric) \
    X(MaterialType::Interface,    MaterialQueueKind::Passthrough) \
    X(MaterialType::Transparent,  MaterialQueueKind::Passthrough)

#endif // LUX_INTEGRATOR_WAVEFRONT_MATERIAL_REGISTRY_CUH
