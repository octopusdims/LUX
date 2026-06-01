#pragma once

#ifndef LUX_SCENE_PRIMITIVE_REF_H
#define LUX_SCENE_PRIMITIVE_REF_H

#include "core/types.h"
#include "geometry/primitive_kind.h"

struct PrimitiveRef {
    int instance_id = -1;
    int mesh_id = -1;
    int local_primitive_id = -1;
    PrimitiveKind kind = PrimitiveKind::Triangle;
};

LuxHDInline PrimitiveRef invalid_primitive_ref() {
    return PrimitiveRef{-1, -1, -1, PrimitiveKind::None};
}

LuxHDInline bool primitive_ref_valid(const PrimitiveRef& ref) {
    return ref.kind != PrimitiveKind::None
        && ref.instance_id >= 0 && ref.mesh_id >= 0 && ref.local_primitive_id >= 0;
}

LuxHDInline bool same_primitive_ref(const PrimitiveRef& a, const PrimitiveRef& b) {
    return a.kind == b.kind
        && a.instance_id == b.instance_id
        && a.mesh_id == b.mesh_id
        && a.local_primitive_id == b.local_primitive_id;
}

#endif // LUX_SCENE_PRIMITIVE_REF_H
