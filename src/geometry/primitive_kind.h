#pragma once

#ifndef LUX_GEOMETRY_PRIMITIVE_KIND_H
#define LUX_GEOMETRY_PRIMITIVE_KIND_H

enum class PrimitiveKind : unsigned char {
    None,
    Triangle,
    Sphere,
    Curve,
    Volume,
};

#endif // LUX_GEOMETRY_PRIMITIVE_KIND_H
