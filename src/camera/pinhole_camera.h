#pragma once

#ifndef LUX_CAMERA_PINHOLE_H
#define LUX_CAMERA_PINHOLE_H

#include "camera/camera.h"

struct PinholeCamera : Camera {
    LuxHDInline PinholeCamera() = default;
    LuxHDInline explicit PinholeCamera(const Camera& camera) : Camera(camera) {}
};

LuxInline PinholeCamera make_pinhole_camera(const vec3& origin, const vec3& look_at,
                                            Float vertical_fov_degrees, Float aspect) {
    return PinholeCamera(
        make_perspective_camera(origin, look_at, vertical_fov_degrees, aspect, 0, 1));
}

#endif // LUX_CAMERA_PINHOLE_H
