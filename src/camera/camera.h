#pragma once

#ifndef LUX_CAMERA_H
#define LUX_CAMERA_H

#include "core/constants.h"
#include "core/ray.cuh"
#include "core/types.h"
#include "core/vec2.cuh"
#include "core/vec3.cuh"

enum class CameraKind {
    Perspective,
    Orthographic,
};

struct CameraSample {
    vec2 p_film = vec2(0);
    vec2 p_lens = vec2(0);
    Float time = 0;
};

struct CameraRay {
    Ray ray;
    vec3 weight = vec3(1);
    bool valid = true;
};

struct PerspectiveCamera {
    vec3 origin = vec3(0);
    vec3 lower_left = vec3(0);
    vec3 horizontal = vec3(0);
    vec3 vertical = vec3(0);
    vec3 lens_u = vec3(1, 0, 0);
    vec3 lens_v = vec3(0, 1, 0);
    Float lens_radius = 0;
    Float focal_distance = 1;
};

struct OrthographicCamera {
    vec3 lower_left = vec3(0);
    vec3 horizontal = vec3(0);
    vec3 vertical = vec3(0);
    vec3 direction = vec3(0, 0, -1);
};

struct Camera {
    CameraKind kind = CameraKind::Perspective;
    PerspectiveCamera perspective;
    OrthographicCamera orthographic;
};

namespace lux_camera_detail {

LuxHDInline vec3 safe_normalize(const vec3& value) {
    Float len = length(value);
    return len > 0 ? value / len : value;
}

LuxHDInline vec2 concentric_sample_disk(const vec2& u) {
    Float sx = Float(2) * u.x - Float(1);
    Float sy = Float(2) * u.y - Float(1);
    if (sx == 0 && sy == 0) return vec2(0);

    Float r;
    Float theta;
    if (fabsf(sx) > fabsf(sy)) {
        r = sx;
        theta = (kPi / Float(4)) * (sy / sx);
    } else {
        r = sy;
        theta = (kPi / Float(2)) - (kPi / Float(4)) * (sx / sy);
    }
    return vec2(r * cosf(theta), r * sinf(theta));
}

struct CameraFrame {
    vec3 forward;
    vec3 right;
    vec3 up;
};

LuxHDInline CameraFrame make_camera_frame(const vec3& origin, const vec3& look_at) {
    vec3 forward = safe_normalize(look_at - origin);
    vec3 world_up(0, 1, 0);
    vec3 right = safe_normalize(cross(world_up, forward));
    if (length2(right) == 0) {
        right = safe_normalize(cross(vec3(0, 0, 1), forward));
    }
    vec3 up = cross(forward, right);
    return CameraFrame{forward, right, up};
}

} // namespace lux_camera_detail

LuxInline Camera make_perspective_camera(const vec3& origin, const vec3& look_at,
                                         Float vertical_fov_degrees, Float aspect,
                                         Float lens_radius = 0,
                                         Float focal_distance = 1) {
    lux_camera_detail::CameraFrame frame =
        lux_camera_detail::make_camera_frame(origin, look_at);

    Float theta = vertical_fov_degrees * kPi / Float(180);
    Float viewport_height = Float(2) * tanf(theta * Float(0.5));
    Float viewport_width = aspect * viewport_height;
    Float focus = focal_distance > 0 ? focal_distance : length(look_at - origin);
    if (focus <= 0) focus = 1;

    vec3 horizontal = (viewport_width * focus) * frame.right;
    vec3 vertical = (viewport_height * focus) * frame.up;
    vec3 lower_left = origin + focus * frame.forward - horizontal * Float(0.5)
                    - vertical * Float(0.5);

    Camera camera;
    camera.kind = CameraKind::Perspective;
    camera.perspective = PerspectiveCamera{
        origin, lower_left, horizontal, vertical,
        frame.right, frame.up, lens_radius, focus
    };
    return camera;
}

LuxInline Camera make_orthographic_camera(const vec3& origin, const vec3& look_at,
                                          Float vertical_size, Float aspect) {
    lux_camera_detail::CameraFrame frame =
        lux_camera_detail::make_camera_frame(origin, look_at);
    Float height = vertical_size > 0 ? vertical_size : Float(2);
    Float width = aspect * height;

    Camera camera;
    camera.kind = CameraKind::Orthographic;
    camera.orthographic = OrthographicCamera{
        origin - frame.right * (width * Float(0.5)) - frame.up * (height * Float(0.5)),
        frame.right * width,
        frame.up * height,
        frame.forward
    };
    return camera;
}

LuxHDInline CameraRay generate_camera_ray(const PerspectiveCamera& camera,
                                          const CameraSample& sample) {
    vec3 target = camera.lower_left + sample.p_film.x * camera.horizontal
                + sample.p_film.y * camera.vertical;
    vec3 origin = camera.origin;
    if (camera.lens_radius > 0) {
        vec2 lens = lux_camera_detail::concentric_sample_disk(sample.p_lens)
                  * camera.lens_radius;
        origin += lens.x * camera.lens_u + lens.y * camera.lens_v;
    }
    vec3 direction = lux_camera_detail::safe_normalize(target - origin);
    return CameraRay{Ray(origin, direction, kRayEpsilon, INFINITY, sample.time), vec3(1), true};
}

LuxHDInline CameraRay generate_camera_ray(const OrthographicCamera& camera,
                                          const CameraSample& sample) {
    vec3 origin = camera.lower_left + sample.p_film.x * camera.horizontal
                + sample.p_film.y * camera.vertical;
    return CameraRay{Ray(origin, camera.direction, kRayEpsilon, INFINITY, sample.time),
                     vec3(1), true};
}

LuxHDInline CameraRay generate_camera_ray(const Camera& camera,
                                          const CameraSample& sample) {
    switch (camera.kind) {
        case CameraKind::Perspective:
            return generate_camera_ray(camera.perspective, sample);
        case CameraKind::Orthographic:
            return generate_camera_ray(camera.orthographic, sample);
    }
    return CameraRay{Ray(), vec3(0), false};
}

LuxHDInline Ray generate_ray(const Camera& camera, Float u, Float v) {
    return generate_camera_ray(camera, CameraSample{vec2(u, v), vec2(0), 0}).ray;
}

#endif // LUX_CAMERA_H
