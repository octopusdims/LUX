#pragma once

#ifndef LUX_SCENE_STAMP_H
#define LUX_SCENE_STAMP_H

#include <cstdint>
#include <limits>

#include "core/types.h"

struct SceneStamp {
    std::uint64_t identity = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t revision = std::numeric_limits<std::uint64_t>::max();

    LuxHDInline bool matches(const SceneStamp& other) const {
        return identity == other.identity && revision == other.revision;
    }
};

#endif // LUX_SCENE_STAMP_H
