#include "util/env_image_loader.h"

#if defined(LUX_HAS_OPENEXR) && LUX_HAS_OPENEXR
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfArray.h>
#include <Imath/ImathBox.h>
#endif

LoadedEnvironmentImage load_exr_environment_image(const std::string& path) {
#if defined(LUX_HAS_OPENEXR) && LUX_HAS_OPENEXR
    Imf::RgbaInputFile file(path.c_str());
    Imath::Box2i window = file.dataWindow();
    int width = window.max.x - window.min.x + 1;
    int height = window.max.y - window.min.y + 1;
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("invalid EXR data window");
    }

    Imf::Array2D<Imf::Rgba> rgba;
    rgba.resizeErase(height, width);
    file.setFrameBuffer(
        &rgba[-window.min.y][-window.min.x], 1, static_cast<size_t>(width));
    file.readPixels(window.min.y, window.max.y);

    LoadedEnvironmentImage image;
    image.width = width;
    image.height = height;
    image.pixels.assign(width * height, vec3(0));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const Imf::Rgba& pixel = rgba[y][x];
            image.pixels[y * width + x] =
                vec3(static_cast<float>(pixel.r),
                     static_cast<float>(pixel.g),
                     static_cast<float>(pixel.b));
        }
    }
    return image;
#else
    (void)path;
    throw std::runtime_error(
        "OpenEXR environment loading is not enabled in this build");
#endif
}
