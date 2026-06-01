#pragma once

#ifndef LUX_APP_RENDER_RUNNER_H
#define LUX_APP_RENDER_RUNNER_H

#include <string>

#include "app/render_config.h"

std::string available_render_scene_names();
int run_render(const RenderConfig& config);

#endif // LUX_APP_RENDER_RUNNER_H
