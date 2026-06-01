#include "app/render_config.h"
#include "app/render_runner.h"

#include <string>

int main(int argc, char** argv) {
    std::string scene_names = available_render_scene_names();
    RenderConfig config;
    if (!parse_render_config(argc, argv, config)) {
        print_render_usage(argv[0], scene_names.c_str());
        return 1;
    }
    if (config.show_help) {
        print_render_usage(argv[0], scene_names.c_str());
        return 0;
    }
    return run_render(config);
}
