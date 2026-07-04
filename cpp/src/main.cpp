// C++ aerial OBB inference CLI — flags aligned with predict_aerial.py.
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "pipeline.hpp"

namespace {

#if !defined(_WIN32) && defined(ORT_RUNTIME_LIB_PATH)
// Linux ORT-CUDA: dlopen'd provider deps live in the venv nvidia-*-cu12 libs.
// Re-exec once with LD_LIBRARY_PATH so CUDA EP can load without a wrapper script.
void ensure_runtime_lib_path(char** argv) {
    if (std::getenv("AERIAL_OBB_REEXEC")) return;
    const char* cur = std::getenv("LD_LIBRARY_PATH");
    std::string want = ORT_RUNTIME_LIB_PATH;
    std::string merged = cur && *cur ? want + ":" + cur : want;
    setenv("LD_LIBRARY_PATH", merged.c_str(), 1);
    setenv("AERIAL_OBB_REEXEC", "1", 1);
    execv("/proc/self/exe", argv);
}
#else
void ensure_runtime_lib_path(char** argv) { (void)argv; }
#endif

}  // namespace

namespace {

void usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " --image PATH --json PATH [options]\n"
        << "  --image, -i PATH       input image (required)\n"
        << "  --model, -m PATH       ONNX model (default runs/zhuangji_obb-2/weights/best.onnx)\n"
        << "  --json PATH            output JSON (required)\n"
        << "  --preview PATH         output preview JPEG\n"
        << "  --tile-size N          (default 640)\n"
        << "  --overlap N            fine tile overlap (default 96)\n"
        << "  --conf F               (default 0.25)\n"
        << "  --device STR           'cpu' or CUDA index '0' (default cpu)\n"
        << "  --strategy STR         two-stage | full-scan (default two-stage)\n"
        << "  --full-scan            shorthand for --strategy full-scan\n"
        << "  --coarse-stride N      (default 1280)\n"
        << "  --coarse-conf F        (default 0.35)\n"
        << "  --hot-margin N         (default 320)\n"
        << "  --nms-iou F            cross-tile merge IoU (default 0.35)\n"
        << "  --intra-threads N      ORT intra-op threads, CPU only (default: ORT auto)\n"
        << "  --no-prefetch          disable next-batch prefetch\n"
        << "  --skip-preview         do not render preview (default on)\n"
        << "  --profile              print per-stage timing\n";
}

bool eat(const char* a, const char* s, const char* l) {
    return std::strcmp(a, s) == 0 || (l && std::strcmp(a, l) == 0);
}

}  // namespace

int main(int argc, char** argv) {
    ensure_runtime_lib_path(argv);

    aerial::Config cfg;
    cfg.model_path = "runs/zhuangji_obb-2/weights/best.onnx";
    cfg.skip_preview = true;
    bool have_preview = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (a == "--image" || a == "-i") cfg.image_path = next("--image");
        else if (a == "--model" || a == "-m") cfg.model_path = next("--model");
        else if (a == "--json") cfg.json_path = next("--json");
        else if (a == "--preview") { cfg.preview_path = next("--preview"); have_preview = true; }
        else if (a == "--tile-size") cfg.tile_size = std::stoi(next("--tile-size"));
        else if (a == "--overlap") cfg.overlap = std::stoi(next("--overlap"));
        else if (a == "--conf") cfg.conf = std::stof(next("--conf"));
        else if (a == "--device") cfg.device = next("--device");
        else if (a == "--strategy") cfg.strategy = next("--strategy");
        else if (a == "--full-scan") cfg.strategy = "full-scan";
        else if (a == "--coarse-stride") cfg.coarse_stride = std::stoi(next("--coarse-stride"));
        else if (a == "--coarse-conf") cfg.coarse_conf = std::stof(next("--coarse-conf"));
        else if (a == "--hot-margin") cfg.hot_margin = std::stoi(next("--hot-margin"));
        else if (a == "--nms-iou") cfg.nms_iou = std::stof(next("--nms-iou"));
        else if (a == "--intra-threads") cfg.intra_threads = std::stoi(next("--intra-threads"));
        else if (a == "--no-prefetch") cfg.prefetch = false;
        else if (a == "--skip-preview") cfg.skip_preview = true;
        else if (a == "--profile") cfg.profile = true;
        else { std::cerr << "unknown arg: " << a << "\n"; usage(argv[0]); return 2; }
    }

    if (cfg.image_path.empty() || cfg.json_path.empty()) {
        std::cerr << "--image and --json are required\n";
        usage(argv[0]);
        return 2;
    }
    if (have_preview) cfg.skip_preview = false;

    try {
        aerial::run_pipeline(cfg);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
