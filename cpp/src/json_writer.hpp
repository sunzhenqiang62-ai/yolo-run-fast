// Emit the same JSON payload shape as predict_aerial.py:
//   {image, size{width,height}, count, backend, strategy, detections[
//     {score, polygon[[x,y]x4], xyxy[4]} ]}
// Hand-rolled writer (no JSON lib dependency); matches indent=2 layout closely.
#pragma once

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "obb_decode.hpp"

namespace aerial {

inline std::string fmt_num(double v) {
    // Match Python json float repr closely enough for diffing; trim trailing
    // zeros while keeping integers compact.
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.10g", v);
    return std::string(buf);
}

inline void write_json(const std::string& path, const std::string& image_path,
                       int orig_w, int orig_h, const std::string& backend,
                       const std::string& strategy,
                       const std::vector<RawHit>& dets) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open json output: " + path);

    auto esc = [](const std::string& s) {
        std::string o;
        for (char c : s) {
            if (c == '\\' || c == '"') o += '\\';
            o += c;
        }
        return o;
    };

    f << "{\n";
    f << "  \"image\": \"" << esc(image_path) << "\",\n";
    f << "  \"size\": {\n    \"width\": " << orig_w << ",\n    \"height\": " << orig_h
      << "\n  },\n";
    f << "  \"count\": " << dets.size() << ",\n";
    f << "  \"backend\": \"" << esc(backend) << "\",\n";
    f << "  \"strategy\": \"" << esc(strategy) << "\",\n";
    f << "  \"detections\": [";
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        f << (i ? ",\n" : "\n");
        f << "    {\n";
        f << "      \"score\": " << fmt_num(d.score) << ",\n";
        f << "      \"polygon\": [\n";
        for (int k = 0; k < 4; ++k) {
            f << "        [\n          " << fmt_num(d.polygon[k][0]) << ",\n          "
              << fmt_num(d.polygon[k][1]) << "\n        ]" << (k < 3 ? ",\n" : "\n");
        }
        f << "      ],\n";
        f << "      \"xyxy\": [\n        " << fmt_num(d.xyxy[0]) << ",\n        "
          << fmt_num(d.xyxy[1]) << ",\n        " << fmt_num(d.xyxy[2]) << ",\n        "
          << fmt_num(d.xyxy[3]) << "\n      ]\n";
        f << "    }";
    }
    f << (dets.empty() ? "" : "\n  ") << "]\n";
    f << "}\n";
}

}  // namespace aerial
