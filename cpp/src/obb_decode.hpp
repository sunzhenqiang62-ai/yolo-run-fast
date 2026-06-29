// Decode the embedded-NMS ONNX output [300,7] = cx,cy,w,h,conf,cls,angle(rad)
// into clipped, tile-offset polygons + axis-aligned xyxy.
// Ported from predict_aerial.py _extract_hits (which consumed
// result.obb.xyxyxyxy / conf). Here we build the polygon ourselves from the
// rotated-box parameters via cv::RotatedRect.
#pragma once

#include <array>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

#include "tiling.hpp"

namespace aerial {

constexpr double kPi = 3.14159265358979323846;

struct RawHit {
    std::array<std::array<double, 2>, 4> polygon;  // 4 (x,y) corners
    double score;
    std::array<double, 4> xyxy;  // x1,y1,x2,y2
};

// One ONNX output row: 7 floats. tile_size is the model input edge (640).
// Rows below conf_thr (incl. the ~0-conf NMS padding rows) are skipped.
inline void decode_tile_output(const float* rows, int num_rows, const TileSpec& spec,
                               float conf_thr, std::vector<RawHit>& out) {
    constexpr int STRIDE = 7;
    for (int r = 0; r < num_rows; ++r) {
        const float* p = rows + r * STRIDE;
        const float conf = p[4];
        if (conf <= conf_thr) continue;

        const float cx = p[0];
        const float cy = p[1];
        const float w = p[2];
        const float h = p[3];
        const float angle_deg = p[6] * static_cast<float>(180.0 / kPi);

        cv::Point2f pts[4];
        cv::RotatedRect(cv::Point2f(cx, cy), cv::Size2f(w, h), angle_deg).points(pts);

        RawHit hit;
        double xmin = 1e18, ymin = 1e18, xmax = -1e18, ymax = -1e18;
        for (int k = 0; k < 4; ++k) {
            // clip to the *unpadded* tile extent, then offset to global coords
            double x = std::min(std::max(static_cast<double>(pts[k].x), 0.0),
                                static_cast<double>(spec.tw));
            double y = std::min(std::max(static_cast<double>(pts[k].y), 0.0),
                                static_cast<double>(spec.th));
            x += spec.x0;
            y += spec.y0;
            hit.polygon[k] = {x, y};
            xmin = std::min(xmin, x);
            ymin = std::min(ymin, y);
            xmax = std::max(xmax, x);
            ymax = std::max(ymax, y);
        }
        hit.score = conf;
        hit.xyxy = {xmin, ymin, xmax, ymax};
        out.push_back(std::move(hit));
    }
}

}  // namespace aerial
