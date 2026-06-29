// Axis-aligned greedy NMS, ported from predict_aerial.py _nms_xyxy.
#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace aerial {

// boxes: flat [N][4] = x1,y1,x2,y2. Returns kept indices, highest score first.
inline std::vector<int> nms_xyxy(const std::vector<std::array<float, 4>>& boxes,
                                 const std::vector<float>& scores, float iou) {
    const size_t n = boxes.size();
    if (n == 0) return {};

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    // descending by score (stable to mirror numpy argsort()[::-1] ordering closely)
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return scores[a] > scores[b]; });

    std::vector<char> suppressed(n, 0);
    std::vector<int> keep;
    keep.reserve(n);

    for (size_t oi = 0; oi < n; ++oi) {
        int i = order[oi];
        if (suppressed[i]) continue;
        keep.push_back(i);
        const float ix1 = boxes[i][0], iy1 = boxes[i][1], ix2 = boxes[i][2], iy2 = boxes[i][3];
        const float area_i = (ix2 - ix1) * (iy2 - iy1);
        for (size_t oj = oi + 1; oj < n; ++oj) {
            int j = order[oj];
            if (suppressed[j]) continue;
            const float xx1 = std::max(ix1, boxes[j][0]);
            const float yy1 = std::max(iy1, boxes[j][1]);
            const float xx2 = std::min(ix2, boxes[j][2]);
            const float yy2 = std::min(iy2, boxes[j][3]);
            const float w = std::max(0.0f, xx2 - xx1);
            const float h = std::max(0.0f, yy2 - yy1);
            const float inter = w * h;
            const float area_j =
                (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]);
            const float uni = area_i + area_j - inter;
            const float ov = uni > 0.0f ? inter / uni : 0.0f;
            if (ov > iou) suppressed[j] = 1;
        }
    }
    return keep;
}

}  // namespace aerial
