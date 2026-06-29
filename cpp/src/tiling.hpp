// Sliding-window tile grid + two-stage hot-region logic.
// Ported 1:1 from predict_aerial.py (build_tile_grid, _expand_rect,
// _rects_overlap, _tile_intersects_rect, build_hot_regions,
// filter_tiles_by_regions).
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace aerial {

struct TileSpec {
    int x0;
    int y0;
    int tw;
    int th;
};

struct Rect {
    int x1;
    int y1;
    int x2;
    int y2;
};

// Mirrors build_tile_grid(): step = stride if given else max(1, tile-overlap).
// Always emits the right/bottom edge tile (clamped) before stopping.
inline std::vector<TileSpec> build_tile_grid(int orig_w, int orig_h, int tile_size,
                                             int overlap, int stride = -1) {
    int step = (stride >= 0) ? stride : std::max(1, tile_size - overlap);
    std::vector<TileSpec> tiles;
    int y0 = 0;
    while (y0 < orig_h) {
        int x0 = 0;
        int th = std::min(tile_size, orig_h - y0);
        while (x0 < orig_w) {
            int tw = std::min(tile_size, orig_w - x0);
            tiles.push_back(TileSpec{x0, y0, tw, th});
            if (x0 + tw >= orig_w) break;
            x0 += step;
        }
        if (y0 + th >= orig_h) break;
        y0 += step;
    }
    return tiles;
}

inline Rect expand_rect(double x1, double y1, double x2, double y2, int margin,
                        int orig_w, int orig_h) {
    return Rect{
        std::max(0, static_cast<int>(x1) - margin),
        std::max(0, static_cast<int>(y1) - margin),
        std::min(orig_w, static_cast<int>(x2) + margin),
        std::min(orig_h, static_cast<int>(y2) + margin),
    };
}

inline bool rects_overlap(const Rect& a, const Rect& b) {
    return !(a.x2 <= b.x1 || b.x2 <= a.x1 || a.y2 <= b.y1 || b.y2 <= a.y1);
}

inline bool tile_intersects_rect(const TileSpec& s, const Rect& r) {
    int tx2 = s.x0 + s.tw;
    int ty2 = s.y0 + s.th;
    return !(tx2 <= r.x1 || r.x2 <= s.x0 || ty2 <= r.y1 || r.y2 <= s.y0);
}

// build_hot_regions(): expand each coarse hit by margin, then iteratively merge
// overlapping rects until stable. Empty input -> single full-image region.
inline std::vector<Rect> build_hot_regions(const std::vector<Rect>& hit_boxes,
                                           int orig_w, int orig_h, int margin) {
    if (hit_boxes.empty()) {
        return {Rect{0, 0, orig_w, orig_h}};
    }
    std::vector<Rect> rects;
    rects.reserve(hit_boxes.size());
    for (const auto& b : hit_boxes) {
        rects.push_back(expand_rect(b.x1, b.y1, b.x2, b.y2, margin, orig_w, orig_h));
    }

    bool merged = true;
    while (merged) {
        merged = false;
        std::vector<Rect> new_rects;
        std::vector<char> used(rects.size(), 0);
        for (size_t i = 0; i < rects.size(); ++i) {
            if (used[i]) continue;
            Rect a = rects[i];
            for (size_t j = i + 1; j < rects.size(); ++j) {
                if (used[j]) continue;
                if (rects_overlap(a, rects[j])) {
                    a.x1 = std::min(a.x1, rects[j].x1);
                    a.y1 = std::min(a.y1, rects[j].y1);
                    a.x2 = std::max(a.x2, rects[j].x2);
                    a.y2 = std::max(a.y2, rects[j].y2);
                    used[j] = 1;
                    merged = true;
                }
            }
            new_rects.push_back(a);
            used[i] = 1;
        }
        rects.swap(new_rects);
    }
    return rects;
}

inline std::vector<TileSpec> filter_tiles_by_regions(const std::vector<TileSpec>& tiles,
                                                     const std::vector<Rect>& regions) {
    if (regions.empty()) return tiles;
    std::vector<TileSpec> out;
    for (const auto& t : tiles) {
        for (const auto& r : regions) {
            if (tile_intersects_rect(t, r)) {
                out.push_back(t);
                break;
            }
        }
    }
    return out;
}

}  // namespace aerial
