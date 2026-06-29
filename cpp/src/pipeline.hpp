// End-to-end orchestration mirroring predict_aerial.predict_aerial():
//   read image -> (two-stage coarse scan -> hot regions -> fine scan | full
//   scan) -> cross-tile NMS merge -> JSON -> optional preview.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <future>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "json_writer.hpp"
#include "nms.hpp"
#include "obb_decode.hpp"
#include "ort_engine.hpp"
#include "reader.hpp"
#include "tiling.hpp"

namespace aerial {

struct ProfileStats {
    double read_ms = 0;
    double infer_ms = 0;
    double merge_ms = 0;
    double preview_ms = 0;
    double cache_ms = 0;
    int coarse_tiles = 0;
    int fine_tiles = 0;
    int total_tiles = 0;

    void report(double total_s) const {
        std::printf(
            "Profile: total=%.1fs | cache=%.0fms read=%.0fms infer=%.0fms "
            "merge=%.0fms preview=%.0fms | tiles coarse=%d fine=%d total=%d\n",
            total_s, cache_ms, read_ms, infer_ms, merge_ms, preview_ms,
            coarse_tiles, fine_tiles, total_tiles);
    }
};

struct Config {
    std::string model_path;
    std::string image_path;
    std::string json_path;
    std::string preview_path;  // empty => no preview
    int tile_size = 640;
    int overlap = 96;
    float conf = 0.25f;
    std::string device = "cpu";
    std::string strategy = "two-stage";
    int coarse_stride = 1280;
    float coarse_conf = 0.35f;
    int hot_margin = 320;
    float nms_iou = 0.35f;
    int preview_max_edge = 4096;
    bool prefetch = true;
    bool profile = false;
    bool skip_preview = true;
    int intra_threads = 0;  // 0 => ORT default (CPU only)
};

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Batched inference over a tile list with optional next-batch prefetch.
inline std::vector<RawHit> predict_tiles(OrtEngine& engine, const CachedReader& reader,
                                         const std::vector<TileSpec>& tiles, float conf,
                                         int tile_size, ProfileStats& stats,
                                         bool prefetch, const std::string& label) {
    std::vector<RawHit> hits;
    const size_t total = tiles.size();
    if (total == 0) return hits;
    const int B = engine.batch();

    auto load = [&](size_t start, size_t end) {
        std::vector<cv::Mat> imgs;
        imgs.reserve(end - start);
        for (size_t i = start; i < end; ++i)
            imgs.push_back(reader.read_bgr(tiles[i], tile_size));
        return imgs;
    };

    auto t_read = Clock::now();
    std::vector<cv::Mat> cur = load(0, std::min<size_t>(B, total));
    stats.read_ms += ms_since(t_read);

    std::future<std::vector<cv::Mat>> fut;
    size_t start = 0;
    size_t processed = 0;
    while (start < total) {
        size_t end = std::min<size_t>(start + B, total);
        size_t nstart = end, nend = std::min<size_t>(end + B, total);
        bool has_next = nstart < total;
        if (has_next && prefetch)
            fut = std::async(std::launch::async, load, nstart, nend);

        auto t_inf = Clock::now();
        engine.set_batch(cur);
        Ort::Value out = engine.run();
        const float* data = out.GetTensorData<float>();
        stats.infer_ms += ms_since(t_inf);

        for (size_t i = start; i < end; ++i) {
            const float* rows = data + (i - start) * OrtEngine::kRows * OrtEngine::kAttrs;
            decode_tile_output(rows, OrtEngine::kRows, tiles[i], conf, hits);
        }

        start = end;
        processed = end;
        if (has_next) {
            auto t_r = Clock::now();
            cur = prefetch ? fut.get() : load(nstart, nend);
            stats.read_ms += ms_since(t_r);
        }
    }
    std::printf("  %s [%zu/%zu] hits=%zu\n", label.c_str(), processed, total, hits.size());
    return hits;
}

inline std::vector<RawHit> merge_hits(const std::vector<RawHit>& hits, float nms_iou) {
    if (hits.empty()) return {};
    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    boxes.reserve(hits.size());
    scores.reserve(hits.size());
    for (const auto& h : hits) {
        boxes.push_back({static_cast<float>(h.xyxy[0]), static_cast<float>(h.xyxy[1]),
                         static_cast<float>(h.xyxy[2]), static_cast<float>(h.xyxy[3])});
        scores.push_back(static_cast<float>(h.score));
    }
    std::vector<int> keep = nms_xyxy(boxes, scores, nms_iou);
    std::vector<RawHit> out;
    out.reserve(keep.size());
    for (int i : keep) out.push_back(hits[i]);
    std::stable_sort(out.begin(), out.end(),
                     [](const RawHit& a, const RawHit& b) { return a.score > b.score; });
    return out;
}

inline void save_preview(const std::string& image_path, const std::vector<RawHit>& dets,
                         const std::string& preview_path, int max_edge) {
    cv::Mat full = cv::imread(image_path, cv::IMREAD_COLOR);
    if (full.empty()) return;
    double scale = std::min(1.0, static_cast<double>(max_edge) /
                                     std::max(full.cols, full.rows));
    cv::Mat preview;
    if (scale < 1.0)
        cv::resize(full, preview, cv::Size(), scale, scale, cv::INTER_AREA);
    else
        preview = full;
    double s = static_cast<double>(preview.cols) / full.cols;

    for (const auto& d : dets) {
        std::vector<cv::Point> pts(4);
        for (int k = 0; k < 4; ++k)
            pts[k] = cv::Point(static_cast<int>(d.polygon[k][0] * s),
                               static_cast<int>(d.polygon[k][1] * s));
        cv::polylines(preview, pts, true, cv::Scalar(0, 255, 0), 2);
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.2f", d.score);
        cv::putText(preview, buf, cv::Point(pts[0].x, std::max(0, pts[0].y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    cv::imwrite(preview_path, preview, {cv::IMWRITE_JPEG_QUALITY, 85});
}

inline int run_pipeline(const Config& cfg) {
    auto t_total = Clock::now();
    ProfileStats stats;

    auto t_cache = Clock::now();
    CachedReader reader(cfg.image_path);
    stats.cache_ms = ms_since(t_cache);
    int orig_w = reader.width();
    int orig_h = reader.height();

    OrtEngine engine(cfg.model_path, cfg.device, cfg.intra_threads);
    std::string backend = std::string("onnxruntime-") + engine.provider();
    std::printf("Backend: %s | batch=%d | dtype=%s | prefetch=%d\n", backend.c_str(),
                engine.batch(), engine.fp16() ? "fp16" : "fp32", cfg.prefetch ? 1 : 0);

    std::vector<TileSpec> fine_tiles =
        build_tile_grid(orig_w, orig_h, cfg.tile_size, cfg.overlap);

    std::vector<RawHit> all_hits;
    if (cfg.strategy == "two-stage") {
        std::vector<TileSpec> coarse_tiles =
            build_tile_grid(orig_w, orig_h, cfg.tile_size, 0, cfg.coarse_stride);
        stats.coarse_tiles = static_cast<int>(coarse_tiles.size());
        std::printf("Image: %dx%d | two-stage | coarse=%zu (stride=%d) fine_pool=%zu\n",
                    orig_w, orig_h, coarse_tiles.size(), cfg.coarse_stride,
                    fine_tiles.size());

        std::printf("Stage 1: coarse scan\n");
        std::vector<RawHit> coarse_hits = predict_tiles(
            engine, reader, coarse_tiles, cfg.coarse_conf, cfg.tile_size, stats,
            cfg.prefetch, "coarse");

        std::vector<Rect> hit_boxes;
        hit_boxes.reserve(coarse_hits.size());
        for (const auto& h : coarse_hits)
            hit_boxes.push_back(Rect{static_cast<int>(h.xyxy[0]), static_cast<int>(h.xyxy[1]),
                                     static_cast<int>(h.xyxy[2]), static_cast<int>(h.xyxy[3])});
        std::vector<Rect> regions =
            build_hot_regions(hit_boxes, orig_w, orig_h, cfg.hot_margin);
        std::printf("  hot regions: %zu (from %zu coarse hits)\n", regions.size(),
                    coarse_hits.size());

        std::vector<TileSpec> selected_fine = filter_tiles_by_regions(fine_tiles, regions);
        stats.fine_tiles = static_cast<int>(selected_fine.size());
        stats.total_tiles = stats.coarse_tiles + stats.fine_tiles;
        std::printf("Stage 2: fine scan (%zu tiles)\n", selected_fine.size());
        all_hits = predict_tiles(engine, reader, selected_fine, cfg.conf, cfg.tile_size,
                                 stats, cfg.prefetch, "fine");
        all_hits.insert(all_hits.end(), coarse_hits.begin(), coarse_hits.end());
    } else {
        stats.fine_tiles = static_cast<int>(fine_tiles.size());
        stats.total_tiles = stats.fine_tiles;
        std::printf("Image: %dx%d | full-scan | tiles=%zu (%dpx, overlap %dpx)\n", orig_w,
                    orig_h, fine_tiles.size(), cfg.tile_size, cfg.overlap);
        all_hits = predict_tiles(engine, reader, fine_tiles, cfg.conf, cfg.tile_size, stats,
                                 cfg.prefetch, "scan");
    }

    auto t_merge = Clock::now();
    std::vector<RawHit> dets = merge_hits(all_hits, cfg.nms_iou);
    stats.merge_ms = ms_since(t_merge);

    write_json(cfg.json_path, cfg.image_path, orig_w, orig_h, backend, cfg.strategy, dets);
    std::printf("Found %zu detection(s) after merge\n", dets.size());
    std::printf("JSON: %s\n", cfg.json_path.c_str());

    if (!cfg.skip_preview && !cfg.preview_path.empty()) {
        auto t_prev = Clock::now();
        save_preview(cfg.image_path, dets, cfg.preview_path, cfg.preview_max_edge);
        stats.preview_ms = ms_since(t_prev);
        std::printf("Preview: %s\n", cfg.preview_path.c_str());
    }

    double total_s = ms_since(t_total) / 1000.0;
    if (cfg.profile)
        stats.report(total_s);
    else
        std::printf("Total: %.1fs\n", total_s);
    return static_cast<int>(dets.size());
}

}  // namespace aerial
