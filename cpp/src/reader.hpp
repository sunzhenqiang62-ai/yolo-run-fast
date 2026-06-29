// Full-image-into-RAM tile reader, mirroring CachedTileReader in
// predict_aerial.py: decode the whole image once, then slice ROIs and pad to
// tile_size with a (114,114,114) border.
#pragma once

#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "tiling.hpp"

namespace aerial {

inline cv::Mat pad_tile(const cv::Mat& bgr, const TileSpec& spec, int tile_size) {
    int pad_h = tile_size - spec.th;
    int pad_w = tile_size - spec.tw;
    if (pad_h > 0 || pad_w > 0) {
        cv::Mat out;
        cv::copyMakeBorder(bgr, out, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT,
                           cv::Scalar(114, 114, 114));
        return out;
    }
    return bgr;
}

class CachedReader {
public:
    explicit CachedReader(const std::string& path) {
        // cv::imread decodes to BGR directly (Python read RGB via PIL then
        // converted to BGR — same result).
        bgr_ = cv::imread(path, cv::IMREAD_COLOR);
        if (bgr_.empty()) {
            throw std::runtime_error("failed to read image: " + path);
        }
    }

    int width() const { return bgr_.cols; }
    int height() const { return bgr_.rows; }

    // Returns a padded tile_size x tile_size BGR tile (CV_8UC3).
    cv::Mat read_bgr(const TileSpec& spec, int tile_size) const {
        cv::Rect roi(spec.x0, spec.y0, spec.tw, spec.th);
        cv::Mat tile = bgr_(roi).clone();
        return pad_tile(tile, spec, tile_size);
    }

private:
    cv::Mat bgr_;
};

}  // namespace aerial
