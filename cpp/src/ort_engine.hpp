// ONNX Runtime wrapper for the fixed-shape, embedded-NMS OBB model.
//   input  "images"  : float32 OR float16 [B, 3, 640, 640]  (NCHW, RGB, /255)
//   output "output0" : float32 [B, 300, 7]  (cx,cy,w,h,conf,cls,angle)
// Batch size B, input edge, and input dtype are read from the model at load
// time, so one binary handles both the fp32/batch-32 and fp16/batch-64 exports.
// The fp16 export keeps the embedded-NMS output in fp32, so the decode path is
// dtype-agnostic. Provider chosen at construction (cpu | cuda, cuda falls back
// to cpu on init failure).
#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "onnxruntime_cxx_api.h"

namespace aerial {

class OrtEngine {
public:
    static constexpr int kRows = 300;
    static constexpr int kAttrs = 7;

    // device: "cpu" or a CUDA device index string ("0", "1", ...).
    OrtEngine(const std::string& model_path, const std::string& device, int intra_threads)
        : env_(ORT_LOGGING_LEVEL_WARNING, "aerial") {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        use_cuda_ = (device != "cpu" && !device.empty());
        if (use_cuda_) {
            try {
                OrtCUDAProviderOptions cuda_opts{};
                cuda_opts.device_id = std::stoi(device);
                opts.AppendExecutionProvider_CUDA(cuda_opts);
                provider_ = "cuda";
            } catch (const std::exception& e) {
                std::cerr << "CUDA provider init failed (" << e.what()
                          << "); falling back to CPU." << std::endl;
                use_cuda_ = false;
            }
        }
        if (!use_cuda_) {
            if (intra_threads > 0) opts.SetIntraOpNumThreads(intra_threads);
            provider_ = "cpu";
        }

        session_ = Ort::Session(env_, model_path.c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;
        in_name_ = session_.GetInputNameAllocated(0, alloc).get();
        out_name_ = session_.GetOutputNameAllocated(0, alloc).get();

        // Read batch / edge / input dtype from the model's input tensor.
        // Keep the TypeInfo alive: GetTensorTypeAndShapeInfo() returns a
        // non-owning view into it, so a temporary would dangle.
        Ort::TypeInfo type_info = session_.GetInputTypeInfo(0);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = info.GetShape();
        if (shape.size() != 4)
            throw std::runtime_error("expected 4-D model input [B,3,H,W]");
        batch_ = shape[0] > 0 ? static_cast<int>(shape[0]) : 32;  // fixed-batch export
        edge_ = shape[2] > 0 ? static_cast<int>(shape[2]) : 640;
        in_fp16_ = info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;

        const size_t count = static_cast<size_t>(batch_) * 3 * edge_ * edge_;
        if (in_fp16_)
            input_fp16_.assign(count, Ort::Float16_t(0.0f));
        else
            input_f32_.assign(count, 0.0f);
    }

    const std::string& provider() const { return provider_; }
    int batch() const { return batch_; }
    int edge() const { return edge_; }
    bool fp16() const { return in_fp16_; }

    // Fill the NCHW input buffer from up to batch() BGR tiles, normalizing to
    // RGB/255 and writing the model's native dtype directly (no extra pass).
    // Unused batch slots are left zeroed. tiles must be edge() x edge() CV_8UC3.
    void set_batch(const std::vector<cv::Mat>& tiles) {
        if (in_fp16_)
            fill_planar(tiles, input_fp16_, [](float v) { return Ort::Float16_t(v); });
        else
            fill_planar(tiles, input_f32_, [](float v) { return v; });
    }

    // Run inference on the current input buffer. The output is always fp32
    // [batch, kRows, kAttrs]; the returned Ort::Value owns it.
    Ort::Value run() {
        const std::array<int64_t, 4> in_shape{batch_, 3, edge_, edge_};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value in_tensor{nullptr};
        if (in_fp16_)
            in_tensor = Ort::Value::CreateTensor(mem, input_fp16_.data(),
                                                 input_fp16_.size() * sizeof(Ort::Float16_t),
                                                 in_shape.data(), in_shape.size(),
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        else
            in_tensor = Ort::Value::CreateTensor<float>(mem, input_f32_.data(),
                                                        input_f32_.size(), in_shape.data(),
                                                        in_shape.size());

        const char* in_names[] = {in_name_.c_str()};
        const char* out_names[] = {out_name_.c_str()};
        auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1,
                                    out_names, 1);
        return std::move(outputs[0]);
    }

private:
    // Normalize BGR->RGB/255 into a planar NCHW buffer of element type T,
    // applying `cvt` (identity for fp32, float->Float16_t for fp16).
    template <typename T, typename Cvt>
    void fill_planar(const std::vector<cv::Mat>& tiles, std::vector<T>& buf, Cvt cvt) {
        const size_t plane = static_cast<size_t>(edge_) * edge_;
        const size_t img_sz = plane * 3;
        size_t n = std::min(tiles.size(), static_cast<size_t>(batch_));
        if (n < static_cast<size_t>(batch_))  // zero the trailing unused slots
            std::fill(buf.begin() + n * img_sz, buf.end(), cvt(0.0f));
        for (size_t b = 0; b < n; ++b) {
            const cv::Mat& m = tiles[b];
            T* rch = buf.data() + b * img_sz;  // R plane
            T* gch = rch + plane;              // G plane
            T* bch = gch + plane;              // B plane
            for (int y = 0; y < edge_; ++y) {
                const uchar* row = m.ptr<uchar>(y);  // BGR interleaved
                size_t off = static_cast<size_t>(y) * edge_;
                for (int x = 0; x < edge_; ++x) {
                    const uchar* px = row + x * 3;
                    rch[off + x] = cvt(px[2] * (1.0f / 255.0f));  // R
                    gch[off + x] = cvt(px[1] * (1.0f / 255.0f));  // G
                    bch[off + x] = cvt(px[0] * (1.0f / 255.0f));  // B
                }
            }
        }
    }

    Ort::Env env_;
    Ort::Session session_{nullptr};
    std::string in_name_;
    std::string out_name_;
    std::vector<float> input_f32_;            // used when in_fp16_ == false
    std::vector<Ort::Float16_t> input_fp16_;  // used when in_fp16_ == true
    int batch_ = 32;
    int edge_ = 640;
    bool in_fp16_ = false;
    bool use_cuda_ = false;
    std::string provider_ = "cpu";
};

}  // namespace aerial
