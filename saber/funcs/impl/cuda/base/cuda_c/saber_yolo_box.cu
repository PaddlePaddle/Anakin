
#include "saber/funcs/impl/cuda/saber_yolo_box.h"

namespace anakin {
namespace saber {

namespace {
__device__
inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}
__device__
inline void get_yolo_box(float* box, const float* x, const int* anchors, int i,
                       int j, int an_idx, int grid_size,
                       int input_size, int index, int stride,
                       int img_height, int img_width) {

    box[0] = (i + sigmoid(x[index])) * img_width / grid_size;
    box[1] = (j + sigmoid(x[index + stride])) * img_height / grid_size;
    box[2] = std::exp(x[index + 2 * stride]) * anchors[2 * an_idx] * img_width /
             input_size;
    box[3] = std::exp(x[index + 3 * stride]) * anchors[2 * an_idx + 1] *
             img_height / input_size;
}
__device__
inline int get_entry_index(int batch, int an_idx, int hw_idx,
                         int an_num, int an_stride, int stride,
                         int entry) {
    return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}
__device__
inline void calc_detection_box(float* boxes, float* box, const int box_idx,
                             const int img_height,
                             const int img_width) {

    boxes[box_idx] = box[0] - box[2] / 2;
    boxes[box_idx + 1] = box[1] - box[3] / 2;
    boxes[box_idx + 2] = box[0] + box[2] / 2;
    boxes[box_idx + 3] = box[1] + box[3] / 2;

    boxes[box_idx] = boxes[box_idx] > 0 ? boxes[box_idx] : static_cast<float>(0);
    boxes[box_idx + 1] =
            boxes[box_idx + 1] > 0 ? boxes[box_idx + 1] : static_cast<float>(0);
    boxes[box_idx + 2] = boxes[box_idx + 2] < img_width - 1
                         ? boxes[box_idx + 2]
                         : static_cast<float>(img_width - 1);
    boxes[box_idx + 3] = boxes[box_idx + 3] < img_height - 1
                         ? boxes[box_idx + 3]
                         : static_cast<float>(img_height - 1);
}
__device__
inline void calc_label_score(float* scores, const float* input,
                           const int label_idx, const int score_idx,
                           const int class_num, const float conf,
                           const int stride) {
    for (int i = 0; i < class_num; i++) {
        scores[score_idx + i] = conf * sigmoid(input[label_idx + i * stride]);
    }
}
}

__global__ void ker_yolo_box(const float* input, const float* imgsize, float* boxes,
                            float* scores, const float conf_thresh,
                            const int* anchors, const int n, const int h,
                            const int w, const int an_num, const int class_num,
                            const int box_num, int input_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float box[4];
    for (; tid < n * box_num; tid += stride) {
        int grid_num = h * w;
        int i = tid / box_num;
        int j = (tid % box_num) / grid_num;
        int k = (tid % grid_num) / w;
        int l = tid % w;

        int an_stride = (5 + class_num) * grid_num;
        int img_height = imgsize[2 * i];
        int img_width = imgsize[2 * i + 1];

        int obj_idx =
                get_entry_index(i, j, k * w + l, an_num, an_stride, grid_num, 4);
        float conf = sigmoid(input[obj_idx]);
        if (conf < conf_thresh) {
            continue;
        }

        int box_idx =
                get_entry_index(i, j, k * w + l, an_num, an_stride, grid_num, 0);
        get_yolo_box(box, input, anchors, l, k, j, h, input_size, box_idx,
                      grid_num, img_height, img_width);
        box_idx = (i * box_num + j * grid_num + k * w + l) * 4;
        calc_detection_box(boxes, box, box_idx, img_height, img_width);

        int label_idx =
                get_entry_index(i, j, k * w + l, an_num, an_stride, grid_num, 5);
        int score_idx = (i * box_num + j * grid_num + k * w + l) * class_num;
        calc_label_score(scores, input, label_idx, score_idx, class_num, conf,
                          grid_num);
    }
}

template <>
SaberStatus SaberYoloBox<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        YoloBoxParam<NV>& param, Context<NV>& ctx) {

    return SaberSuccess;
}

template <>
SaberStatus SaberYoloBox<NV, AK_FLOAT>::init(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        YoloBoxParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberYoloBox<NV, AK_FLOAT>::dispatch(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        YoloBoxParam<NV>& param) {

    auto* input = inputs[0];
    auto* img_size = inputs[1];
    auto* boxes = outputs[0];
    auto* scores = outputs[1];

    auto anchors = param.anchors;
    int class_num = param.class_num;
    float conf_thresh = param.conf_thresh;
    int downsample_ratio = param.downsample_ratio;

    const int n = input->num();
    const int h = input->height();
    const int w = input->width();
    const int box_num = boxes->valid_shape()[1];
    const int an_num = anchors.size() / 2;
    int input_size = downsample_ratio * h;

    Buffer<NV> _anchors_buf;
    _anchors_buf.re_alloc(sizeof(int) * anchors.size());

    cudaMemcpyAsync(_anchors_buf.get_data_mutable(), anchors.data(),
            sizeof(int) * anchors.size(), cudaMemcpyHostToDevice, _ctx->get_compute_stream());

    const float* input_data = (const float*)input->data();
    const float* imgsize_data = (const float*)img_size->data();
    float* boxes_data = (float*)boxes->mutable_data();
    float* scores_data =(float*)scores->mutable_data();

    int grid_dim = (n * box_num + 512 - 1) / 512;
    grid_dim = grid_dim > 8 ? 8 : grid_dim;

    ker_yolo_box<<<grid_dim, 512, 0, _ctx->get_compute_stream()>>>(
            input_data, imgsize_data, boxes_data, scores_data, conf_thresh,
            (const int*)_anchors_buf.get_data(), n, h, w, an_num, class_num, box_num, input_size);

    return SaberSuccess;
}

template class SaberYoloBox<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberYoloBox, YoloBoxParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberYoloBox, YoloBoxParam, NV, AK_INT8);

} // namespace saber.
} // namespace anakin.
