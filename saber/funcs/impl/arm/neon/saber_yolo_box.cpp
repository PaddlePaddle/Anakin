
#include "saber/funcs/impl/arm/saber_yolo_box.h"
#include <cmath>
namespace anakin {
namespace saber {
namespace {
inline float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

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

inline int get_entry_index(int batch, int an_idx, int hw_idx,
        int an_num, int an_stride, int stride,
        int entry) {
    return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

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

inline void calc_label_score(float* scores, const float* input,
        const int label_idx, const int score_idx,
        const int class_num, const float conf,
        const int stride) {
    for (int i = 0; i < class_num; i++) {
        scores[score_idx + i] = conf * sigmoid(input[label_idx + i * stride]);
    }
}
}

template <>
SaberStatus SaberYoloBox<ARM, AK_FLOAT>::create(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        YoloBoxParam<ARM>& param, Context<ARM>& ctx) {

    return SaberSuccess;
}

template <>
SaberStatus SaberYoloBox<ARM, AK_FLOAT>::init(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        YoloBoxParam<ARM>& param, Context<ARM>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberYoloBox<ARM, AK_FLOAT>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        YoloBoxParam<ARM>& param) {
#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    auto* input = inputs[0];
    auto* imgsize = inputs[1];
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

    const int stride = h * w;
    const int an_stride = (class_num + 5) * stride;

    auto anchors_data = anchors.data();

    const float* input_data = (const float*)input->data();
    const int* imgsize_data = (const int*)imgsize->data();

    float* boxes_data = (float*)boxes->mutable_data();
//    memset(boxes_data, 0, boxes->numel() * sizeof(float));

    float* scores_data = (float*)scores->mutable_data();
//    memset(scores_data, 0, scores->numel() * sizeof(float));

    float box[4];
    for (int i = 0; i < n; i++) {
        int img_height = imgsize_data[2 * i];
        int img_width = imgsize_data[2 * i + 1];

        for (int j = 0; j < an_num; j++) {
            for (int k = 0; k < h; k++) {
                for (int l = 0; l < w; l++) {
                    int obj_idx =
                            get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 4);
                    float conf = sigmoid(input_data[obj_idx]);
                    if (conf < conf_thresh) {
                        continue;
                    }

                    int box_idx =
                            get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 0);
                    get_yolo_box(box, input_data, anchors_data, l, k, j, h, input_size,
                            box_idx, stride, img_height, img_width);
                    box_idx = (i * box_num + j * stride + k * w + l) * 4;
                    calc_detection_box(boxes_data, box, box_idx, img_height,
                            img_width);

                    int label_idx =
                            get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 5);
                    int score_idx = (i * box_num + j * stride + k * w + l) * class_num;
                    calc_label_score(scores_data, input_data, label_idx, score_idx,
                            class_num, conf, stride);
                }
            }
        }
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "YoloBox : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("YoloBox", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}

// template class SaberYoloBox<ARM, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberYoloBox, YoloBoxParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberYoloBox, YoloBoxParam, ARM, AK_INT8);

} // namespace saber.
} // namespace anakin.
