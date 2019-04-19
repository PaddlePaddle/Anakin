
#include "saber/core/context.h"
#include "saber/funcs/yolo_box.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>

using namespace anakin::saber;

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

template <typename dtype, typename TargetType_D, typename TargetType_H>
void yolo_box_cpu(const std::vector<Tensor<TargetType_H>*>& input,
        std::vector<Tensor<TargetType_H>*>& output,\
        YoloBoxParam<TargetType_D>& param) {

    auto* in = input[0];
    auto* imgsize = input[1];
    auto* boxes = output[0];
    auto* scores = output[1];
    auto anchors = param.anchors;
    int class_num = param.class_num;
    float conf_thresh = param.conf_thresh;
    int downsample_ratio = param.downsample_ratio;

    const int n = in->num();
    const int h = in->height();
    const int w = in->width();
    const int box_num = boxes->valid_shape()[1];
    const int an_num = anchors.size() / 2;
    int input_size = downsample_ratio * h;

    const int stride = h * w;
    const int an_stride = (class_num + 5) * stride;

    auto anchors_data = anchors.data();

    const float* input_data = (const float*)in->data();
    const float* imgsize_data = (const float*)imgsize->data();

    float* boxes_data = (float*)boxes->mutable_data();
    float* scores_data = (float*)scores->mutable_data();

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
}

template <typename TargetType_D, typename TargetType_H>
void test_yolo() {
    //Init the test_base
    TestSaberBase<TargetType_D, TargetType_H, AK_FLOAT, YoloBox, YoloBoxParam> testbase(2, 2);
    YoloBoxParam<TargetType_D> param({1, 2, 3, 4}, 5, 0.5, 5);
    for (int w_in : {16, 20, 32, 64}) {
        for (int h_in : {16, 20, 32, 64}) {
            for (int ch_in : {20}) {
                for (int num_in:{1, 3, 5}) {
                    Shape shape0({num_in, ch_in, h_in, w_in});
                    Shape shape1({num_in, 2, 4}, Layout_NHW);

                    Tensor<TargetType_D> input0;
                    Tensor<TargetType_D> input1;

                    testbase.set_param(param);

                    input0.re_alloc(shape0, AK_FLOAT);
                    input1.re_alloc(shape1, AK_FLOAT);

                    std::vector<Tensor<TargetType_D>*> ins{&input0, &input1};
                    fill_tensor_rand(input0, -10, 10);
                    fill_tensor_rand(input1, -10, 10);
                    testbase.add_custom_input(ins);
                    testbase.run_test(yolo_box_cpu<float, TargetType_D, TargetType_H>);
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_yolo_box) {

#ifdef USE_CUDA
    test_yolo<NV, NVHX86>();
#endif

#ifdef USE_X86_PLACE
    test_yolo<X86, X86>();
#endif

#ifdef USE_ARM_PLACE
    test_yolo<ARM, ARM>();
#endif

}
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
