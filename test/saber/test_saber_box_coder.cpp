#include "core/context.h"
#include "funcs/box_coder.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;
enum BOX_CODER_VAR {
    FIX_SIZE_VAR = 0,
    NO_VAR = 1,
    FROM_INPUT_VAR = 2
};
template <typename Dtype, BOX_CODER_VAR fix_size_var, typename TargetType_D, typename TargetType_H>
static inline void box_coder(Tensor<TargetType_H>* proposals,
                             const Tensor<TargetType_H>* anchors,
                             const Tensor<TargetType_H>* bbox_deltas,
                             const Tensor<TargetType_H>* variances,
                             BoxCoderParam<TargetType_D>& param
                            ) {
    const size_t row = bbox_deltas->num();
    const size_t col = bbox_deltas->channel();
    const size_t anchor_nums = row * col;
    const size_t anchor_len = anchors->valid_shape()[1];
    CHECK_EQ(anchor_len, 5) << "anchor length is 5";
    int out_len = 4;
    int var_len = 4;
    int delta_len = 4;
    const Dtype* anchor_data = (const Dtype*) anchors->data();
    const Dtype* bbox_deltas_data = (const Dtype*) bbox_deltas->data();
    Dtype* proposals_data = (Dtype*) proposals->data();
    const Dtype* variances_data = nullptr;
    float normalized = !param.box_normalized ? 1.f : 0;

    if (variances) {
        variances_data = (const Dtype*)variances->data();
    }

    for (int64_t row_id = 0; row_id < row; ++row_id) {
        for (int64_t col_id = 0; col_id < col; ++col_id) {
            size_t delta_offset = row_id * col * delta_len + col_id * delta_len;
            size_t out_offset = row_id * col * out_len + col_id * out_len;
            int prior_box_offset = param.axis == 0 ? col_id * anchor_len : row_id * anchor_len;
            int var_offset = param.axis == 0 ? col_id * var_len : row_id * var_len;
            auto anchor_data_tmp = anchor_data + prior_box_offset + 1;
            auto bbox_deltas_data_tmp = bbox_deltas_data + delta_offset;
            auto proposals_data_tmp = proposals_data + out_offset;
            auto anchor_width = anchor_data_tmp[2] - anchor_data_tmp[0] + normalized;
            auto anchor_height = anchor_data_tmp[3] - anchor_data_tmp[1] + normalized;
            auto anchor_center_x = anchor_data_tmp[0] + 0.5 * anchor_width;
            auto anchor_center_y = anchor_data_tmp[1] + 0.5 * anchor_height;
            Dtype bbox_center_x = 0, bbox_center_y = 0;
            Dtype bbox_width = 0, bbox_height = 0;

            if (fix_size_var == FROM_INPUT_VAR) {
                auto variances_data_tmp = variances_data + var_offset;
                bbox_center_x =
                    variances_data_tmp[0] * bbox_deltas_data_tmp[0] * anchor_width +
                    anchor_center_x;
                bbox_center_y = variances_data_tmp[1] *
                                bbox_deltas_data_tmp[1] * anchor_height + anchor_center_y;
                bbox_width = std::exp(variances_data_tmp[2] *
                                      bbox_deltas_data_tmp[2]) * anchor_width;
                bbox_height = std::exp(variances_data_tmp[3] *
                                       bbox_deltas_data_tmp[3]) * anchor_height;
            }

            if (fix_size_var == FIX_SIZE_VAR) {
                bbox_center_x =
                    variances_data[0] * bbox_deltas_data_tmp[0] * anchor_width +
                    anchor_center_x;
                bbox_center_y = variances_data[1] *
                                bbox_deltas_data_tmp[1] * anchor_height + anchor_center_y;
                bbox_width = std::exp(variances_data[2] *
                                      bbox_deltas_data_tmp[2]) * anchor_width;
                bbox_height = std::exp(variances_data[3] *
                                       bbox_deltas_data_tmp[3]) * anchor_height;

            } else if (fix_size_var == NO_VAR) {
                bbox_center_x =
                    bbox_deltas_data_tmp[0] * anchor_width + anchor_center_x;
                bbox_center_y =
                    bbox_deltas_data_tmp[1] * anchor_height + anchor_center_y;
                bbox_width = std::exp(bbox_deltas_data_tmp[2]) * anchor_width;
                bbox_height = std::exp(bbox_deltas_data_tmp[3]) * anchor_height;
            }

            proposals_data_tmp[0] = bbox_center_x - bbox_width / 2;
            proposals_data_tmp[1] = bbox_center_y - bbox_height / 2;
            proposals_data_tmp[2] = bbox_center_x + bbox_width / 2 - normalized;
            proposals_data_tmp[3] = bbox_center_y + bbox_height / 2 - normalized;
        }
    }
}

template <typename dtype, typename TargetType_D, typename TargetType_H>
void boxcoder_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                    std::vector<Tensor<TargetType_H>*>& outputs, BoxCoderParam<TargetType_D>& param) {
    Tensor<TargetType_H>* anchor = inputs[0];
    Tensor<TargetType_H>* delta = inputs[1];
    Tensor<TargetType_H>* variances = nullptr;
    Tensor<TargetType_H>* proposal = outputs[0];

    if (param.variance() != nullptr && param.variance()->valid_size() > 0) {
        Tensor<TargetType_H> host_tenosr(param.variance()->valid_shape());
        host_tenosr.copy_from(*param.variance());
        variances = &host_tenosr;
        CHECK(variances->valid_size() == 4);
        box_coder<dtype, FIX_SIZE_VAR, TargetType_D, TargetType_H>(proposal, anchor, delta, variances,
                param);
    } else if (inputs.size() >= 3) {
        variances = inputs[2];
        box_coder<dtype, FROM_INPUT_VAR, TargetType_D, TargetType_H>(proposal, anchor, delta, variances,
                param);
    } else {
        box_coder<dtype, NO_VAR, TargetType_D, TargetType_H>(proposal, anchor, delta, variances, param);
    }
};

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_model() {

    TestSaberBase<TargetType_D, TargetType_H, Dtype, BoxCoder, BoxCoderParam> testbase(2, 1);
    int box_num = 10;
    int class_num = 11;
    Shape prior_box_shape({box_num, 5, 1, 1}, Layout_NCHW);
    Shape delta_shape({class_num, box_num, 1, 4}, Layout_NCHW);
    Shape var_shape({1, 1, 1, 4}, Layout_NCHW);
    Tensor<TargetType_D> var_tensor(var_shape);
    fill_tensor_rand(var_tensor, 0, 1);
    BoxCoderParam<TargetType_D> param(&var_tensor, false, 0);



    testbase.set_param(param);//set param
    std::vector<Shape> shape_v;
    shape_v.push_back(prior_box_shape);//scale
    shape_v.push_back(delta_shape);//x
    testbase.set_input_shape(shape_v);//add some input shape
    testbase.set_rand_limit(-1.f, 1.f);
    testbase.run_test(boxcoder_basic<float, TargetType_D, TargetType_H>, 0.00001, true, false);//run test


}

TEST(TestSaberFunc, test_func_axpy) {

#ifdef USE_CUDA
    //Init the test_base
    test_model<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    test_model<AK_FLOAT, X86, X86>();
#endif
#ifdef USE_ARM_PLACE
    test_model<AK_FLOAT, ARM, ARM>();
#endif
}


int main(int argc, const char** argv) {

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

