#include "saber/core/context.h"
#include "saber/funcs/anchor_generator.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include <cmath>

using namespace anakin::saber;
template <typename dtype, typename TargetType_D, typename TargetType_H>
void anchor_generator_cpu_base(const std::vector<Tensor<TargetType_H>* >& inputs,
                  std::vector<Tensor<TargetType_H>* >& outputs,
                  AnchorGeneratorParam<TargetType_D>& param) {
    const dtype* src = (const dtype*)inputs[0]->data();
    dtype* dst = (dtype*)outputs[0]->mutable_data();
    dtype* var = (dtype*)outputs[1]->mutable_data();
    auto anchor_sizes = param.anchor_sizes;
    auto aspect_ratios = param.aspect_ratios;
    auto stride = param.stride;
    auto variances = param.variances;
    auto offset = param.offset;
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    int stride_w = stride[0];
    int stride_h = stride[1];
    auto anchor_tmp = dst;
    auto var_tmp = var;
    for (int h_idx = 0; h_idx < height; h_idx++) {
        for (int w_idx = 0; w_idx < width; w_idx++) {
            dtype x_ctr = (w_idx * stride_w) + offset * (stride_w - 1);
            dtype y_ctr = (h_idx * stride_h) + offset * (stride_h - 1);
            for (size_t r = 0; r < aspect_ratios.size(); r++) {
                auto ar = aspect_ratios[r];
                for (size_t s = 0; s < anchor_sizes.size(); s++) {
                    auto anchor_size = anchor_sizes[s];
                    dtype area = stride_w * stride_h;
                    dtype area_ratios = area / ar;
                    dtype base_w = round(sqrt(area_ratios));
                    dtype base_h = round(base_w * ar);
                    dtype scale_w = anchor_size / stride_w;
                    dtype scale_h = anchor_size / stride_h;
                    dtype half_width = 0.5 * (scale_w * base_w - 1);
                    dtype half_height = 0.5 * (scale_h * base_h - 1);
                    anchor_tmp[0] = x_ctr - half_width;
                    anchor_tmp[1] = y_ctr - half_height;
                    anchor_tmp[2] = x_ctr + half_width;
                    anchor_tmp[3] = y_ctr + half_height;
                    var_tmp[0] = variances[0];
                    var_tmp[1] = variances[1];
                    var_tmp[2] = variances[2];
                    var_tmp[3] = variances[3];
                    anchor_tmp += 4;
                    var_tmp += 4;
                }
            }
        }
    }

}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_anchor_generator() {
   std::vector<float> anchor_sizes = {16, 32, 64, 128};
   std::vector<float> aspect_ratios = {0.5, 1, 2};
   std::vector<float> stride = {4, 4};
   std::vector<float> variances = {0.1, 0.2, 0.3, 0.4};
   auto offset = 0.5;
    TestSaberBase<TargetType_D, TargetType_H, Dtype, 
            AnchorGenerator, AnchorGeneratorParam> testbase(1, 2);
    for (int w_in : {16, 32}) {
        for (int h_in : {16, 32}) {
            for (int ch_in : {1, 5, 7}) {
                for (int num_in : {1, 2, 5}) {
                    Shape shape({num_in, ch_in, h_in, w_in});
                    AnchorGeneratorParam<TargetType_D> param(anchor_sizes,
                            aspect_ratios,
                            variances,
                            stride,
                            offset);
                    testbase.set_param(param);
                    testbase.set_rand_limit(-5.0, 5.0);
                    testbase.set_input_shape(shape);
                    testbase.run_test(anchor_generator_cpu_base<float, TargetType_D, TargetType_H>, 2.1e-5f, true, false);
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_op_anchor_generator) {
#ifdef USE_CUDA
test_anchor_generator<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
test_anchor_generator<AK_FLOAT, X86, X86>();
#endif

}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
