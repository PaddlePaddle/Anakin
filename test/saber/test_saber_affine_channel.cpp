#include "saber/core/context.h"
#include "saber/funcs/affine_channel.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include <cmath>

using namespace anakin::saber;

template <typename dtype, typename TargetType_D, typename TargetType_H>
void affine_channel_cpu_base(const std::vector<Tensor<TargetType_H>* >& inputs,
                  std::vector<Tensor<TargetType_H>* >& outputs,
                  AffineChannelParam<TargetType_D>& param) {
    const dtype* src = (const dtype*)inputs[0]->data();
    const dtype* scale = (const dtype*)inputs[1]->data();
    const dtype* bias = (const dtype*)inputs[2]->data();
    dtype* dst = (dtype*)outputs[0]->mutable_data();
    int channel_idx = inputs[0]->channel_index();
    int channel = inputs[0]->channel();
    CHECK_EQ(inputs[1]->valid_size(), channel) << "affine channel input scale dims are not valid";
    CHECK_EQ(inputs[2]->valid_size(), channel) << "affine channel input bias dims are not valid";
    int outer_num = inputs[0]->count_valid(0, channel_idx);
    int inner_num = inputs[0]->count_valid(channel_idx+1, inputs[0]->dims());
    int id = 0;
    for (int i = 0; i < outer_num; i++) {
        for (int j = 0; j < channel; j++) {
            for (int k = 0; k < inner_num; k++) {
                dst[id] = src[id] * scale[j] + bias[j];
                id++;
            }
        }
    }
}

TEST(TestSaberFunc, test_op_affine_channel) {

#ifdef USE_CUDA
    TestSaberBase<NV, NVHX86, AK_FLOAT, AffineChannel, AffineChannelParam> testbase(3, 1);

    for (int w_in : {8, 8, 16}) {
        for (int h_in : {2, 8, 32}) {
            for (int ch_in : {2, 3, 8, 64}) {
                for (int num_in : {1, 21, 32}) {
                    Shape shape({num_in, ch_in, h_in, w_in});
                    Shape scale_shape({1, ch_in, 1, 1});
                    Shape bias_shape({1, ch_in, 1, 1});
                    std::vector<Shape> shape_vec = {shape, scale_shape, bias_shape};
                    AffineChannelParam<NV> param;
                    testbase.set_param(param);
                    testbase.set_rand_limit(-5.0, 5.0);
                    testbase.add_inputs_shape(shape_vec);
                    testbase.run_test(affine_channel_cpu_base<float, NV, NVHX86>, 2.1e-5f);
                }
            }
        }
    }
#endif

#ifdef USE_X86_PLACE
    TestSaberBase<X86, X86, AK_FLOAT, AffineChannel, AffineChannelParam> testbase_x86(3, 1);

    for (int w_in : {8, 8, 16}) {
        for (int h_in : {2, 8, 32}) {
            for (int ch_in : {2, 3, 8, 64}) {
                for (int num_in : {1, 21, 32}) {
                    Shape shape({num_in, ch_in, h_in, w_in});
                    Shape scale_shape({1, ch_in, 1, 1});
                    Shape bias_shape({1, ch_in, 1, 1});
                    std::vector<Shape> shape_vec = {shape, scale_shape, bias_shape};
                    AffineChannelParam<X86> param_x86;
                    testbase_x86.set_param(param_x86);
                    testbase_x86.set_rand_limit(-5.0, 5.0);
                    testbase_x86.add_inputs_shape(shape_vec);
                    testbase_x86.run_test(affine_channel_cpu_base<float, X86, X86>);
                }
            }
        }
    }
#endif

}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
