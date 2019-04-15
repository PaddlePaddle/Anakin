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
    Tensor<TargetType_H> weight_tensor(param.weight()->valid_shape());
    Tensor<TargetType_H> bias_tensor(param.bias()->valid_shape());
    weight_tensor.copy_from(*param.weight());
    bias_tensor.copy_from(*param.bias());
    AffineChannelParam<TargetType_H> param_h(&weight_tensor, &bias_tensor);
    
    const dtype* scale = (const dtype*)param_h.weight()->data();
    const dtype* bias = (const dtype*)param_h.bias()->data();
    dtype* dst = (dtype*)outputs[0]->mutable_data();
    int channel_idx = inputs[0]->channel_index();
    int channel = inputs[0]->channel();
    CHECK_EQ(param.weight()->valid_size(), channel) << "affine channel input scale dims are not valid";
    CHECK_EQ(param.bias()->valid_size(), channel) << "affine channel input bias dims are not valid";
    int outer_num = inputs[0]->count_valid(0, channel_idx);
    int inner_num = inputs[0]->count_valid(channel_idx+1, inputs[0]->dims());
    int id = 0;
    for (int i = 0; i < outer_num; i++) {
        for (int j = 0; j < channel; j++) {
            for (int k = 0; k < inner_num; k++) {
                dst[id] = src[id] * scale[j] + bias[j];
                //LOG(INFO) << "id" << id;
                //LOG(INFO) << "j" << j;
                //LOG(INFO) << "outer_num" << outer_num;
                //LOG(INFO) << "inner_num" << inner_num;
                id++;
            }
        }
    }
}
template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_affine_channel() {
    TestSaberBase<TargetType_D, TargetType_H, Dtype, AffineChannel, AffineChannelParam> testbase(1, 1);

    for (int w_in : {8, 8, 16}) {
        for (int h_in : {2, 8, 32}) {
            for (int ch_in : {2, 3, 8, 64}) {
                for (int num_in : {1, 21, 32}) {
//    for (int w_in : {8}) {
//        for (int h_in : {2}) {
//            for (int ch_in : {2}) {
//                for (int num_in : {2}) {
                    Shape shape({num_in, ch_in, h_in, w_in});
                    Shape scale_shape({1, ch_in, 1, 1}, Layout_NCHW);
                    Shape bias_shape({1, ch_in, 1, 1}, Layout_NCHW);
                    Tensor<TargetType_D> scale(scale_shape, AK_FLOAT);
                    Tensor<TargetType_D> bias(bias_shape, AK_FLOAT);
                    std::vector<Shape> shape_vec = {shape};
                    fill_tensor_rand(scale, -1.0f, 1.0f);
                    fill_tensor_rand(bias, -1.0f, 1.0f);
                    AffineChannelParam<TargetType_D> param(&scale, &bias);
                    testbase.set_param(param);
                    testbase.set_rand_limit(-5.0, 5.0);
                    testbase.add_inputs_shape(shape_vec);
                    testbase.run_test(affine_channel_cpu_base<float, TargetType_D, TargetType_H>, 2.1e-5f);
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_op_affine_channel) {

#ifdef USE_CUDA
    Env<NV>::env_init();
    test_affine_channel<AK_FLOAT, NV, NVHX86>();
#endif

#ifdef USE_X86_PLACE
//    Env<X86>::env_init();
//    test_affine_channel<AK_FLOAT, X86, X86>();
#endif

}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
