#include "saber/core/context.h"
#include "saber/funcs/conv.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "conv_func_helper.h"
#include <vector>

using namespace anakin::saber;

template<typename TargetType, typename TargetType_H>
void test_conv_results(int group,
                       int input_num, int in_channels, int height, int width,
                       int out_channels, int kernel_h, int kernel_w,
                       int stride_h, int stride_w, int dilation_h, int dilation_w,
                       int pad_h, int pad_w, bool bias_term) {
//    LOG(INFO) << "conv param: "
//            << " input_num = " << input_num
//            << " in_channels = " << in_channels
//            << " height = " << height
//            << " width = " << width
//            << " group = " << group
//            << " pad_h = " << pad_h
//            << " pad_w = " << pad_w
//            << " stride_h = " << stride_h
//            << " stride_w = " << stride_w
//            << " dilation_h = " << dilation_h
//            << " dilation_w = " << dilation_w
//            << " kernel_h = " << kernel_h
//            << " kernel_w = " << kernel_w
//            << " out_channels = " << out_channels;

    Shape input_s({input_num, in_channels, height, width}, Layout_NCHW);
    Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);

    // init input Tensor
    Tensor<TargetType> input_dev;
    Tensor<TargetType_H> input_host;
    input_dev.re_alloc(input_s, AK_FLOAT);
    input_host.re_alloc(input_s, AK_FLOAT);
    fill_tensor_rand(input_dev, -255.0, 255.0);
    input_host.copy_from(input_dev);

    // init weights Tensor
    Tensor<TargetType> weights_dev;
    Tensor<TargetType_H> weights_host;
    weights_dev.re_alloc(weights_s, AK_FLOAT);
    weights_host.re_alloc(weights_s, AK_FLOAT);
    fill_tensor_rand(weights_dev, -255.0, 255.0);
    weights_host.copy_from(weights_dev);

    Tensor<TargetType> bias_dev;
    Tensor<TargetType_H> bias_host;
    if (bias_term) {
        bias_dev.re_alloc(bias_s, AK_FLOAT);
        bias_host.re_alloc(bias_s, AK_FLOAT);
        fill_tensor_rand(bias_dev, -255.0, 255.0);
        bias_host.copy_from(bias_dev);
    }
    Tensor<TargetType> output_dev;
    Tensor<TargetType_H> output_host;
    Tensor<TargetType_H> check_host;

    Context<TargetType> ctx1(0, 1, 1);

    ConvParam<TargetType> param(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);
    Conv<TargetType, AK_FLOAT> conv;
    std::vector<Tensor<TargetType>* > input_v;
    std::vector<Tensor<TargetType>* > output_v;
    input_v.push_back(&input_dev);
    output_v.push_back(&output_dev);
    conv.compute_output_shape(input_v, output_v, param);
    output_dev.re_alloc(output_dev.valid_shape(), AK_FLOAT);

    conv.init(input_v, output_v, param, SPECIFY, VENDER_IMPL, ctx1);
    conv(input_v, output_v, param, ctx1);

    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
    output_v[0]->record_event(stream);
    output_v[0]->sync();
    output_host.re_alloc(output_dev.valid_shape(), AK_FLOAT);
    output_host.copy_from(output_dev);

    check_host.re_alloc(output_host.valid_shape(), AK_FLOAT);

    conv_basic_check<NVHX86>(input_host, check_host,
                             (const float*)weights_host.data(), (const float*)bias_host.data(),
                             group, kernel_w, kernel_h, stride_w, stride_h,
                             dilation_w, dilation_h, pad_w, pad_h, bias_term, false);
    double max_ratio = 0.0;
    double max_diff = 0.0;
    tensor_cmp_host((const float*)output_host.data(), (const float*)check_host.data(),
                    check_host.valid_size(), max_ratio, max_diff);
    if (max_ratio > 0.001) {
        LOG(INFO) << " PASS!!!";
    } else {
        LOG(INFO) << "FAIL!!! max_ratio = "<< max_ratio << " max_diff = "<<max_diff;
    }

}

TEST(TestSaberFunc, test_saber_conv_results) {
    int group = 4;
    int input_num = 1;
    int in_channels = 4;
    int height = 16;
    int width = 16;
    int out_channels = 4;
    int kernel_h = 7;
    int kernel_w = 2;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int pad_h = 1;
    int pad_w = 1;
    bool bias_term = true;
    Env<NV>::env_init();
    Env<NVHX86>::env_init();

    test_conv_results<NV, NVHX86>(group,
            input_num, in_channels, height, width,
            out_channels, kernel_h, kernel_w,
            stride_h, stride_w, dilation_h, dilation_w,
            pad_h, pad_w, bias_term);
}

TEST(TestSaberFunc, test_saber_conv_speed) {

}

TEST(TestSaberFunc, test_saber_conv_op_func) {

}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}