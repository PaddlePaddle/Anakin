#include "saber/core/context.h"
#include "saber/funcs/conv.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "conv_func_helper.h"
#include <vector>

using namespace anakin::saber;

#define SPEED_CHECK

template<typename TargetType, typename TargetType_H>
int test_conv_results(int group,
                      int input_num, int in_channels, int height, int width,
                      int out_channels, int kernel_h, int kernel_w,
                      int stride_h, int stride_w, int dilation_h, int dilation_w,
                      int pad_h, int pad_w, bool bias_term, bool with_relu,
                      SaberImplStrategy strategy, ImplEnum imp) {

    Shape input_s({input_num, in_channels, height, width}, Layout_NCHW);
    Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);

    // init input Tensor
    Tensor<TargetType> input_dev;
    Tensor<TargetType_H> input_host;
    input_dev.re_alloc(input_s, AK_FLOAT);
    input_host.re_alloc(input_s, AK_FLOAT);
    fill_tensor_rand(input_dev, -10.0f, 20.f);
    input_host.copy_from(input_dev);

    // init weights Tensor
    Tensor<TargetType> weights_dev;
    Tensor<TargetType_H> weights_host;
    weights_dev.re_alloc(weights_s, AK_FLOAT);
    weights_host.re_alloc(weights_s, AK_FLOAT);
    fill_tensor_rand(weights_dev, -10.f, 20.0f);
    weights_host.copy_from(weights_dev);

    Tensor<TargetType> bias_dev;
    Tensor<TargetType_H> bias_host;
    if (bias_term) {
        bias_dev.re_alloc(bias_s, AK_FLOAT);
        bias_host.re_alloc(bias_s, AK_FLOAT);
        fill_tensor_rand(bias_dev, -10.0f, 20.0f);
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
    if (with_relu) {
        ActivationParam<TargetType> act_param(Active_relu);
        param.activation_param = act_param;
    }
    Conv<TargetType, AK_FLOAT> conv;
    std::vector<Tensor<TargetType>* > input_v;
    std::vector<Tensor<TargetType>* > output_v;
    input_v.push_back(&input_dev);
    output_v.push_back(&output_dev);
    conv.compute_output_shape(input_v, output_v, param);
    output_dev.re_alloc(output_dev.valid_shape(), AK_FLOAT);

    conv.init(input_v, output_v, param, strategy, imp, ctx1);
    conv(input_v, output_v, param, ctx1);
    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
    output_v[0]->record_event(stream);
    output_v[0]->sync();
    output_host.re_alloc(output_dev.valid_shape(), AK_FLOAT);
    output_host.copy_from(output_dev);
    check_host.re_alloc(output_host.valid_shape(), AK_FLOAT);

    conv_basic_check<TargetType_H>(input_host, check_host,
                                   (const float*)weights_host.data(), (const float*)bias_host.data(),
                                   group, kernel_w, kernel_h, stride_w, stride_h,
                                   dilation_w, dilation_h, pad_w, pad_h, bias_term,
                                   param.activation_param.has_active);
    double max_ratio = 0.0;
    double max_diff = 0.0;
    tensor_cmp_host((const float*)output_host.data(), (const float*)check_host.data(),
                    check_host.valid_size(), max_ratio, max_diff);
    if (max_ratio < 1e-4) {
                LOG(INFO) << " PASS!!! max_ratio = " << max_ratio << " max_diff = " << max_diff;
        return 0;
    } else {
        print_tensor_valid(output_host);
        print_tensor_valid(check_host);
        LOG(FATAL) << "FAIL!!! max_ratio = " << max_ratio << " max_diff = " << max_diff
                   << " conv param: "
                   << " input_num = " << input_num
                   << " in_channels = " << in_channels
                   << " height = " << height
                   << " width = " << width
                   << " group = " << group
                   << " pad_h = " << pad_h
                   << " pad_w = " << pad_w
                   << " stride_h = " << stride_h
                   << " stride_w = " << stride_w
                   << " dilation_h = " << dilation_h
                   << " dilation_w = " << dilation_w
                   << " kernel_h = " << kernel_h
                   << " kernel_w = " << kernel_w
                   << " out_channels = " << out_channels
                   << " impl: " << ((imp == VENDER_IMPL) ? " VENDER " : " SABER")
                   << " bias_term = " << (bias_term ? "true" : "false")
                   << " with_relu = " << (with_relu ? "true" : "false");
        return -1;
    }

}

template<typename TargetType, typename TargetType_H>
void test_conv_speed(int group,
                     int input_num, int in_channels, int height, int width,
                     int out_channels, int kernel_h, int kernel_w,
                     int stride_h, int stride_w, int dilation_h, int dilation_w,
                     int pad_h, int pad_w, bool bias_term, bool with_relu) {

    Shape input_s({input_num, in_channels, height, width}, Layout_NCHW);
    Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);

    // init input Tensor
    Tensor<TargetType> input_dev;
    Tensor<TargetType_H> input_host;
    input_dev.re_alloc(input_s, AK_FLOAT);
    input_host.re_alloc(input_s, AK_FLOAT);
    fill_tensor_rand(input_dev, -10.0f, 20.f);
    input_host.copy_from(input_dev);
    input_dev.set_scale({0.1});
    // init weights Tensor
    Tensor<TargetType> weights_dev;
    Tensor<TargetType_H> weights_host;
    weights_dev.re_alloc(weights_s, AK_FLOAT);
    weights_host.re_alloc(weights_s, AK_FLOAT);
    fill_tensor_rand(weights_dev, -10.f, 20.0f);
    weights_host.copy_from(weights_dev);

    Tensor<TargetType> bias_dev;
    Tensor<TargetType_H> bias_host;
    if (bias_term) {
        bias_dev.re_alloc(bias_s, AK_FLOAT);
        bias_host.re_alloc(bias_s, AK_FLOAT);
        fill_tensor_rand(bias_dev, -10.0f, 20.0f);
        bias_host.copy_from(bias_dev);
    }
    Tensor<TargetType> output_dev;
    Tensor<TargetType> output_cudnn_dev;
    Tensor<TargetType_H> output_host;

    Context<TargetType> ctx1(0, 1, 1);
    ConvParam<TargetType> param(group, pad_h, pad_w,
                                stride_h, stride_w,
                                dilation_h, dilation_w,
                                &weights_dev, &bias_dev);
    if (with_relu) {
        ActivationParam<TargetType> act_param(Active_relu);
        param.activation_param = act_param;
    }
    Conv<TargetType, AK_FLOAT> conv;
    Conv<TargetType, AK_FLOAT> conv_cudnn;
    std::vector<Tensor<TargetType>* > input_v;
    std::vector<Tensor<TargetType>* > output_v;
    std::vector<Tensor<TargetType>* > output_cudnn_v;
    input_v.push_back(&input_dev);
    output_v.push_back(&output_dev);
    output_cudnn_v.push_back(&output_cudnn_dev);
    conv.compute_output_shape(input_v, output_v, param);
    output_dev.re_alloc(output_dev.valid_shape(), AK_FLOAT);
    output_cudnn_dev.re_alloc(output_dev.valid_shape(), AK_FLOAT);

    conv.init(input_v, output_v, param, SPECIFY, SABER_IMPL, ctx1);
    conv_cudnn.init(input_v, output_cudnn_v, param, SPECIFY, VENDER_IMPL, ctx1);
    conv(input_v, output_v, param, ctx1);
    conv_cudnn(input_v, output_v, param, ctx1);
    SaberTimer<NV> saber_t1, cudnn_t1;

    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
    output_v[0]->record_event(stream);
    output_v[0]->sync();
    output_cudnn_v[0]->record_event(stream);
    output_cudnn_v[0]->sync();
    int ts = 100;
    for (int i = 0; i < ts; ++i) {
        saber_t1.start(ctx1);
        conv(input_v, output_v, param, ctx1);
        output_v[0]->record_event(stream);
        output_v[0]->sync();
        saber_t1.end(ctx1);

        cudnn_t1.start(ctx1);
        conv_cudnn(input_v, output_v, param, ctx1);
        output_cudnn_v[0]->record_event(stream);
        output_cudnn_v[0]->sync();
        cudnn_t1.end(ctx1);
    }
    LOG(INFO) << "saber time: "<<saber_t1.get_average_ms()<< " ms cudnn time: "<< cudnn_t1.get_average_ms()<<" ms";
}

TEST(TestSaberFunc, test_saber_depthwise_conv_results) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
#endif
    std::vector<int> kernel_h_v{1, 3};
    std::vector<int> kernel_w_v{1, 3};
    std::vector<int> pad_h_v{0, 1};
    std::vector<int> pad_w_v{0, 1};
    std::vector<int> stride_h_v{1, 2};
    std::vector<int> stride_w_v{1, 2};
    std::vector<int> dilation_h_v{1};
    std::vector<int> dilation_w_v{1};
    std::vector<int> group_v{4, 7, 32};
    std::vector<int> in_h_v{4, 32};
    std::vector<int> in_w_v{4, 32};
    std::vector<int> input_num_v{3, 1};
    std::vector<bool> bias_term_v{true, false};
    std::vector<bool> with_relu_v{true, false};

#pragma omp parallel for num_threads(8) collapse(3) schedule(dynamic)
    for (int input_num_i = 0; input_num_i < input_num_v.size(); input_num_i++)
    for (int bias_term_i = 0; bias_term_i < bias_term_v.size(); bias_term_i++)
    for (int with_relu_i = 0; with_relu_i < with_relu_v.size(); with_relu_i++)
    for (auto kernel_h : kernel_h_v)
    for (auto kernel_w : kernel_w_v)
    for (auto pad_h : pad_h_v)
    for (auto pad_w : pad_w_v)
    for (auto stride_h : stride_h_v)
    for (auto stride_w : stride_w_v)
    for (auto height : in_h_v)
    for (auto width : in_w_v)
    for (auto dilation_h : dilation_h_v)
    for (auto dilation_w : dilation_w_v)
    for (auto bias_term : bias_term_v)
    for (auto with_relu : with_relu_v)
    for (auto group : group_v) {
        int input_num = input_num_v[input_num_i];
        int out_channels = group;
        int in_channels = group;
        bool with_relu = with_relu_v[with_relu_i];
        bool bias_term = bias_term_v[bias_term_i];
#ifdef USE_CUDA
        if (test_conv_results<NV, NVHX86>(group, input_num, in_channels,
                                    height, width, out_channels, kernel_h,
                                    kernel_w, stride_h, stride_w, dilation_h, dilation_w,
                                    pad_h, pad_w, bias_term, with_relu, SPECIFY, VENDER_IMPL) !=0)  {
                                        LOG(INFO) << "cudnn results error!";
                                    }

        if (test_conv_results<NV, NVHX86>(group, input_num, in_channels,
                            height, width, out_channels, kernel_h,
                            kernel_w, stride_h, stride_w, dilation_h, dilation_w,
                            pad_h, pad_w, bias_term, with_relu, SPECIFY, SABER_IMPL) != 0) {
                                LOG(INFO) << " saber results error!";
                            }

#endif
//#ifdef USE_X86_PLACE
//        test_conv_results<X86, X86>(group,
//                                      input_num, in_channels,
//                                      height,
//                                      width,
//                                      out_channels, kernel_h,
//                                      kernel_w,
//                                      stride_h, stride_w,
//                                      dilation_h,
//                                      dilation_w,
//                                      pad_h, pad_w, bias_term,
//                                      SPECIFY,
//                                      VENDER_IMPL);
//#endif
                                                                }
}
#ifdef SPEED_CHECK
TEST(TestSaberFunc, test_saber_depthwise_conv_speed) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
#endif
    std::vector<int> kernel_h_v{1, 3};
    std::vector<int> kernel_w_v{1, 3};
    std::vector<int> pad_h_v{0, 1};
    std::vector<int> pad_w_v{0, 1};
    std::vector<int> stride_h_v{1, 2};
    std::vector<int> stride_w_v{1, 2};
    std::vector<int> dilation_h_v{1};
    std::vector<int> dilation_w_v{1};
    std::vector<int> group_v{8, 32};
    std::vector<int> in_h_v{17, 32};
    std::vector<int> in_w_v{17, 32};
    std::vector<int> input_num_v{3, 1};
    std::vector<bool> bias_term_v{true, false};
    std::vector<bool> with_relu_v{true, true};

#pragma omp parallel for num_threads(8) collapse(3) schedule(dynamic)
    for (int input_num_i = 0; input_num_i < input_num_v.size(); input_num_i++)
    for (int bias_term_i = 0; bias_term_i < bias_term_v.size(); bias_term_i++)
    for (int with_relu_i = 0; with_relu_i < with_relu_v.size(); with_relu_i++)
    for (auto kernel_h : kernel_h_v)
    for (auto kernel_w : kernel_w_v)
    for (auto pad_h : pad_h_v)
    for (auto pad_w : pad_w_v)
    for (auto stride_h : stride_h_v)
    for (auto stride_w : stride_w_v)
    for (auto height : in_h_v)
    for (auto width : in_w_v)
    for (auto dilation_h : dilation_h_v)
    for (auto dilation_w : dilation_w_v)
    for (auto bias_term : bias_term_v)
    for (auto with_relu : with_relu_v)
    for (auto group : group_v) {
        int input_num = input_num_v[input_num_i];
        int out_channels = group;
        int in_channels = group;
        bool with_relu = with_relu_v[with_relu_i];
        bool bias_term = bias_term_v[bias_term_i];
#ifdef USE_CUDA
        test_conv_speed<NV, NVHX86>(group, input_num, in_channels,
            height, width, out_channels, kernel_h,
            kernel_w, stride_h, stride_w, dilation_h, dilation_w,
            pad_h, pad_w, bias_term, with_relu);

#endif
//#ifdef USE_X86_PLACE
//        test_conv_results<X86, X86>(group,
//                                      input_num, in_channels,
//                                      height,
//                                      width,
//                                      out_channels, kernel_h,
//                                      kernel_w,
//                                      stride_h, stride_w,
//                                      dilation_h,
//                                      dilation_w,
//                                      pad_h, pad_w, bias_term,
//                                      SPECIFY,
//                                      VENDER_IMPL);
//#endif
    }
}
#endif
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
