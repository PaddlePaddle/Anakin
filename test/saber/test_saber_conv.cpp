#include "saber/core/context.h"
#include "saber/funcs/conv.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "conv_func_helper.h"
#include <vector>

using namespace anakin::saber;
#define CHECK_RESULT
//#define CHECK_SPEED
#ifdef AMD_GPU
#define RUN_BASIC_TEST true
#else
#define RUN_BASIC_TEST false
#endif
#if 0
#ifdef USE_BM_PLACE
TEST(TestSaberFunc, test_saber_conv_results_bm) {
    Env<BM>::env_init();
    Env<X86>::env_init();
    TestSaberBase<BM,X86,AK_FLOAT,Conv,ConvParam> testbase_bm;
    std::vector<int> kernel{1, 3};
    std::vector<int> pad{0, 1};
    std::vector<int> stride_h_v{1};
    std::vector<int> dilation_h_w{1, 2};
    std::vector<int> in_channels_v{1, 2};
    std::vector<int> out_channels_v{1, 2};
    std::vector<int> group_v{1};
    std::vector<int> in_h_v{6};
    std::vector<int> in_w_v{6};
    std::vector<int> input_num_v{2, 1};
    std::vector<bool> bias_term_v{true, false};
    std::vector<bool> with_relu_v{false};

    for (int input_num :{1,2})
    for (int out_channels :{1,2,5})
    for (int in_channels :{1,2,5})
    for (auto kernel_h_w : kernel)
    for (auto pad_h_w : pad)
    for (auto stride_h : stride_h_v)
    for (auto stride_w : stride_h_v)
    for (auto height : in_h_v)
    for (auto width : in_w_v)
    for (auto dilation : dilation_h_w)
    for (auto bias_term : bias_term_v)
    for (auto with_relu : with_relu_v)
    for (auto group : group_v) {
        LOG(INFO)<<"info :"<<input_num<<","<< in_channels<<","<<
        height<<","<< width<<","<< out_channels<<","<< kernel_h_w<<","<<
        kernel_h_w<<","<< stride_h<<","<< stride_w<<","<< dilation<<","<< dilation<<","<<
        pad_h_w<<","<< pad_h_w<<","<< bias_term;
        Shape weights_s({out_channels, in_channels, kernel_h_w, kernel_h_w}, Layout_NCHW);
        Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);
        Tensor<BM> weights_dev;
        Tensor<BM> bias_dev;

        weights_dev.re_alloc(weights_s, AK_FLOAT);
        fill_tensor_rand(weights_dev, -5.f, 5.0f);
        if (bias_term) {
            bias_dev.re_alloc(bias_s, AK_FLOAT);
            fill_tensor_rand(bias_dev, -5.0f, 5.0f);
        }
        ConvParam<BM> param_bm(group, pad_h_w, pad_h_w,
                               stride_h, stride_w,
                               dilation, dilation,
                               &weights_dev, &bias_dev);
        testbase_bm.set_param(param_bm);//set param
        testbase_bm.set_input_shape(Shape({input_num,in_channels,height,width},
                                          Layout_NCHW));//add some input shape
        testbase_bm.run_test(conv_cpu_func<float, BM, X86>, 1e-3);//run test

    }
}
#endif
#endif

TEST(TestSaberFunc, test_saber_conv_results) {
#ifdef USE_CUDA
//    Env<NV>::env_init();
//    Env<NVHX86>::env_init();
//    TestSaberBase<NV, NVHX86, AK_FLOAT, Conv, ConvParam> testbase_nv;
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
    TestSaberBase<X86, X86, AK_FLOAT, Conv, ConvParam> testbase_x86;
#endif
    std::vector<int> kernel_h_v{1, 3};
    std::vector<int> kernel_w_v{1, 3};
    std::vector<int> pad_h_v{0, 1};
    std::vector<int> pad_w_v{0, 1};
    std::vector<int> stride_h_v{1, 2};
    std::vector<int> stride_w_v{1, 2};
    std::vector<int> dilation_h_v{1};
    std::vector<int> dilation_w_v{1};
    std::vector<int> group_v{1};
    std::vector<int> in_h_v{4, 16};
    std::vector<int> in_w_v{16, 8};
    std::vector<int> input_num_v{1, 3};
    std::vector<int> input_channels_v{4};
    std::vector<int> output_channels_v{4};
    std::vector<bool> bias_term_v{true, false};
    std::vector<bool> with_relu_v{true, false};
    if (RUN_BASIC_TEST) {
    for (int bias_term : bias_term_v)
    for (int with_relu : with_relu_v)
    for (auto kernel_h : kernel_h_v)
    for (auto kernel_w : kernel_w_v)
    for (auto pad_h : pad_h_v)
    for (auto pad_w : pad_w_v)
    for (auto stride_h : stride_h_v)
    for (auto stride_w : stride_w_v)
    for (auto dilation_h : dilation_h_v)
    for (auto dilation_w : dilation_w_v)
    for (auto bias_term : bias_term_v)
    for (auto with_relu : with_relu_v)
    for (auto in_channels : input_channels_v)
    for (auto out_channels : output_channels_v)
    for (auto group : group_v) {

        Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
        Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);
#ifdef USE_CUDA
//        Tensor<NV> weights_dev;
//        Tensor<NV> bias_dev;
//
//        weights_dev.re_alloc(weights_s, AK_FLOAT);
//        fill_tensor_rand(weights_dev, -5.f, 5.0f);
//        if (bias_term) {
//            bias_dev.re_alloc(bias_s, AK_FLOAT);
//            fill_tensor_rand(bias_dev, -5.0f, 5.0f);
//        }
//        ConvParam<NV> param_nv(group, pad_h, pad_w,
//                               stride_h, stride_w,
//                               dilation_h, dilation_w,
//                               &weights_dev, &bias_dev);
//        if (with_relu) {
//            param_nv.activation_param = ActivationParam<NV>(Active_relu);
//        }
#endif
#ifdef USE_X86_PLACE
        Tensor<X86> weights_x86;
        weights_x86.re_alloc(weights_s, AK_FLOAT);
        fill_tensor_rand(weights_x86, -5.f, 5.0f);

        Tensor<X86> bias_x86;
        if (bias_term) {
            bias_x86.re_alloc(bias_s, AK_FLOAT);
            fill_tensor_rand(bias_x86, -5.0f, 5.0f);
        }
        ConvParam<X86> param_x86(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_x86, &bias_x86);
        if (with_relu) {
            param_x86.activation_param = ActivationParam<X86>(Active_relu);
        }
#endif
        for (auto input_num : input_num_v)
        for (auto height : in_h_v)
        for (auto width : in_w_v) {
#ifdef USE_CUDA

//            testbase_nv.set_param(param_nv);//set param
//            testbase_nv.set_input_shape(Shape({input_num,in_channels,height,width},
//                                              Layout_NCHW));//add some input shape
//            testbase_nv.run_test(conv_cpu_func<float, NV, NVHX86>, 1e-3);//run test
#endif
#ifdef USE_X86_PLACE
            testbase_x86.set_param(param_x86);//set param
            testbase_x86.set_input_shape(Shape({input_num, in_channels, height, width},
                                               Layout_NCHW));//add some input shape
            testbase_x86.run_test(conv_cpu_func<float, X86, X86>, 1e-3);//run test
#endif
        }
    }
    }
}
template<typename TargetType, typename TargetType_H>
int test_conv_results(int group,
                      int input_num, int in_channels, int height, int width,
                      int out_channels, int kernel_h, int kernel_w,
                      int stride_h, int stride_w, int dilation_h, int dilation_w,
                      int pad_h, int pad_w, bool bias_term, bool with_relu,
                      SaberImplStrategy strategy, ImplEnum imp) {

    LOG(INFO)<< " conv param: "
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
             << " bias_term = " << (bias_term ? "true" : "false")
             << " with_relu = " << (with_relu ? "true" : "false");

    Shape input_s({input_num, in_channels, height, width}, Layout_NCHW);
    Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);

    // init input Tensor
    Tensor<TargetType> input_dev;
    Tensor<TargetType_H> input_host;
    input_dev.re_alloc(input_s, AK_FLOAT);
    input_host.re_alloc(input_s, AK_FLOAT);
    fill_tensor_rand(input_dev, -10.0f, 10.0f);
    input_host.copy_from(input_dev);
//    input_dev.set_scale({10.1f / 128});
//    LOG(INFO) << input_dev.get_scale()[0];

    // init weights Tensor
    Tensor<TargetType> weights_dev;
    Tensor<TargetType_H> weights_host;
    weights_dev.re_alloc(weights_s, AK_FLOAT);
    weights_host.re_alloc(weights_s, AK_FLOAT);
    fill_tensor_rand(weights_dev, -10.0f, 10.0f);
    weights_host.copy_from(weights_dev);

    Tensor<TargetType> bias_dev;
    Tensor<TargetType_H> bias_host;
    if (bias_term) {
        bias_dev.re_alloc(bias_s, AK_FLOAT);
        bias_host.re_alloc(bias_s, AK_FLOAT);
        fill_tensor_rand(bias_dev, -10.0f, 10.0f);
        bias_host.copy_from(bias_dev);
    }
    Tensor<TargetType> output_dev;
    Tensor<TargetType_H> output_host;
    Tensor<TargetType_H> check_host;

    Context<TargetType> ctx1(0, 1, 1);
//    ActivationParam<TargetType> act_param(Active_relu);
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
    conv.trans_weights(*param.mutable_weight(), *param.mutable_bias(),
                       param.pad_h, param.pad_w, param.dilation_h, param.dilation_w,
                       param.stride_h, param.stride_w, param.group, imp);

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
//    print_tensor_valid(check_host);
    double max_ratio = 0.0;
    double max_diff = 0.0;
    tensor_cmp_host((const float*)output_host.data(), (const float*)check_host.data(),
                    check_host.valid_size(), max_ratio, max_diff);
    if (max_ratio > 1e-3) {

        //print_tensor_valid(output_host);
        LOG(FATAL) << " max_ratio = " << max_ratio << " max_diff = " << max_diff;
    }
    LOG(INFO) << " max_ratio = " << max_ratio << " max_diff = " << max_diff;
    return 0;
}

TEST(TestSaberFunc, test_saber_cuda_conv_results) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
#endif
    std::vector<int> kernel_h_v{1, 3};
    std::vector<int> kernel_w_v{1, 3};
    std::vector<int> pad_h_v{0, 1};
    std::vector<int> pad_w_v{0, 1};
    std::vector<int> stride_h_v{1, 2};
    std::vector<int> stride_w_v{1, 2};
    std::vector<int> dilation_h_v{1, 2};
    std::vector<int> dilation_w_v{1, 2};
    std::vector<int> in_channels_v{4, 8};
    std::vector<int> out_channels_v{4, 8};
//    std::vector<int> group_v{1, 2, 32};
    std::vector<int> in_h_v{24, 36};
    std::vector<int> in_w_v{24, 36};
    std::vector<int> input_num_v{1, 3};
    std::vector<bool> bias_term_v{true, false};
    std::vector<bool> with_relu_v{true, false};
#ifdef USE_CUDA
    if (RUN_BASIC_TEST) {
    for (auto input_num : input_num_v) {
    for (auto out_channels : out_channels_v) {
    for (auto in_channels : in_channels_v) {
    for (auto kernel_h : kernel_h_v) {
    for (auto kernel_w : kernel_w_v) {
    for (auto height : in_h_v) {
    for (auto width : in_w_v) {
    for (auto stride_h : stride_h_v) {
    for (auto stride_w : stride_w_v) {
    for (auto dilation_h : dilation_h_v) {
    for (auto dilation_w : dilation_w_v) {
    for (auto pad_h : pad_h_v) {
    for (auto pad_w : pad_w_v) {
    for (auto bias_term : bias_term_v) {
    for (auto with_relu : with_relu_v) {
        test_conv_results<NV, NVHX86>(1,
                                      input_num,
                                      in_channels,
                                      height,
                                      width,
                                      out_channels,
                                      kernel_h,
                                      kernel_w,
                                      stride_h, stride_w,
                                      dilation_h, dilation_w,
                                      pad_h, pad_w, bias_term,
                                      with_relu,
                                      SPECIFY,
                                      VENDER_IMPL);
        test_conv_results<NV, NVHX86>(1,
                                      input_num,
                                      in_channels,
                                      height,
                                      width,
                                      out_channels,
                                      kernel_h,
                                      kernel_w,
                                      stride_h, stride_w,
                                      dilation_h, dilation_w,
                                      pad_h, pad_w, bias_term,
                                      with_relu,
                                      SPECIFY,
                                      SABER_IMPL);
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
#endif
}
TEST(TestSaberFunc, test_saber_amd_conv_results) {
#ifdef AMD_GPU
    Env<AMD>::env_init();
    Env<AMDHX86>::env_init();
#endif
    std::vector<int> kernel_h_v{1, 3};
    std::vector<int> kernel_w_v{1, 3};
    std::vector<int> pad_h_v{0, 1};
    std::vector<int> pad_w_v{0, 1};
    std::vector<int> stride_h_v{1, 2};
    std::vector<int> stride_w_v{1, 2};
    std::vector<int> dilation_h_v{1, 2};
    std::vector<int> dilation_w_v{1, 2};
    std::vector<int> in_channels_v{4, 8};
    std::vector<int> out_channels_v{4, 8};
//    std::vector<int> group_v{1, 2, 32};
    std::vector<int> in_h_v{24, 36};
    std::vector<int> in_w_v{24, 36};
    std::vector<int> input_num_v{1, 3};
    std::vector<bool> bias_term_v{true, false};
    std::vector<bool> with_relu_v{true, false};
#ifdef AMD_GPU
    if (RUN_BASIC_TEST) {
    for (auto input_num : input_num_v) {
    for (auto out_channels : out_channels_v) {
    for (auto in_channels : in_channels_v) {
    for (auto kernel_h : kernel_h_v) {
    for (auto kernel_w : kernel_w_v) {
    for (auto height : in_h_v) {
    for (auto width : in_w_v) {
    for (auto stride_h : stride_h_v) {
    for (auto stride_w : stride_w_v) {
    for (auto dilation_h : dilation_h_v) {
    for (auto dilation_w : dilation_w_v) {
    for (auto pad_h : pad_h_v) {
    for (auto pad_w : pad_w_v) {
    for (auto bias_term : bias_term_v) {
    for (auto with_relu : with_relu_v) {
/*
        test_conv_results<AMD, AMDHX86>(1,
                                      input_num,
                                      in_channels,
                                      height,
                                      width,
                                      out_channels,
                                      kernel_h,
                                      kernel_w,
                                      stride_h, stride_w,
                                      dilation_h, dilation_w,
                                      pad_h, pad_w, bias_term,
                                      with_relu,
                                      SPECIFY,
                                      VENDER_IMPL);
*/
        test_conv_results<AMD, AMDHX86>(1,
                                      input_num,
                                      in_channels,
                                      height,
                                      width,
                                      out_channels,
                                      kernel_h,
                                      kernel_w,
                                      stride_h, stride_w,
                                      dilation_h, dilation_w,
                                      pad_h, pad_w, bias_term,
                                      with_relu,
                                      SPECIFY,
                                      SABER_IMPL);
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
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
