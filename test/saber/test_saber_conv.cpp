#include "saber/core/context.h"
#include "saber/funcs/conv.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "conv_func_helper.h"
#include <vector>
#include "saber/funcs/impl/x86/x86_utils.h"

using namespace anakin::saber;
#define CHECK_RESULT
//#define CHECK_SPEED
#define RUN_BASIC_TEST false
#define RUN_BASIC_TEST_ARM true
#if 0
#ifdef USE_BM_PLACE
TEST(TestSaberFunc, test_saber_conv_results_bm) {
    Env<BM>::env_init();
    Env<X86>::env_init();
    TestSaberBase<BM, X86, AK_FLOAT, Conv, ConvParam> testbase_bm;
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

    for (int input_num : {
                1, 2
            })

        for (int out_channels : {
                    1, 2, 5
                })

            for (int in_channels : {
                        1, 2, 5
                    })

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
                                                        LOG(INFO) << "info :" << input_num << "," << in_channels << "," <<
                                                                  height << "," << width << "," << out_channels << "," << kernel_h_w << "," <<
                                                                  kernel_h_w << "," << stride_h << "," << stride_w << "," << dilation << "," << dilation << "," <<
                                                                  pad_h_w << "," << pad_h_w << "," << bias_term;
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
                                                        testbase_bm.set_input_shape(Shape({input_num, in_channels, height, width},
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
    std::vector<int> kernel_h_v {1, 3};
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
                      SaberImplStrategy strategy, ImplEnum imp, float eps = 1e-3, int threads = 1) {

    LOG(INFO) << " conv param: "
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
              << " with_relu = " << (with_relu ? "true" : "false")
              << " threads = " << threads;

    Shape input_s({input_num, in_channels, height, width}, Layout_NCHW);
    Shape weights_s({out_channels, in_channels / group, kernel_h, kernel_w}, Layout_NCHW);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);

    // init input Tensor
    Tensor<TargetType> input_dev;
    Tensor<TargetType_H> input_host;
    input_dev.re_alloc(input_s, AK_FLOAT);
    input_host.re_alloc(input_s, AK_FLOAT);
    fill_tensor_rand(input_dev, -10.0f, 10.0f);
    //fill_tensor_const(input_dev, 1.f);
    input_host.copy_from(input_dev);
    //    input_dev.set_scale({10.1f / 128});
    //    LOG(INFO) << input_dev.get_scale()[0];

    // init weights Tensor
    Tensor<TargetType> weights_dev;
    Tensor<TargetType_H> weights_host;
    weights_dev.re_alloc(weights_s, AK_FLOAT);
    weights_host.re_alloc(weights_s, AK_FLOAT);
    fill_tensor_rand(weights_dev, -10.0f, 10.0f);
    //fill_tensor_const(weights_dev, 1.f);
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
    #ifdef USE_ARM_PLACE
    ctx1.set_run_mode(SABER_POWER_HIGH, threads);
    #endif
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
    if (max_ratio > eps) {
        if (max_diff > eps){
          print_tensor_valid(weights_host);
          print_tensor_valid(output_host);
          print_tensor_valid(check_host);
          LOG(FATAL) << " max_ratio = " << max_ratio << " max_diff = " << max_diff;
        }
    }
    return 0;
}

template <typename dtype>
int count_diff(const dtype* src1, const dtype* src2,
               int size, double max_ratio,
               bool signed_input = false, bool wino = false) {
    if (max_ratio <= 0) {
        max_ratio = 0.1;
    }
    int count = 0;
    if (wino) {
        // It's a known issue that winograd convolution result is not bitwise identical as direct convolution result.
        return count;
    }
    for (int i = 0; i < size; ++i) {
        if (signed_input && (fabs(src1[i] - src2[i]) <= 1))
            continue;
        double ratio = fabs(src1[i] - src2[i]) / fabs(src1[i] + src2[i] + 1e-12);
        if (ratio > max_ratio) {
            ++count;
        }
    }
    return count;
}

template<typename TargetType, typename TargetType_H>
int test_conv_results_x86_C16R(int group,
                              int input_num, int in_channels, int height, int width,
                              int out_channels, int kernel_h, int kernel_w,
                              int stride_h, int stride_w, int dilation_h, int dilation_w,
                              int pad_h, int pad_w, bool bias_term, bool with_relu,
                              SaberImplStrategy strategy, ImplEnum imp,bool input_nchw=false, bool output_nhwc=false,
                              bool output_uint8=false) {
    float abs_w_x=1.f;
    float abs_b=2.f;

            LOG(INFO) << " conv param: "
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
    Shape input_s;
    if (input_nchw){
        input_s=Shape({input_num, in_channels, height, width}, Layout_NCHW);
    }else{
        input_s=Shape({input_num, in_channels, height, width}, Layout_NCHW_C16R);
    }
    Shape weights_s({out_channels, in_channels / group, kernel_h, kernel_w}, Layout_NCHW);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);
    int out_height = (pad_h * 2 + height - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int out_width = (pad_w * 2 + width - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    Shape output_dev_s;
    if (output_nhwc){
        output_dev_s=Shape({input_num,  out_height, out_width,out_channels}, Layout_NHWC);
    }else{
        output_dev_s=Shape({input_num, out_channels, out_height, out_width}, Layout_NCHW_C16R);
    }


    // init input Tensor
    Tensor<TargetType> input_dev;
    Tensor<TargetType_H> input_host;
    input_dev.re_alloc(input_s, AK_FLOAT);
    input_host.re_alloc(input_s, AK_FLOAT);


    fill_tensor_const(input_dev, abs_w_x);
//    fill_tensor_seq(input_dev);
//    fill_tensor_rand(input_dev, -abs_w_x, abs_w_x);
    input_host.copy_from(input_dev);


    // init weights Tensor
    Tensor<TargetType> weights_dev;
    Tensor<TargetType_H> weights_host;
    weights_dev.re_alloc(weights_s, AK_FLOAT);
    weights_host.re_alloc(weights_s, AK_FLOAT);

    fill_tensor_rand(weights_dev, -abs_w_x, abs_w_x);
    bool nothing_flag = false;
    std::string nothing_str = "";
//    fill_tensor_const(weights_dev, abs_w_x);
//    load_tensor_in_io_format(weights_dev,nothing_flag,nothing_str,"../fp32/record+weights+conv+out+0+64_3_3_3_+nchw+ak_float+0.txt");

    weights_host.copy_from(weights_dev);

    Tensor<TargetType> bias_dev;
    Tensor<TargetType_H> bias_host;

    if (bias_term) {
        bias_dev.re_alloc(bias_s, AK_FLOAT);
        bias_host.re_alloc(bias_s, AK_FLOAT);
//        fill_tensor_const(bias_dev, 1);
//        fill_tensor_const(bias_dev, abs_b);
//        load_tensor_in_io_format(bias_dev,nothing_flag,nothing_str,"../fp32/record+bias+conv+out+0+1_64_1_1_+nchw+ak_float+0.txt");
        fill_tensor_rand(bias_dev, -abs_b, abs_b);
        bias_host.copy_from(bias_dev);
    }

    Tensor<TargetType> output_dev(output_dev_s);
    if (output_uint8){
        output_dev.re_alloc(output_dev_s,AK_UINT8);
        float max_out=(in_channels*kernel_h*kernel_w*abs_w_x*abs_w_x+abs_b);
        output_dev.set_scale({max_out/127.f});
//        output_dev.set_scale({0.038397});
        LOG(INFO)<<"max out "<<max_out;

    }
    Tensor<TargetType_H> output_host(output_dev_s);
    Tensor<TargetType_H> check_host;
    fill_tensor_rand(output_dev, 0.f, 0.f);
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
    SABER_CHECK(conv.init(input_v, output_v, param, strategy, imp, ctx1));
    SABER_CHECK(conv(input_v, output_v, param, ctx1));

    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();

    Tensor<TargetType_H> nchw_input_tensor(Shape({input_num, in_channels, height, width}));
    reorder_nchwc_nchw(input_host, nchw_input_tensor);
    check_host.re_alloc(Shape({input_num, out_channels, out_height, out_width}), AK_FLOAT);
    Tensor<TargetType_H> nchw_output_check(check_host.valid_shape());
    conv_basic_check<TargetType_H>(nchw_input_tensor, check_host,
                                   (const float*)weights_host.data(), (const float*)bias_host.data(),
                                   group, kernel_w, kernel_h, stride_w, stride_h,
                                   dilation_w, dilation_h, pad_w, pad_h, bias_term,
                                   param.activation_param.has_active);
    LOG(INFO) << "cal check finish";
    Tensor<TargetType_H> nchwc16_output_check(check_host.valid_shape());
    if (output_nhwc){
        anakin::saber::reorder_nhwc_nchw(output_dev, nchwc16_output_check);
    }else{
        anakin::saber::reorder_nchwc_nchw(output_dev, nchwc16_output_check);
    }

    double max_ratio = 0.0;
    double max_diff = 0.0;
    if (output_uint8){
        tensor_cmp_host_mlu((const float*)nchwc16_output_check.data(), (const float*)check_host.data(),
                            check_host.valid_size(), max_ratio, max_diff);
        if (max_ratio < 0.15) {
            LOG(INFO)<<"mean ak "<<tensor_mean_value_valid(nchwc16_output_check);
            LOG(INFO)<<"mean "<<tensor_mean_value_valid(check_host);
            LOG(INFO) << "PASS!!! ratio = " << max_ratio <<" in "<<nchwc16_output_check.valid_size();
            return 0;
        }else{
            write_tensorfile(output_dev,"output_dev");
            write_tensorfile(nchwc16_output_check,"nchwc16_output_check");
            write_tensorfile(check_host,"check_host");
            LOG(INFO)<<"mean ak "<<tensor_mean_value_valid(nchwc16_output_check);
            LOG(INFO)<<"mean "<<tensor_mean_value_valid(check_host);
//            print_tensor(output_dev);
//            print_tensor(nchwc16_output_check);
//            print_tensor(check_host);

            LOG(FATAL) << "FAIL!!! ratio = " << max_ratio<<" in "<<nchwc16_output_check.valid_size()<<","
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
                       << " out_channels = " << out_channels;
        }

    }else{
        tensor_cmp_host((const float*)nchwc16_output_check.data(), (const float*)check_host.data(),
                        check_host.valid_size(), max_ratio, max_diff);
        if (max_ratio > 1e-3) {
            print_tensor(output_dev);
            print_tensor(nchwc16_output_check);
            print_tensor(check_host);
            LOG(FATAL) << " max_ratio = " << max_ratio << " max_diff = " << max_diff;
        } else {
            LOG(INFO) << "passed";
        }
    }




    return 0;
}



template<typename TargetType, typename TargetType_H>
int test_conv_results_x86_C8R(int group,
                              int input_num, int in_channels, int height, int width,
                              int out_channels, int kernel_h, int kernel_w,
                              int stride_h, int stride_w, int dilation_h, int dilation_w,
                              int pad_h, int pad_w, bool bias_term, bool with_relu,
                              SaberImplStrategy strategy, ImplEnum imp) {

    LOG(INFO) << " conv param: "
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

    Shape input_s({input_num, in_channels, height, width}, Layout_NCHW_C8R);
    Shape weights_s({out_channels, in_channels / group, kernel_h, kernel_w}, Layout_NCHW);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);
    int out_height = (pad_h * 2 + height - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int out_width = (pad_w * 2 + width - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    Shape output_dev_s({input_num, out_channels, out_height, out_width}, Layout_NCHW_C8R);
    // init input Tensor
    Tensor<TargetType> input_dev;
    Tensor<TargetType_H> input_host;
    input_dev.re_alloc(input_s, AK_FLOAT);
    input_host.re_alloc(input_s, AK_FLOAT);
//    {
//        float *tmp= static_cast<float*>(input_dev.mutable_data());
//        for(int i=0;i<height;i++){
//            for(int j=0;j<width;j++){
//                for(int c=0;c<8;c++){
//                    int index=i*width*8+j*8+c;
//                    tmp[index]=i*width+j;
//                }
//            }
//
//        }
//    }

//        fill_tensor_const(input_dev, 1.f);
//    fill_tensor_seq(input_dev);
    fill_tensor_rand(input_dev, -2.0f, 2.0f);
    input_host.copy_from(input_dev);


    // init weights Tensor
    Tensor<TargetType> weights_dev;
    Tensor<TargetType_H> weights_host;
    weights_dev.re_alloc(weights_s, AK_FLOAT);
    weights_host.re_alloc(weights_s, AK_FLOAT);
//        fill_tensor_const(weights_dev, 1.f);
    //    fill_tensor_seq(weights_dev);
    fill_tensor_rand(weights_dev, -2.0f, 2.0f);
    weights_host.copy_from(weights_dev);

    Tensor<TargetType> bias_dev;
    Tensor<TargetType_H> bias_host;

    if (bias_term) {
        bias_dev.re_alloc(bias_s, AK_FLOAT);
        bias_host.re_alloc(bias_s, AK_FLOAT);
        //        fill_tensor_const(bias_dev, 3.f);
        fill_tensor_rand(bias_dev, -2.0f, 2.0f);
        bias_host.copy_from(bias_dev);
    }

    Tensor<TargetType> output_dev(output_dev_s);
    Tensor<TargetType_H> output_host(output_dev_s);
    Tensor<TargetType_H> check_host;
    fill_tensor_rand(output_dev, -2.0f, 2.0f);
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
    //    output_dev.set_layout_without_shape(Layout_NCHW_C8);
    conv.compute_output_shape(input_v, output_v, param);
    //            LOG(INFO)<<"layout "<<output_dev.get_layout();
    //    output_dev.re_alloc(output_dev.valid_shape(), AK_FLOAT);
    //    output_dev.re_alloc(output_dev_s, AK_FLOAT);

    //            LOG(INFO)<<"layout "<<output_dev.get_layout();
    SABER_CHECK(conv.init(input_v, output_v, param, strategy, imp, ctx1));
    //            LOG(INFO)<<"layout "<<output_dev.get_layout()<<","<<output_dev.size()<<","<<output_dev.valid_size();
    SABER_CHECK(conv(input_v, output_v, param, ctx1));
    //    LOG(INFO)<<"conv finish";
    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
    //    output_v[0]->record_event(stream);
    //    output_v[0]->sync();
    //    output_host.re_alloc(output_dev.valid_shape(), AK_FLOAT);
    //    output_host.copy_from(output_dev);

    //    print_tensor(input_dev);
    //    print_tensor(output_dev);
    //    print_tensor(output_host);
    Tensor<TargetType_H> nchwc8_input_check(Shape({input_num, in_channels, height, width}));
    anakin::saber::reorder_nchwc_nchw(input_host, nchwc8_input_check);
    check_host.re_alloc(Shape({input_num, out_channels, out_height, out_width}), AK_FLOAT);
    Tensor<TargetType_H> nchw_output_check(check_host.valid_shape());
    conv_basic_check<TargetType_H>(nchwc8_input_check, check_host,
                                   (const float*)weights_host.data(), (const float*)bias_host.data(),
                                   group, kernel_w, kernel_h, stride_w, stride_h,
                                   dilation_w, dilation_h, pad_w, pad_h, bias_term,
                                   param.activation_param.has_active);
    LOG(INFO) << "cal check finish";
    //    print_tensor_valid(check_host);

    //    anakin::saber::input_reorder_nChwc8(check_host,nchw_output_check);
    Tensor<TargetType_H> nchwc8_output_check(check_host.valid_shape());
    anakin::saber::reorder_nchwc_nchw(output_dev, nchwc8_output_check);
    double max_ratio = 0.0;
    double max_diff = 0.0;
    tensor_cmp_host((const float*)nchwc8_output_check.data(), (const float*)check_host.data(),
                    check_host.valid_size(), max_ratio, max_diff);

    if (max_ratio > 1e-3 && max_diff > 1e-3) {
        print_tensor(nchwc8_output_check);
        print_tensor(check_host);
//        print_tensor(input_host);
//        print_tensor(weights_dev);
        LOG(FATAL) << " max_ratio = " << max_ratio << " max_diff = " << max_diff;
    } else {
        LOG(INFO) << "passed";
    }

    return 0;
}


template<typename TargetType, typename TargetType_H>
int test_conv_results_x86(int group,
                          int input_num, int in_channels, int height, int width,
                          int out_channels, int kernel_h, int kernel_w,
                          int stride_h, int stride_w, int dilation_h, int dilation_w,
                          int pad_h, int pad_w, bool bias_term, bool with_relu,
                          SaberImplStrategy strategy, ImplEnum imp) {

    LOG(INFO) << " conv param: "
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
    int out_height = (pad_h * 2 + height - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int out_width = (pad_w * 2 + width - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    Shape output_dev_s({input_num, (out_channels + 7) / 8, out_height, out_width, 8}, Layout_NCHW_C8);
    // init input Tensor
    Tensor<TargetType> input_dev;
    Tensor<TargetType_H> input_host;
    input_dev.re_alloc(input_s, AK_FLOAT);
    input_host.re_alloc(input_s, AK_FLOAT);
//    {
//        float *tmp= static_cast<float*>(input_dev.mutable_data());
//        for(int i=0;i<height;i++){
//            for(int j=0;j<width;j++){
//                    int index=i*width+j;
//                    tmp[index]=i*width+j;
//                }
//            }
//
//
//    }

        fill_tensor_const(input_dev, 1.f);
//    fill_tensor_rand(input_dev, -2.0f, 2.0f);
    input_host.copy_from(input_dev);


    // init weights Tensor
    Tensor<TargetType> weights_dev;
    Tensor<TargetType_H> weights_host;
    weights_dev.re_alloc(weights_s, AK_FLOAT);
    weights_host.re_alloc(weights_s, AK_FLOAT);
        fill_tensor_const(weights_dev, 1.f);
//    fill_tensor_rand(weights_dev, -2.0f, 2.0f);
    weights_host.copy_from(weights_dev);

    Tensor<TargetType> bias_dev;
    Tensor<TargetType_H> bias_host;

    if (bias_term) {
        bias_dev.re_alloc(bias_s, AK_FLOAT);
        bias_host.re_alloc(bias_s, AK_FLOAT);
        //        fill_tensor_const(bias_dev, 3.f);
        fill_tensor_rand(bias_dev, -2.0f, 2.0f);
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
    //    output_dev.set_layout_without_shape(Layout_NCHW_C8);
    conv.compute_output_shape(input_v, output_v, param);
//    LOG(INFO) << "layout " << output_dev.get_layout();
    output_dev.re_alloc(output_dev.valid_shape(), AK_FLOAT);

    //    output_dev.re_alloc(output_dev_s, AK_FLOAT);

//    LOG(INFO) << "layout " << output_dev.get_layout();
    conv.init(input_v, output_v, param, strategy, imp, ctx1);
//    LOG(INFO) << "layout " << output_dev.get_layout() << ","
// << output_dev.size() << "," <<output_dev.valid_size();
    conv(input_v, output_v, param, ctx1);
#if 0
    int epoch=1000;
    int warm_up=10;
    for (int i=0; i<warm_up; i++) {
        conv(input_v, output_v, param, ctx1);
    }
    SaberTimer<X86> x86_timer;
    x86_timer.start(ctx1);

    for (int i=0; i<epoch; i++) {
        conv(input_v, output_v, param, ctx1);
    }
    x86_timer.end(ctx1);
    double ms=x86_timer.get_average_ms();
    LOG(INFO) << "dev conv finish in "<<ms/epoch;
#endif
    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
    output_v[0]->record_event(stream);
    output_v[0]->sync();
    output_host.re_alloc(output_dev.valid_shape(), AK_FLOAT);
    output_host.copy_from(output_dev);

    //    print_tensor(input_dev);
    //    print_tensor(output_dev);
    //    print_tensor(output_host);
    check_host.re_alloc(Shape({input_num, out_channels, out_height, out_width}), AK_FLOAT);
    Tensor<TargetType_H> nchw_output_check(check_host.valid_shape());
    conv_basic_check<TargetType_H>(input_host, check_host,
                                   (const float*)weights_host.data(), (const float*)bias_host.data(),
                                   group, kernel_w, kernel_h, stride_w, stride_h,
                                   dilation_w, dilation_h, pad_w, pad_h, bias_term,
                                   param.activation_param.has_active);
    //    print_tensor_valid(check_host);

    //    anakin::saber::input_reorder_nChwc8(check_host,nchw_output_check);
    //    Tensor<TargetType_H> nchwc8_output_check(check_host.valid_shape());
    //    anakin::saber::reorder_nchwc8_nchw(output_host,nchwc8_output_check);
    double max_ratio = 0.0;
    double max_diff = 0.0;
    tensor_cmp_host((const float*)output_host.data(), (const float*)check_host.data(),
                    check_host.valid_size(), max_ratio, max_diff);

    if (max_ratio > 1e-3 && max_diff>1e-3) {
//        print_tensor(output_dev);
//        print_tensor(check_host);
//        print_tensor(input_host);
        LOG(FATAL) << " max_ratio = " << max_ratio << " max_diff = " << max_diff;
    } else {
        LOG(INFO) << "passed "<<" max_ratio = " << max_ratio << " max_diff = " << max_diff;
    }

    return 0;
}

#if defined(USE_X86_PLACE)
#include "saber/funcs/impl/x86/kernel/jit_generator.h"
#define X86_CONV_ONE_TEST 1
TEST(TestSaberFunc, test_saber_x86_conv_results) {

    Env<X86>::env_init();
    bool use_avx512=jit::mayiuse(jit::avx512_common);
    bool use_avx2=jit::mayiuse(jit::avx2);
    //#ifdef USE_OPENMP
    //    omp_set_dynamic(0);
    //    omp_set_num_threads(1);
    //#endif

    SaberImplStrategy strategy = SPECIFY;
    ImplEnum imp = SABER_IMPL;
#if X86_CONV_ONE_TEST
    int group = 1;
    int input_num = 1;
    int in_channels = 3;
    int height = 224;
    int width = 224;
    int out_channels = 64;
    int kernel_h = 3;
    int kernel_w = 3;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int pad_h = 1;
    int pad_w = 1;
    bool bias_term = true;
    bool with_relu = true;
#else

        std::vector<int> kernel_h_v{1, 3};
        std::vector<int> kernel_w_v{1, 3};
        std::vector<int> pad_h_v{0, 1};
        std::vector<int> pad_w_v{0, 1};
        std::vector<int> stride_h_v{1, 2};
        std::vector<int> stride_w_v{1, 2};
        std::vector<int> dilation_h_v{1, 2};
        std::vector<int> dilation_w_v{1, 2};
        std::vector<int> in_channels_v{16};
        std::vector<int> out_channels_v{32};
        std::vector<int> group_v{1};
        std::vector<int> in_h_v{12, 21};
        std::vector<int> in_w_v{12, 21};
        std::vector<int> input_num_v{1, 3};
        std::vector<bool> bias_term_v{true, false};
        std::vector<bool> with_relu_v{true, false};
        for (auto group : group_v) {
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
#endif
if(use_avx512) {
    for (int i = 0; i < 1; i++) {
        test_conv_results_x86_C16R<X86, X86>(group,
                                             input_num, in_channels,
                                             height, width,
                                             out_channels, kernel_h,
                                             kernel_w,
                                             stride_h, stride_w,
                                             dilation_h, dilation_w,
                                             pad_h, pad_w, bias_term,
                                             with_relu,
                                             strategy, SABER_IMPL, true, true,true);
    }
}

//
//if(use_avx2) {
//    for (int i = 0; i < 1; i++) {
//        test_conv_results_x86_C8R<X86, X86>(group,
//                                            input_num, in_channels,
//                                            height, width,
//                                            out_channels, kernel_h,
//                                            kernel_w,
//                                            stride_h, stride_w,
//                                            dilation_h, dilation_w,
//                                            pad_h, pad_w, bias_term,
//                                            with_relu,
//                                            strategy, SABER_IMPL);
//    }
//}

//    for (int i = 0; i < 1; i++) {
//        test_conv_results_x86<X86, X86>(group,
//                                        input_num,
//                                        in_channels,
//                                        height,
//                                        width,
//                                        out_channels,
//                                        kernel_h,
//                                        kernel_w,
//                                        stride_h,
//                                        stride_w,
//                                        dilation_h,
//                                        dilation_w,
//                                        pad_h,
//                                        pad_w,
//                                        bias_term,
//                                        with_relu,
//                                        strategy,
//                                        imp);
//    }

#if !X86_CONV_ONE_TEST
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

#endif

TEST(TestSaberFunc, test_saber_cuda_conv_results) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
#endif
    std::vector<int> kernel_h_v {1, 3};
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


TEST(TestSaberFunc, test_saber_arm_conv_results) {
#ifdef USE_ARM_PLACE

    Env<ARM>::env_init();
//!ToDO add set_run_mode interface

//! conv1x1s1
#if 1
    if (RUN_BASIC_TEST_ARM) {
            for (auto input_num : {1, 2}) {
            for (auto out_channels : {1, 5, 16}) {
            for (auto in_channels : {1, 3, 8}) {
            for (auto kernel_w : {1}) {
            for (auto height : {1, 3, 8, 15, 28, 32, 38, 75}) {
            for (auto stride_w : {1}) {
            for (auto dilation_w : {1}) {
            for (auto pad_w : {0}) {
            for (auto bias_term : {false, true}) {
            for (auto with_relu : {false, true}) {
            for (auto group: {1, 2, 4}){
            for (auto threads: {1, 2, 4}){
                if (in_channels % group != 0 || out_channels % group != 0) {
                  continue;
                }
                int width = height;
                test_conv_results<ARM, ARM>(  group,
                                              input_num,
                                              in_channels,
                                              height,
                                              width,
                                              out_channels,
                                              kernel_w,
                                              kernel_w,
                                              stride_w, stride_w,
                                              dilation_w, dilation_w,
                                              pad_w, pad_w, bias_term,
                                              with_relu,
                                              SPECIFY,
                                              SABER_IMPL,
                                              1e-3f,
                                              threads);
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

//! conv3x3s1(not winograd)
#if 0
    if (RUN_BASIC_TEST_ARM) {
            for (auto input_num : {1, 2}) {
            for (auto out_channels : {3, 5, 16}) {
            for (auto in_channels : {1, 3, 8}) {
            for (auto kernel_w : {3}) {
            for (auto height : {3, 4, 15, 28, 32, 38, 75, 112}) {
            for (auto stride_w : {1}) {
            for (auto dilation_w : {1}) {
            for (auto pad_w : {0, 1, 2}) {
            for (auto bias_term : {false, true}) {
            for (auto with_relu : {false, true}) {
            for (auto group: {1}){
            for (auto threads: {1, 2, 4}){
                if (in_channels % group != 0 || out_channels % group != 0) {
                  continue;
                }
                int width = height;
                test_conv_results<ARM, ARM>(  group,
                                              input_num,
                                              in_channels,
                                              height,
                                              width,
                                              out_channels,
                                              kernel_w,
                                              kernel_w,
                                              stride_w, stride_w,
                                              dilation_w, dilation_w,
                                              pad_w, pad_w, bias_term,
                                              with_relu,
                                              SPECIFY,
                                              SABER_IMPL,
                                              12-3f,
                                              threads);
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

//! conv3x3s1(winograd)
#if 0
    if (RUN_BASIC_TEST_ARM) {
            for (auto input_num : {1, 2}) {
            for (auto out_channels : {32, 64}) {
            for (auto in_channels : {32, 64}) {
            for (auto kernel_w : {3}) {
            for (auto height : {38, 75, 112}) {
            for (auto stride_w : {1}) {
            for (auto dilation_w : {1}) {
            for (auto pad_w : {0, 1, 2}) {
            for (auto bias_term : {false, true}) {
            for (auto with_relu : {false, true}) {
            for (auto group: {1}){
            for (auto threads: {1, 2, 4}){
                if (in_channels % group != 0 || out_channels % group != 0) {
                  continue;
                }
                int width = height;
                test_conv_results<ARM, ARM>(  group,
                                              input_num,
                                              in_channels,
                                              height,
                                              width,
                                              out_channels,
                                              kernel_w,
                                              kernel_w,
                                              stride_w, stride_w,
                                              dilation_w, dilation_w,
                                              pad_w, pad_w, bias_term,
                                              with_relu,
                                              SPECIFY,
                                              SABER_IMPL,
                                              1e-2f,
                                              threads);
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

//! conv3x3s2
#if 1
    if (RUN_BASIC_TEST_ARM) {
            for (auto input_num : {1, 2}) {
            for (auto out_channels : {3, 5, 16}) {
            for (auto in_channels : {1, 3, 8}) {
            for (auto kernel_w : {3}) {
            for (auto height : {7, 15, 28, 32, 38, 75, 112}) {
            for (auto stride_w : {2}) {
            for (auto dilation_w : {1}) {
            for (auto pad_w : {0, 1, 2}) {
            for (auto bias_term : {false, true}) {
            for (auto with_relu : {false, true}) {
            for (auto group: {1}){
            for (auto threads: {1, 2, 4}){
                if (in_channels % group != 0 || out_channels % group != 0) {
                  continue;
                }
                int width = height;
                test_conv_results<ARM, ARM>(  group,
                                              input_num,
                                              in_channels,
                                              height,
                                              width,
                                              out_channels,
                                              kernel_w,
                                              kernel_w,
                                              stride_w, stride_w,
                                              dilation_w, dilation_w,
                                              pad_w, pad_w, bias_term,
                                              with_relu,
                                              SPECIFY,
                                              SABER_IMPL,
                                              1e-3f,
                                              threads);
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

//! conv3x3dw
#if 1
    if (RUN_BASIC_TEST_ARM) {
            for (auto input_num : {1, 2}) {
            for (auto in_channels : {3, 5, 16}) {
            for (auto kernel_w : {3}) {
            for (auto height : {15, 28, 32, 38, 75, 112}) {
            for (auto stride_w : {1, 2}) {
            for (auto dilation_w : {1}) {
            for (auto pad_w : {0, 1}) {
            for (auto bias_term : {false, true}) {
            for (auto with_relu : {false, true}) {
            for (auto threads: {1, 2, 4}){
                int width = height;
                int out_channels = in_channels;
                int group = in_channels;
                test_conv_results<ARM, ARM>(  group,
                                              input_num,
                                              in_channels,
                                              height,
                                              width,
                                              out_channels,
                                              kernel_w,
                                              kernel_w,
                                              stride_w, stride_w,
                                              dilation_w, dilation_w,
                                              pad_w, pad_w, bias_term,
                                              with_relu,
                                              SPECIFY,
                                              SABER_IMPL,
                                              1e-3f,
                                              threads);
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

//! conv5x5s1dw
#if 0
#ifdef __aarch64__

    if (RUN_BASIC_TEST_ARM) {
            for (auto input_num : {1}) {
            for (auto in_channels : {3}) {
            for (auto kernel_w : {5}) {
            for (auto height : {15}) {
            for (auto stride_w : {1}) {
            for (auto dilation_w : {1}) {
            for (auto pad_w : {0}) {
            for (auto bias_term : {false}) {
            for (auto with_relu : {false}) {
            for (auto threads: {1, 2, 4}){
                int width = height;
                int out_channels = in_channels;
                int group = in_channels;
                test_conv_results<ARM, ARM>(  group,
                                              input_num,
                                              in_channels,
                                              height,
                                              width,
                                              out_channels,
                                              kernel_w,
                                              kernel_w,
                                              stride_w, stride_w,
                                              dilation_w, dilation_w,
                                              pad_w, pad_w, bias_term,
                                              with_relu,
                                              SPECIFY,
                                              SABER_IMPL,
                                              1e-3f,
                                              threads);
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
#endif

//! conv5x5s2p2 dw
#if 1
    if (RUN_BASIC_TEST_ARM) {
            for (auto input_num : {1, 2}) {
            for (auto in_channels : {3, 5, 16, 32}) {
            for (auto kernel_w : {5}) {
            for (auto height : {5, 15, 28, 32, 38, 75, 112}) {
            for (auto stride_w : {2}) {
            for (auto dilation_w : {1}) {
            for (auto pad_w : {2}) {
            for (auto bias_term : {false, true}) {
            for (auto with_relu : {false, true}) {
            for (auto threads: {1, 2, 4}){
                int width = height;
                int out_channels = in_channels;
                int group = in_channels;
                test_conv_results<ARM, ARM>(  group,
                                              input_num,
                                              in_channels,
                                              height,
                                              width,
                                              out_channels,
                                              kernel_w,
                                              kernel_w,
                                              stride_w, stride_w,
                                              dilation_w, dilation_w,
                                              pad_w, pad_w, bias_term,
                                              with_relu,
                                              SPECIFY,
                                              SABER_IMPL,
                                              1e-3f,
                                              threads);
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

//! otherwise conv, invoke gemm
#if 1
    if (RUN_BASIC_TEST_ARM) {
            for (auto input_num : {1, 2}) {
            for (auto out_channels : {4, 8, 16}) {
            for (auto in_channels : {1, 4, 8}) {
            for (auto kernel_w : {2, 4, 5}) {
            for (auto height : {15, 28, 32, 38, 75, 112}) {
            for (auto stride_w : {1, 2, 4}) {
            for (auto dilation_w : {1, 2}) {
            for (auto pad_w : {0, 1, 2}) {
            for (auto bias_term : {false, true}) {
            for (auto with_relu : {false, true}) {
            for (auto group: {1, 2}){
            for (auto threads: {1, 2, 4}){
                if (in_channels % group != 0 || out_channels % group != 0) {
                  continue;
                }
                int width = height;
                test_conv_results<ARM, ARM>(  group,
                                              input_num,
                                              in_channels,
                                              height,
                                              width,
                                              out_channels,
                                              kernel_w,
                                              kernel_w,
                                              stride_w, stride_w,
                                              dilation_w, dilation_w,
                                              pad_w, pad_w, bias_term,
                                              with_relu,
                                              SPECIFY,
                                              SABER_IMPL,
                                              1e-3f,
                                              threads);
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

#endif
}
int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
