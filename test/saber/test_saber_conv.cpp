#include "saber/core/context.h"
#include "saber/funcs/conv.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "conv_func_helper.h"
#include <vector>

using namespace anakin::saber;

template<typename TargetType, typename TargetType_H>
int test_conv_results(int group,
                       int input_num, int in_channels, int height, int width,
                       int out_channels, int kernel_h, int kernel_w,
                       int stride_h, int stride_w, int dilation_h, int dilation_w,
                       int pad_h, int pad_w, bool bias_term,
                       SaberImplStrategy strategy, ImplEnum imp) {

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
                             dilation_w, dilation_h, pad_w, pad_h, bias_term, false);
    double max_ratio = 0.0;
    double max_diff = 0.0;
    tensor_cmp_host((const float*)output_host.data(), (const float*)check_host.data(),
                    check_host.valid_size(), max_ratio, max_diff);
    if (max_ratio < 1e-5) {
        LOG(INFO) << " PASS!!! max_ratio = " << max_ratio << " max_diff = " << max_diff;
        return 0;
    } else {
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
                << " out_channels = " << out_channels;
        return -1;
    }

}

template<typename TargetType, typename TargetType_H>
void test_conv_speed(int group,
                     int input_num, int in_channels, int height, int width,
                     int out_channels, int kernel_h, int kernel_w,
                     int stride_h, int stride_w, int dilation_h, int dilation_w,
                     int pad_h, int pad_w, bool bias_term,
                     SaberImplStrategy strategy, ImplEnum imp) {

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

    conv.init(input_v, output_v, param, strategy, imp, ctx1);
    conv(input_v, output_v, param, ctx1);
    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
    output_v[0]->record_event(stream);
    output_v[0]->sync();

    SaberTimer<TargetType> timer;
    int ts = 100;
    for (int i = 0; i < ts; ++i) {
        timer.start(ctx1);
        conv(input_v, output_v, param, ctx1);
        output_v[0]->record_event(stream);
        output_v[0]->sync();
        timer.end(ctx1);
    }
    LOG(INFO)  << " conv param: "
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
        << " average time: "<< timer.get_average_ms()<< " ms";
}

template<typename TargetType, typename TargetType_H>
void test_conv_ab_test(int group,
                       int input_num, int in_channels, int height1, int width1,
                       int height2, int width2,
                       int out_channels, int kernel_h, int kernel_w,
                       int stride_h, int stride_w, int dilation_h, int dilation_w,
                       int pad_h, int pad_w, bool bias_term,
                       SaberImplStrategy strategy, ImplEnum imp) {

    Shape input_s1({input_num, in_channels, height1, width1}, Layout_NCHW);
    Shape input_s2({input_num, in_channels, height2, width2}, Layout_NCHW);

    Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);

    // init input Tensor
    std::vector<Tensor<TargetType> > input_dev_ab(2);
    std::vector<Tensor<TargetType_H> > input_host_ab(2);
    input_dev_ab[0].re_alloc(input_s1, AK_FLOAT);
    input_host_ab[0].re_alloc(input_s1, AK_FLOAT);
    fill_tensor_rand(input_dev_ab[0], -10.0f, 10.0f);
    input_host_ab[0].copy_from(input_dev_ab[0]);

    input_dev_ab[1].re_alloc(input_s2, AK_FLOAT);
    input_host_ab[1].re_alloc(input_s2, AK_FLOAT);
    fill_tensor_rand(input_dev_ab[1], -10.0f, 10.0f);
    input_host_ab[1].copy_from(input_dev_ab[1]);

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

    ConvParam<TargetType> param(group, pad_h, pad_w,
                                stride_h, stride_w,
                                dilation_h, dilation_w,
                                &weights_dev, &bias_dev);

    Conv<TargetType, AK_FLOAT> conv;
    std::vector<Tensor<TargetType>* > input_v;
    std::vector<Tensor<TargetType>* > output_v;

    Tensor<TargetType> input_dev;
    input_dev.re_alloc(input_s1, AK_FLOAT);
    input_dev.copy_from(input_dev_ab[0]);

    input_v.push_back(&input_dev);
    output_v.push_back(&output_dev);
    conv.compute_output_shape(input_v, output_v, param);
    output_dev.re_alloc(output_dev.valid_shape(), AK_FLOAT);
    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
    conv.init(input_v, output_v, param, strategy, imp, ctx1);
    for (int i = 0; i < 6; ++i) {

        input_v[0]->reshape(input_dev_ab[i & 0x01].valid_shape());
        input_v[0]->copy_from(input_dev_ab[i & 0x01]);
        conv(input_v, output_v, param, ctx1);
        output_v[0]->record_event(stream);
        output_v[0]->sync();

        output_host.re_alloc(output_dev.valid_shape(), AK_FLOAT);
        output_host.copy_from(output_dev);
        check_host.re_alloc(output_host.valid_shape(), AK_FLOAT);

        conv_basic_check<TargetType_H>(input_host_ab[i & 0x01], check_host,
                                       (const float *) weights_host.data(), (const float *) bias_host.data(),
                                       group, kernel_w, kernel_h, stride_w, stride_h,
                                       dilation_w, dilation_h, pad_w, pad_h, bias_term, false);
        double max_ratio = 0.0;
        double max_diff = 0.0;
        tensor_cmp_host((const float *) output_host.data(), (const float *) check_host.data(),
                        check_host.valid_size(), max_ratio, max_diff);

        if (max_ratio < 1e-5) {
            LOG(INFO) << (i & 0x01) <<" PASS!!! max_ratio = " << max_ratio << " max_diff = " << max_diff;
        } else {
//            print_tensor_valid(output_host);
            print_tensor_valid(check_host);
            LOG(FATAL) << " FAIL in ab test!!! max_ratio = " << max_ratio << " max_diff = " << max_diff
                       << " conv param: "
                       << " input_num = " << input_num
                       << " in_channels = " << in_channels
                       << " height1 = " << height1
                       << " width1 = " << width1
                       << " height2 = " << height2
                       << " width2 = " << width2
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
                       << " impl: " << ((imp == VENDER_IMPL) ? " VENDER " : " SABER");

        }
    }
}

TEST(TestSaberFunc, test_saber_conv_results) {
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
    std::vector<int> dilation_h_v{1, 2};
    std::vector<int> dilation_w_v{1, 2};
    std::vector<int> in_channels_v{3, 32};
    std::vector<int> out_channels_v{32, 57};
    std::vector<int> group_v{1, 2, 32};
    std::vector<int> in_h_v{17, 32};
    std::vector<int> in_w_v{17, 32};
    std::vector<int> input_num_v{3, 1};
    std::vector<bool> bias_term_v{true, false};

#pragma omp parallel for num_threads(8) collapse(3) schedule(dynamic)
    for (int input_num_i = 0; input_num_i < input_num_v.size(); input_num_i++)
    for (int out_channels_i = 0; out_channels_i < out_channels_v.size(); out_channels_i++)
    for (int in_channels_i = 0; in_channels_i < in_channels_v.size(); in_channels_i++)
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
    for (auto group : group_v) {
        int input_num = input_num_v[input_num_i];
        int out_channels = out_channels_v[out_channels_i];
        int in_channels = in_channels_v[in_channels_i];
        if (in_channels % group != 0) {
            continue;
        }
        if (out_channels % group != 0) {
            continue;
        }
#ifdef USE_CUDA
        if (test_conv_results<NV, NVHX86>(group, input_num, in_channels,
                                    height, width, out_channels, kernel_h,
                                    kernel_w, stride_h, stride_w, dilation_h, dilation_w,
                                    pad_h, pad_w, bias_term, SPECIFY, VENDER_IMPL) !=0)  {
                                        LOG(INFO) << "cudnn results error!";
                                    }
        if (group == 1) {
            if ((kernel_h != 3) || (kernel_w != 3) ||
                (kernel_h ==3 && kernel_w == 3 && dilation_h == 1 && dilation_w == 1)) {
                if (test_conv_results<NV, NVHX86>(group, input_num, in_channels,
                                    height, width, out_channels, kernel_h,
                                    kernel_w, stride_h, stride_w, dilation_h, dilation_w,
                                    pad_h, pad_w, bias_term, SPECIFY, SABER_IMPL) != 0) {
                                        LOG(INFO) << " saber results error!";
                                    }
            }
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

TEST(TestSaberFunc, test_saber_conv_speed) {

    std::vector<int> kernel_h_v{1, 3};
    std::vector<int> kernel_w_v{1, 3};
    std::vector<int> pad_h_v{0, 1};
    std::vector<int> pad_w_v{0, 1};
    std::vector<int> stride_h_v{1, 2};
    std::vector<int> stride_w_v{1, 2};
    std::vector<int> dilation_h_v{1, 2};
    std::vector<int> dilation_w_v{1, 2};
    std::vector<int> in_channels_v{3, 32};
    std::vector<int> out_channels_v{32, 57};
    std::vector<int> group_v{1, 2, 32};
    std::vector<int> in_h_v{17, 32};
    std::vector<int> in_w_v{17, 32};
    std::vector<int> input_num_v{1, 3};
    std::vector<bool> bias_term_v{true, false};
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
#endif

    for (auto input_num : input_num_v)
    for (auto out_channels : out_channels_v)
    for (auto in_channels : in_channels_v)
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
    for (auto group : group_v) {
        if (in_channels % group != 0) {
            continue;
        }
        if (out_channels % group != 0) {
            continue;
        }
#ifdef USE_CUDA
        test_conv_speed<NV, NVHX86>(group, input_num, in_channels,
                                    height, width, out_channels, kernel_h,
                                    kernel_w, stride_h, stride_w, dilation_h, dilation_w,
                                    pad_h, pad_w, bias_term, SPECIFY, VENDER_IMPL);
        if (group == 1) {
            if ((kernel_h != 3) || (kernel_w != 3) ||
                (kernel_h ==3 && kernel_w == 3 && dilation_h == 1 && dilation_w == 1)) {
                test_conv_speed<NV, NVHX86>(group, input_num, in_channels,
                                    height, width, out_channels, kernel_h,
                                    kernel_w, stride_h, stride_w, dilation_h, dilation_w,
                                    pad_h, pad_w, bias_term, SPECIFY, SABER_IMPL);
            }
        }
#endif
//#ifdef USE_X86_PLACE
//        test_conv_speed<X86, X86>(group,
//                                    input_num, in_channels,
//                                    height,
//                                    width,
//                                    out_channels, kernel_h,
//                                    kernel_w,
//                                    stride_h, stride_w,
//                                    dilation_h,
//                                    dilation_w,
//                                    pad_h, pad_w, bias_term,
//                                    SPECIFY,
//                                    VENDER_IMPL);
//#endif
    }

}

TEST(TestSaberFunc, test_saber_conv_op_func) {
    std::vector<int> kernel_h_v{1, 3};
    std::vector<int> kernel_w_v{1, 3};
    std::vector<int> in_channels_v{5, 16};
    std::vector<int> out_channels_v{7, 16};
    std::vector<int> in_h_v{117, 224};
    std::vector<int> in_w_v{117, 224};
    std::vector<int> input_num_v{3, 1};
    std::vector<bool> bias_term_v{true, false};
    std::vector<int> in_h2_v{48, 67};
    std::vector<int> in_w2_v{74, 35};
    std::vector<int> group_v{1};
#pragma omp parallel for num_threads(4) collapse(2) schedule(dynamic)
    for (int input_num_i = 0; input_num_i < input_num_v.size(); input_num_i++)
    for (int out_channels_i = 0; out_channels_i < out_channels_v.size(); out_channels_i++)
    for (auto in_channels : in_channels_v)
    for (auto kernel_h : kernel_h_v)
    for (auto kernel_w : kernel_w_v)
    for (auto height : in_h_v)
    for (auto width : in_w_v)
    for (auto height2 : in_h2_v)
    for (auto width2 : in_w2_v)
    for (auto bias_term : bias_term_v)
    for (auto group : group_v) {
        int input_num = input_num_v[input_num_i];
        int out_channels = out_channels_v[out_channels_i];
        if (in_channels % group != 0) {
            continue;
        }
        if (out_channels % group != 0) {
            continue;
        }
        int pad_h = 1;
        int pad_w = 1;
#ifdef USE_CUDA
        test_conv_ab_test<NV, NVHX86>(group, input_num, in_channels,
                                    height, width, height2, width2, out_channels, kernel_h,
                                    kernel_w, 1, 1, 1, 1,
                                    pad_h, pad_w, bias_term, SPECIFY, VENDER_IMPL);
        if (group == 1) {
            test_conv_ab_test<NV, NVHX86>(group, input_num, in_channels,
                                          height, width, height2, width2, out_channels, kernel_h,
                                          kernel_w, 1, 1, 1, 1,
                                          pad_h, pad_w, bias_term, SPECIFY, SABER_IMPL);
        }
#endif
    }
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}