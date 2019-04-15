#include "saber/core/context.h"
#include "saber/funcs/conv_pooling.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "conv_func_helper.h"
#include <vector>

using namespace anakin::saber;

template <typename dtype>
int count_diff(const dtype* src1, const dtype* src2, int size, double max_ratio) {
    if (max_ratio <= 0) {
        max_ratio = 0.1;
    }

    int count = 0;

    for (int i = 0; i < size; ++i) {
        double ratio = fabs(src1[i] - src2[i]) / fabs(src1[i] + src2[i] + 1e-12);

        if (ratio > max_ratio) {
            ++count;
        }
    }

    return count;
}

template<typename TargetType, typename TargetType_H>
int test_conv_pool_results(int group,
                           int input_num, int in_channels, int height, int width,
                           int out_channels, int conv_kernel_h, int conv_kernel_w,
                           int conv_stride_h, int conv_stride_w, int conv_dilation_h, int conv_dilation_w,
                           int conv_pad_h, int conv_pad_w, bool bias_term, bool relu,
                           int pool_stride_h, int pool_stride_w, int pool_pad_h, int pool_pad_w,
                           int pool_kernel_h, int pool_kernel_w, PoolingType pool_type,
                           SaberImplStrategy strategy, ImplEnum imp) {

    LOG(INFO) << " conv param: "
              << " input_num = " << input_num
              << " in_channels = " << in_channels
              << " height = " << height
              << " width = " << width
              << " group = " << group
              << " conv_pad_h = " << conv_pad_h
              << " conv_pad_w = " << conv_pad_w
              << " conv_stride_h = " << conv_stride_h
              << " conv_stride_w = " << conv_stride_w
              << " conv_dilation_h = " << conv_dilation_h
              << " conv_dilation_w = " << conv_dilation_w
              << " conv_kernel_h = " << conv_kernel_h
              << " conv_kernel_w = " << conv_kernel_w
              << " pool_pad_h = " << pool_pad_h
              << " pool_pad_w = " << pool_pad_w
              << " pool_stride_h = " << pool_stride_h
              << " pool_stride_w = " << pool_stride_w
              << " pool_kernel_h = " << pool_kernel_h
              << " pool_kernel_w = " << pool_kernel_w
              << " out_channels = " << out_channels
              << " relu = " << (relu ? "true" : "false")
              << " bias_term = " << (bias_term ? "true" : "false");

#ifdef USE_CUDA
    return 0;
#endif
#ifdef USE_X86_PLACE
    Shape input_s({input_num, height, width, in_channels}, Layout_NHWC);
    Shape weights_s({out_channels, in_channels, conv_kernel_h, conv_kernel_w}, Layout_NCHW);
    Shape weights_s_dw({group, in_channels / group, conv_kernel_h, conv_kernel_w}, Layout_NCHW);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);

    // generate conv_output shape
    int conv_out_height = (conv_pad_h * 2 + height - (conv_dilation_h * (conv_kernel_h - 1) + 1)) /
                          conv_stride_h + 1;
    int conv_out_width = (conv_pad_w * 2 + width - (conv_dilation_w * (conv_kernel_w - 1) + 1)) /
                         conv_stride_w + 1;
    Shape conv_output_s({input_num, conv_out_height, conv_out_width, out_channels}, Layout_NHWC);

    // generate conv_pool_output shape
    int out_height = (conv_out_height + 2 * pool_pad_h - pool_kernel_h) / pool_stride_h + 1;
    int out_width = (conv_out_width + 2 * pool_pad_w - pool_kernel_w) / pool_stride_w + 1;
    Shape output_s({input_num, out_height, out_width, out_channels}, Layout_NHWC);

    // init input Tensor
    Tensor<TargetType> input_dev;
    Tensor<TargetType_H> input_host;
    input_dev.re_alloc(input_s, AK_UINT8);
    input_host.re_alloc(input_s, AK_UINT8);
    fill_tensor_rand(input_dev, 0.0f, 32.0f);
    input_host.copy_from(input_dev);
    input_dev.set_scale({1 / 512.f});

    // init weights Tensor
    Tensor<TargetType> weights_dev;
    Tensor<TargetType_H> weights_host;

    if (group > 1) {
        weights_dev.re_alloc(weights_s_dw, AK_INT8);
        weights_host.re_alloc(weights_s_dw, AK_INT8);
    } else {
        weights_dev.re_alloc(weights_s, AK_INT8);
        weights_host.re_alloc(weights_s, AK_INT8);
    }

    fill_tensor_rand(weights_dev, -64.0f, 64.0f);
    weights_host.copy_from(weights_dev);
    std::vector<float> scale_w_init;

    for (int i = 0; i < out_channels; i ++) {
        scale_w_init.push_back(1 / 128.f);
    }

    weights_dev.set_scale(scale_w_init);

    Tensor<TargetType> bias_dev;
    Tensor<TargetType_H> bias_host;

    if (bias_term) {
        bias_dev.re_alloc(bias_s, AK_INT32);
        bias_host.re_alloc(bias_s, AK_INT32);
        fill_tensor_rand(bias_dev, -1.0f, 1.0f);
        bias_host.copy_from(bias_dev);
    }

    Tensor<TargetType_H> check_host;

    Context<TargetType> ctx1(0, 1, 1);
    ActivationParam<TargetType> act_param;

    if (relu) {
        ActivationParam<TargetType> act_relu_param(Active_relu);
        act_param = act_relu_param;
    }

    ConvParam<TargetType> conv_param(group, conv_pad_h, conv_pad_w,
                                     conv_stride_h, conv_stride_w,
                                     conv_dilation_h, conv_dilation_w,
                                     &weights_dev, bias_term ? &bias_dev : nullptr,
                                     act_param, 1.f, 0.f,AK_UINT8, round_mode::nearest);

    PoolingParam<TargetType> pool_param(pool_kernel_h, pool_kernel_w,
                                        pool_pad_h, pool_pad_w, pool_stride_h, pool_stride_w,
                                        pool_type);
    ConvPoolingParam<TargetType> param(conv_param, pool_param);
    // init output Tensor
    Tensor<TargetType> output_dev;
    Tensor<TargetType_H> output_host;
    Tensor<TargetType_H> conv_output_host;

    if (conv_param.activation_param.has_active) {
        output_dev.re_alloc(output_s, AK_UINT8);
        conv_output_host.re_alloc(conv_output_s, AK_UINT8);
        output_host.re_alloc(output_s, AK_UINT8);
        output_dev.set_scale({1 / 256.0f});
        conv_output_host.set_scale({1 / 256.0f});
    } else {
        output_dev.re_alloc(output_s, AK_INT8);
        conv_output_host.re_alloc(conv_output_s, AK_INT8);
        output_host.re_alloc(output_s, AK_INT8);
        output_dev.set_scale({1 / 128.0f});
        conv_output_host.set_scale({1 / 128.0f});
    }

    output_host.copy_from(output_dev);

    ConvPooling<TargetType, AK_INT8> conv_pooling;
    std::vector<Tensor<TargetType>* > input_v;
    std::vector<Tensor<TargetType>* > output_v;
    input_v.push_back(&input_dev);
    output_v.push_back(&output_dev);
    // conv.compute_output_shape(input_v, output_v, param);
    // output_dev.re_alloc(output_dev.valid_shape(), AK_INT8);

    if (conv_pooling.init(input_v, output_v, param, strategy, imp, ctx1) == SaberSuccess) {
        conv_pooling(input_v, output_v, param, ctx1);
    } else {
        LOG(INFO) << "init return non Success!";
        return -1;
    }

    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
    output_v[0]->record_event(stream);
    output_v[0]->sync();

    if (conv_param.activation_param.has_active) {
        output_host.re_alloc(output_dev.valid_shape(), AK_UINT8);
        output_host.copy_from(output_dev);
        // print_tensor_valid(output_host);
        check_host.re_alloc(output_host.valid_shape(), AK_UINT8);
    } else {
        output_host.re_alloc(output_dev.valid_shape(), AK_INT8);
        output_host.copy_from(output_dev);
        check_host.re_alloc(output_host.valid_shape(), AK_INT8);
    }

    // calc scale info
    std::vector<float> scale;
    float scale_in = input_dev.get_scale()[0];
    float scale_out = output_dev.get_scale()[0];
    auto scale_w = weights_dev.get_scale();
    std::vector<float>().swap(scale);

    for (int i = 0; i < scale_w.size(); i++) {
        scale.push_back((scale_w[i] * scale_in) / scale_out);
    }

    conv_basic_check_int8<X86>(input_host, conv_output_host,
                               (const char*)weights_host.data(), bias_term ? (const int*)bias_host.data() : nullptr,
                               group, conv_kernel_w, conv_kernel_h, conv_stride_w, conv_stride_h,
                               conv_dilation_w, conv_dilation_h, conv_pad_w, conv_pad_h, bias_term,
                               conv_param.activation_param.has_active, scale);
    pool_basic_check_int8(conv_output_host, check_host, pool_kernel_w, pool_kernel_h, pool_stride_w,
                          pool_stride_h,
                          pool_pad_w, pool_pad_h, pool_type);
    int count = count_diff((const unsigned char*)output_host.data(),
                           (const unsigned char*)check_host.data(), check_host.valid_size(), 2e-1);

    // print_tensor_valid(check_host);
    // double max_ratio = 0.0;
    // double max_diff = 0.0;
    // tensor_cmp_host((const float*)output_host.data(), (const float*)check_host.data(),
    //                check_host.valid_size(), max_ratio, max_diff);
    if ((double)count / output_host.valid_size() < 0.02) {
        // LOG(INFO) << " PASS!!! max_ratio = " << max_ratio << " max_diff = " << max_diff;
        LOG(INFO) << "PASS!!! count = " << count;
        return 0;
    } else {
        print_tensor_valid(output_host);
        print_tensor_valid(check_host);
        // LOG(FATAL) << "FAIL!!! max_ratio = " << max_ratio << " max_diff = " << max_diff

        LOG(FATAL) << "FAIL!!! count = " << count
                   << " conv param: "
                   << " input_num = " << input_num
                   << " in_channels = " << in_channels
                   << " height = " << height
                   << " width = " << width
                   << " group = " << group
                   << " conv_pad_h = " << conv_pad_h
                   << " conv_pad_w = " << conv_pad_w
                   << " conv_stride_h = " << conv_stride_h
                   << " conv_stride_w = " << conv_stride_w
                   << " conv_dilation_h = " << conv_dilation_h
                   << " conv_dilation_w = " << conv_dilation_w
                   << " conv_kernel_h = " << conv_kernel_h
                   << " conv_kernel_w = " << conv_kernel_w
                   << " pool_pad_h = " << pool_pad_h
                   << " pool_pad_w = " << pool_pad_w
                   << " pool_stride_h = " << pool_stride_h
                   << " pool_stride_w = " << pool_stride_w
                   << " pool_kernel_h = " << pool_kernel_h
                   << " pool_kernel_w = " << pool_kernel_w
                   << " out_channels = " << out_channels
                   << " relu = " << (relu ? "true" : "false")
                   << " bias_term = " << (bias_term ? "true" : "false");
        return -1;
    }

#endif
}

TEST(TestSaberFunc, test_saber_conv_int8_results) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
#endif
    std::vector<int> groups{1};
    std::vector<int> conv_kernel_h_v{3};
    std::vector<int> conv_kernel_w_v{3};
    std::vector<int> conv_pad_h_v{0};
    std::vector<int> conv_pad_w_v{0};
    std::vector<int> conv_stride_h_v{1};
    std::vector<int> conv_stride_w_v{1};
    std::vector<int> conv_dilation_h_v{1};
    std::vector<int> conv_dilation_w_v{1};
    std::vector<int> pool_kernel_h_v{2, 3};
    std::vector<int> pool_kernel_w_v{2, 3};
    std::vector<int> pool_pad_h_v{0};
    std::vector<int> pool_pad_w_v{0};
    std::vector<int> pool_stride_h_v{2, 3};
    std::vector<int> pool_stride_w_v{2, 3};
    std::vector<PoolingType> pool_type_v{Pooling_max};
    std::vector<int> in_channels_v{16};
    std::vector<int> out_channels_v{16};
    std::vector<int> in_h_v{32};
    std::vector<int> in_w_v{32};
    std::vector<int> input_num_v{1};
    std::vector<bool> bias_term_v{true};
    std::vector<bool> relu_v{true};

    for (auto group : groups) {
    for (auto input_num : input_num_v) {
    for (auto out_channels : out_channels_v) {
    for (auto in_channels : in_channels_v) {
    for (auto conv_kernel_h : conv_kernel_h_v) {
    for (auto conv_kernel_w : conv_kernel_w_v) {
    for (auto height : in_h_v) {
    for (auto width : in_w_v) {
    for (auto conv_stride_h : conv_stride_h_v) {
    for (auto conv_stride_w : conv_stride_w_v) {
    for (auto conv_dilation_h : conv_dilation_h_v) {
    for (auto conv_dilation_w : conv_dilation_w_v) {
    for (auto conv_pad_h : conv_pad_h_v) {
    for (auto conv_pad_w : conv_pad_w_v) {
    for (auto pool_kernel_h : pool_kernel_h_v) {
    for (auto pool_kernel_w : pool_kernel_w_v) {
    for (auto pool_stride_h : pool_stride_h_v) {
    for (auto pool_stride_w : pool_stride_w_v) {
    for (auto pool_pad_h : pool_pad_h_v) {
    for (auto pool_pad_w : pool_pad_w_v) {
    for (auto pool_type : pool_type_v) {
    for (auto bias_term : bias_term_v) {
    for (auto relu : relu_v) {
    #ifdef USE_CUDA
    #endif
    #ifdef USE_X86_PLACE

        if (jit::mayiuse(
                jit::avx512_core)&&jit::mayiuse(
                jit::avx512_core_vnni)) {
            test_conv_pool_results<X86, X86>(
                    group,
                    input_num,
                    in_channels,
                    height,
                    width,
                    out_channels,
                    conv_kernel_h,
                    conv_kernel_w,
                    conv_stride_h,
                    conv_stride_w,
                    conv_dilation_h,
                    conv_dilation_w,
                    conv_pad_h,
                    conv_pad_w,
                    bias_term,
                    relu,
                    pool_stride_h,
                    pool_stride_w,
                    pool_pad_h,
                    pool_pad_w,
                    pool_kernel_h,
                    pool_kernel_w,
                    pool_type,
                    SPECIFY,
                    SABER_IMPL);
        }


    #endif
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
    }
    }
    }
    }
    }
    }
    }

}


int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
//    InitTest();
//    RUN_ALL_TESTS(argv[0]);
    return 0;
}
