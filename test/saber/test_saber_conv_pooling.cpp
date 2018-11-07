#include "saber/core/context.h"
#include "saber/funcs/conv_pooling.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "conv_func_helper.h"
#include "saber/funcs/funcs_utils.h"

#include <vector>

using namespace anakin::saber;

template<typename dtype, typename TargetType_D, typename TargetType_H>
void pooling_cpu_func(const std::vector<Tensor<TargetType_H>*>& input,
                      std::vector<Tensor<TargetType_H>*>& output,
                      PoolingParam<TargetType_D>& param) {
    const dtype* src_ptr = static_cast<dtype*>(input[0]->data());
    dtype* dst_ptr = static_cast<dtype*>(output[0]->mutable_data());

    int in_n = input[0]->num();
    int in_c = input[0]->channel();
    int in_h = input[0]->height();
    int in_w = input[0]->width();
    int size_in_n = in_c * in_h * in_w;
    int size_in_c = in_h * in_w;

    int out_h = output[0]->height();
    int out_w = output[0]->width();
    int size_out_n = in_c * out_h * out_w;
    int size_out_c = out_h * out_w;

    for (int ind_n = 0; ind_n < in_n; ++ind_n) {
        for (int ind_c = 0; ind_c < in_c; ++ind_c) {
            for (int ind_h = 0; ind_h < out_h; ++ind_h) {
                int sh = ind_h * param.stride_h;
                int eh = sh + param.window_h;

                sh = (sh - param.pad_h) < 0 ? 0 : sh - param.pad_h;
                eh = (eh - param.pad_h) > in_h ? in_h : eh - param.pad_h;

                for (int ind_w = 0; ind_w < out_w; ++ind_w) {
                    int sw = ind_w * param.stride_w;
                    int ew = sw + param.window_w;
                    sw = (sw - param.pad_w) < 0 ? 0 : sw - param.pad_w;
                    ew = (ew - param.pad_w) > in_w ? in_w : ew - param.pad_w;

                    dtype result;

                    int dst_ind = ind_n * size_out_n + ind_c * size_out_c + ind_h * out_w + ind_w;

                    for (int kh = sh; kh < eh; ++kh) {
                        for (int kw = sw; kw < ew; ++kw) {
                            int src_ind = ind_n * size_in_n + ind_c * size_in_c + kh * in_w + kw;

                            if (kh == sh && kw == sw) {
                                result = src_ptr[src_ind];
                            } else {
                                if (param.pooling_type == Pooling_max) {
                                    result = result >= src_ptr[src_ind] ? result : src_ptr[src_ind];
                                }

                                if (param.pooling_type == Pooling_average_include_padding) {
                                    result += src_ptr[src_ind];
                                }

                                if (param.pooling_type == Pooling_average_exclude_padding) {
                                    result += src_ptr[src_ind];
                                }
                            }

                        }
                    }

                    if (param.pooling_type == Pooling_average_include_padding) {
                        result /= param.window_h * param.window_w;
                    }

                    if (param.pooling_type == Pooling_average_exclude_padding) {
                        result /= (ew - sw) * (eh - sh);
                    }

                    dst_ptr[dst_ind] = result;

                }
            }
        }

    }
}

template<typename dtype, typename TargetType_D, typename TargetType_H>
void conv_pool_cpu_func(const std::vector<Tensor<TargetType_H>*>& input,
                        std::vector<Tensor<TargetType_H>*>& output,
                        ConvPoolingParam<TargetType_D>& param) {

    int group = param.conv_param.group;
    int input_num = input[0]->num();
    int input_channel = input[0]->channel();
    int input_height = input[0]->height();
    int input_width = input[0]->width();
    int output_channel = output[0]->channel();
    int output_height = output[0]->height();
    int output_width = output[0]->width();
    int stride_h = param.conv_param.stride_h;
    int stride_w = param.conv_param.stride_w;
    int dilation_h = param.conv_param.dilation_h;
    int dilation_w = param.conv_param.dilation_w;
    int pad_h = param.conv_param.pad_h;
    int pad_w = param.conv_param.pad_w;
    int kernel_h = param.conv_param.weight()->height();
    int kernel_w = param.conv_param.weight()->width();
    bool bias_term = param.conv_param.bias()->valid_size() > 0;
    bool with_relu = param.conv_param.activation_param.has_active;

    Tensor<TargetType_H> weights_host;
    Tensor<TargetType_H> bias_host;
    weights_host.re_alloc(param.conv_param.weight()->valid_shape(), AK_FLOAT);
    weights_host.copy_from(*(param.conv_param.weight()));
    bias_host.re_alloc(param.conv_param.bias()->valid_shape(), AK_FLOAT);
    bias_host.copy_from(*(param.conv_param.bias()));
    const dtype* bias_ptr = bias_term ? (const float*)bias_host.data() : nullptr;
    Shape conv_shape = conv_compute_shape(input[0]->valid_shape(), param.conv_param);
    Tensor<TargetType_H> conv_output;
    conv_output.re_alloc(conv_shape, AK_FLOAT);
    std::vector<Tensor<TargetType_H>* > conv_output_v;
    conv_output_v.push_back(&conv_output);
    conv_basic_check<TargetType_H>(*input[0], conv_output,
                                   (const dtype*)weights_host.data(), bias_ptr,
                                   group, kernel_w, kernel_h, stride_w, stride_h,
                                   dilation_w, dilation_h, pad_w, pad_h, bias_term,
                                   with_relu);
    pooling_cpu_func<float, TargetType_D, TargetType_H>(conv_output_v, output, param.pooling_param);
}


template <typename TargetType, typename TargetType_H>
void test_conv_pool() {
    Env<TargetType>::env_init();
    Env<TargetType_H>::env_init();
    TestSaberBase<TargetType, TargetType_H, AK_FLOAT, ConvPooling, ConvPoolingParam> testbase;

    std::vector<int> kernel_h_v {3};
    std::vector<int> kernel_w_v {3};
    std::vector<int> pad_h_v {1};
    std::vector<int> pad_w_v {1};
    std::vector<int> stride_h_v {1};
    std::vector<int> stride_w_v {1};
    std::vector<int> dilation_h_v {1};
    std::vector<int> dilation_w_v {1};
    std::vector<int> group_v {1};
    std::vector<int> in_h_v {12, 16};
    std::vector<int> in_w_v {16, 18};
    std::vector<int> input_num_v {1, 3};
    std::vector<int> input_channels_v {17, 4};
    std::vector<int> output_channels_v {4, 17};
    std::vector<bool> bias_term_v {true, false};

    for (auto kernel_h : kernel_h_v)
        for (auto kernel_w : kernel_w_v)
            for (auto pad_h : pad_h_v)
                for (auto pad_w : pad_w_v)
                    for (auto stride_h : stride_h_v)
                        for (auto stride_w : stride_w_v)
                            for (auto dilation_h : dilation_h_v)
                                for (auto dilation_w : dilation_w_v)
                                    for (auto bias_term : bias_term_v)
                                        for (auto in_channels : input_channels_v)
                                            for (auto out_channels : output_channels_v)
                                                for (auto group : group_v) {

                                                    Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
                                                    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);
                                                    Tensor<TargetType> weights_dev;
                                                    Tensor<TargetType> bias_dev;

                                                    weights_dev.re_alloc(weights_s, AK_FLOAT);
                                                    fill_tensor_rand(weights_dev, -5.f, 5.0f);

                                                    if (bias_term) {
                                                        bias_dev.re_alloc(bias_s, AK_FLOAT);
                                                        fill_tensor_rand(bias_dev, -5.0f, 5.0f);
                                                    }

                                                    ConvParam<TargetType> conv_param(group, pad_h, pad_w,
                                                                                     stride_h, stride_w,
                                                                                     dilation_h, dilation_w,
                                                                                     &weights_dev, &bias_dev);
                                                    PoolingParam<TargetType> pool_param(2, 2, 0, 0, 2, 2, Pooling_max);
                                                    ConvPoolingParam<TargetType> param(conv_param, pool_param);

                                                    for (auto input_num : input_num_v)
                                                        for (auto height : in_h_v)
                                                            for (auto width : in_w_v) {
                                                                // open random fill output!
                                                                testbase.set_random_output(true);
                                                                testbase.set_param(param);//set param
                                                                testbase.set_input_shape(Shape({input_num, in_channels, height, width},
                                                                                               Layout_NCHW));//add some input shape
                                                                testbase.run_test(conv_pool_cpu_func<float, TargetType, TargetType_H>, 1e-3);//run test
                                                            }
                                                }
}

TEST(TestSaberFunc, test_saber_conv_results) {

#ifdef USE_X86_PLACE
test_conv_pool<X86, X86>();
#endif

#ifdef AMD_GPU
test_conv_pool<AMD, AMDHX86>();
#endif

}





int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
