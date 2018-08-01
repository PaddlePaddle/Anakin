#include "saber/core/context.h"
#include "saber/funcs/conv.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func_x86.h"
#include <vector>
#include "debug.h"
using namespace anakin::saber;

template<typename targetType>
static void conv_basic_check(Tensor<targetType,AK_FLOAT> &tensor_in,Tensor<targetType,AK_FLOAT> &tensor_out,
                      const float *weights, const float *bias, int group,
                      int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h,
                      int pad_w, int pad_h, bool flag_bias, bool flag_relu) {

    auto src_data = reinterpret_cast<const float*>(tensor_in.data());
    auto dst_data_ref = reinterpret_cast<float*>(tensor_out.mutable_data());
    auto weights_data = weights;
    bool with_bias = flag_bias;
    auto bias_data = bias;

    int in_num = tensor_out.num();
    int out_channels = tensor_out.channel();
    int out_h = tensor_out.height();
    int out_w = tensor_out.width();

    int in_channel = tensor_in.channel();
    int in_h = tensor_in.height();
    int in_w = tensor_in.width();
    int out_c_group = out_channels / group;
    int in_c_group = in_channel / group;

    for (int n = 0; n < in_num; ++n) {
        for (int g = 0; g < group; ++g) {
            for (int oc = 0; oc < out_c_group; ++oc) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        int out_idx = n * group * out_c_group * out_h * out_w + g * out_c_group * out_h * out_w
                                      + oc * out_h * out_w + oh * out_w + ow;
                        dst_data_ref[out_idx] = with_bias ? (float)(bias_data[g * out_c_group + oc]) : 0.f;

                        for (int ic = 0; ic < in_c_group; ++ic) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int iw = ow * stride_w - pad_w + kw * (dilation_w);
                                    int ih = oh * stride_h - pad_h + kh * (dilation_h);
                                    if (iw < 0 || iw >= in_w) continue;
                                    if (ih < 0 || ih >= in_h) continue;

                                    int iidx = n * in_channel * in_h * in_w
                                               + g * in_c_group * in_h * in_w
                                               + ic * in_h * in_w
                                               + ih * in_w
                                               + iw;
                                    int widx = g * out_c_group * in_c_group * kernel_h * kernel_w
                                               + oc * in_c_group * kernel_h * kernel_w
                                               + ic * kernel_h * kernel_w
                                               + kh * kernel_w
                                               + kw;

                                    dst_data_ref[out_idx]
                                            += src_data[iidx]
                                               * weights_data[widx];
                                }
                            }
                        }

                        if (flag_relu) {
                            dst_data_ref[out_idx] = dst_data_ref[out_idx] > 0.f ? dst_data_ref[out_idx] : 0.f;
                        }
                    }
                }
            }
        }
    }
}

//fill_tensor_host_rand(TENSOR,-10.0f, 10.0f);
#define FILL_TENSOR(TENSOR){\
fill_tensor_host_rand(TENSOR,-10.0f, 10.0f);\
}while(0)


template<typename TargetType, typename TargetType_H>
int test_conv_results(int group,
                      int input_num, int in_channels, int height, int width,
                      int out_channels, int kernel_h, int kernel_w,
                      int stride_h, int stride_w, int dilation_h, int dilation_w,
                      int pad_h, int pad_w, bool bias_term,
                      SaberImplStrategy strategy, ImplEnum imp) {

    Shape input_s({input_num, in_channels, height, width});
    Shape weights_s({out_channels, in_channels, kernel_h, kernel_w});
    Shape bias_s({1, out_channels, 1, 1});

    // init input Tensor
    Tensor<TargetType,AK_FLOAT> input_dev;
    Tensor<TargetType_H,AK_FLOAT> input_host;
    input_dev.re_alloc(input_s);
    input_host.re_alloc(input_s);
    FILL_TENSOR(input_dev);
    input_host.copy_from(input_dev);

    // init weights Tensor
    Tensor<TargetType,AK_FLOAT> weights_dev;
    Tensor<TargetType_H,AK_FLOAT> weights_host;
    weights_dev.re_alloc(weights_s);
    weights_host.re_alloc(weights_s);
    FILL_TENSOR(weights_dev);
    weights_host.copy_from(weights_dev);

    Tensor<TargetType,AK_FLOAT> bias_dev;
    Tensor<TargetType_H,AK_FLOAT> bias_host;

    if (bias_term) {
        bias_dev.re_alloc(bias_s);
        bias_host.re_alloc(bias_s);
        FILL_TENSOR(bias_dev);
        bias_host.copy_from(bias_dev);
    }

    Tensor<TargetType,AK_FLOAT> output_dev;
    Tensor<TargetType_H,AK_FLOAT> output_host;
    Tensor<TargetType_H,AK_FLOAT> check_host;

    Context<TargetType> ctx1(0, 1, 1);

    ConvParam<Tensor<TargetType,AK_FLOAT> > param(group, pad_h, pad_w,
                                stride_h, stride_w,
                                dilation_h, dilation_w,
                                &weights_dev, &bias_dev);
    Conv<TargetType, AK_FLOAT> conv;
    std::vector<Tensor<TargetType,AK_FLOAT>* > input_v;
    std::vector<Tensor<TargetType,AK_FLOAT>* > output_v;
    input_v.push_back(&input_dev);
    output_v.push_back(&output_dev);
    conv.compute_output_shape(input_v, output_v, param);
    output_dev.re_alloc(output_dev.valid_shape());

    conv.init(input_v, output_v, param, strategy, imp, ctx1);
    conv(input_v, output_v, param, ctx1);

    typename Tensor<TargetType,AK_FLOAT>::API::stream_t stream = ctx1.get_compute_stream();
    output_v[0]->record_event(stream);
    output_v[0]->sync();
    output_host.re_alloc(output_dev.valid_shape());
    output_host.copy_from(output_dev);
    check_host.re_alloc(output_host.valid_shape());

    conv_basic_check<TargetType_H>(input_host, check_host,
                                   (const float*)weights_host.data(), (const float*)bias_host.data(),
                                   group, kernel_w, kernel_h, stride_w, stride_h,
                                   dilation_w, dilation_h, pad_w, pad_h, bias_term, false);
    double max_ratio = 0.0;
    double max_diff = 0.0;
    record_tensor_to_file(output_host,"output_host");
    record_tensor_to_file(check_host,"check_host");
    tensor_cmp_host((const float*)output_host.data(), (const float*)check_host.data(),
                    check_host.valid_size(), max_ratio, max_diff);

    if (fabs(max_ratio) < 1e-4||fabs(max_diff)<1e-4) {
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


TEST(TestSaberFuncX86, test_saber_conv_results) {

    Env<X86>::env_init();


    std::vector<int> kernel_h_v {1, 3};
    std::vector<int> kernel_w_v{1, 3};
    std::vector<int> pad_h_v{0, 1};
    std::vector<int> pad_w_v{0, 1};
    std::vector<int> stride_h_v{1, 2};
    std::vector<int> stride_w_v{1, 2};
    std::vector<int> dilation_h_v{1, 2};
    std::vector<int> dilation_w_v{1, 2};
    std::vector<int> in_channels_v{3, 32};
    std::vector<int> out_channels_v{32, 57};
    std::vector<int> group_v{1};
    std::vector<int> in_h_v{17, 32};
    std::vector<int> in_w_v{17, 32};
    std::vector<int> input_num_v{2};
    std::vector<bool> bias_term_v{true, false};

//    std::vector<int> kernel_h_v {3};
//    std::vector<int> kernel_w_v{1};
//    std::vector<int> pad_h_v{1};
//    std::vector<int> pad_w_v{1};
//    std::vector<int> stride_h_v{2};
//    std::vector<int> stride_w_v{1};
//    std::vector<int> dilation_h_v{2};
//    std::vector<int> dilation_w_v{1};
//    std::vector<int> in_channels_v{4};
//    std::vector<int> out_channels_v{4};
//    std::vector<int> group_v{2};
//    std::vector<int> in_h_v{4};
//    std::vector<int> in_w_v{4};
//    std::vector<int> input_num_v{2};
//    std::vector<bool> bias_term_v{false};

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

                                                                test_conv_results<X86, X86>(group,
                                                                                              input_num, in_channels,
                                                                                              height,
                                                                                              width,
                                                                                              out_channels, kernel_h,
                                                                                              kernel_w,
                                                                                              stride_h, stride_w,
                                                                                              dilation_h,
                                                                                              dilation_w,
                                                                                              pad_h, pad_w, bias_term,
                                                                                              SPECIFY,
                                                                                              SABER_IMPL);

                                                            }
}


int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}