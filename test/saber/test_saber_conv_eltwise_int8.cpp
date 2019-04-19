#include "saber/core/context.h"
#include "saber/funcs/conv_eltwise.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "conv_func_helper.h"
#include "saber/core/tensor_op.h"
#include <vector>
#if defined(USE_X86_PLACE)
#include "jit_generator.h"
#endif
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
int test_conv_results(int group,
                      int input_num, int in_channels, int height, int width,
                      int out_channels, int kernel_h, int kernel_w,
                      int stride_h, int stride_w, int dilation_h, int dilation_w,
                      int pad_h, int pad_w, bool bias_term, bool relu,
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
              << " bias_term = " << (bias_term ? "true" : "false");

#ifdef USE_X86_PLACE
    Shape input_s({input_num, height, width, in_channels}, Layout_NHWC);
    Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
    Shape weights_s_dw({group, in_channels / group, kernel_h, kernel_w}, Layout_NCHW);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);
    int out_height = (pad_h * 2 + height - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int out_width = (pad_w * 2 + width - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    Shape output_s({input_num, out_height, out_width, out_channels}, Layout_NHWC);

    // init input Tensor
    Tensor<TargetType> input_dev;
    Tensor<TargetType_H> input_host;
    input_dev.re_alloc(input_s, AK_UINT8);
    input_host.re_alloc(input_s, AK_UINT8);
    fill_tensor_rand(input_dev, 0.0f, 32.0f);
    input_host.copy_from(input_dev);
    input_dev.set_scale({1 / 512.f});
    // LOG(INFO) << input_dev.get_scale()[0];

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

    // int bias
    Tensor<TargetType> bias_dev;
    Tensor<TargetType_H> bias_host;

    if (bias_term) {
        bias_dev.re_alloc(bias_s, AK_INT32);
        bias_host.re_alloc(bias_s, AK_INT32);
        fill_tensor_rand(bias_dev, -1.0f, 1.0f);
        bias_host.copy_from(bias_dev);
    }

    Context<TargetType> ctx1(0, 1, 1);
    ActivationParam<TargetType> act_param;

    if (relu) {
        ActivationParam<TargetType> act_relu_param(Active_relu);
        act_param = act_relu_param;
    }

    ConvParam<TargetType> conv_param(group, pad_h, pad_w,
                                     stride_h, stride_w,
                                     dilation_h, dilation_w,
                                     &weights_dev, bias_term ? &bias_dev : nullptr,
                                      act_param,1.f,0.f,round_mode::nearest);

    std::vector<float> coeff;
    coeff.push_back(1.0f);
    coeff.push_back(0.5f);
    EltwiseParam<TargetType> elt_param(Eltwise_sum, coeff);
    ConvEltwiseParam<TargetType> param(conv_param, elt_param);

    // init output Tensor
    Tensor<TargetType> output_dev;
    Tensor<TargetType_H> output_host;
    Tensor<TargetType_H> check_host;

    if (conv_param.activation_param.has_active) {
        output_dev.re_alloc(output_s, AK_UINT8);
        output_host.re_alloc(output_s, AK_UINT8);
        output_dev.set_scale({1 / 256.0f});
        check_host.re_alloc(output_host.valid_shape(), AK_UINT8);
    } else {
        output_dev.re_alloc(output_s, AK_INT8);
        output_host.re_alloc(output_s, AK_INT8);
        output_dev.set_scale({1 / 128.0f});
        check_host.re_alloc(output_host.valid_shape(), AK_INT8);
    }

    fill_tensor_const(output_dev, 4.0f);
    check_host.copy_from(output_dev);
    output_host.copy_from(output_dev);

    ConvEltwise<TargetType, AK_INT8> conv_eltwise;
    std::vector<Tensor<TargetType>* > input_v;
    std::vector<Tensor<TargetType>* > output_v;
    input_v.push_back(&input_dev);
    output_v.push_back(&output_dev);

    if (conv_eltwise.init(input_v, output_v, param, strategy, imp, ctx1) == SaberSuccess) {
        conv_eltwise(input_v, output_v, param, ctx1);
    } else {
        LOG(INFO) << "conv_eltwise init fail";
        return 0;
    }

    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
    output_v[0]->record_event(stream);
    output_v[0]->sync();

    if (conv_param.activation_param.has_active) {
        output_host.re_alloc(output_dev.valid_shape(), AK_UINT8);
        output_host.copy_from(output_dev);
    } else {
        output_host.re_alloc(output_dev.valid_shape(), AK_INT8);
        output_host.copy_from(output_dev);
    }

    // calc scale info
    std::vector<float> scale;
    float scale_in = input_dev.get_scale()[0];
    float scale_out = output_dev.get_scale()[0];
    auto scale_w = weights_dev.get_scale();
    std::vector<float>().swap(scale);

    for (int i = 0; i < scale_w.size(); i++) {
        scale.push_back((scale_w[i]*scale_in) / scale_out);
    }

    conv_basic_check_int8<X86>(input_host, check_host,
                               (const char*)weights_host.data(), bias_term ? (const int*)bias_host.data() : nullptr,
                               group, kernel_w, kernel_h, stride_w, stride_h,
                               dilation_w, dilation_h, pad_w, pad_h, bias_term,
                               conv_param.activation_param.has_active, scale, &elt_param);
    int count = count_diff((const unsigned char*)output_host.data(),
                           (const unsigned char*)check_host.data(), check_host.valid_size(), 2e-1);


    if ((double)count / output_host.valid_size() < 0.02) {
        LOG(INFO) << "PASS!!! count = " << count;
        return 0;
    } else {
        print_tensor_valid(output_host);
        print_tensor_valid(check_host);
        LOG(FATAL) << "FAIL!!! count = " << count
                   << " conv param: "
                   << " group = " << group
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
#endif
}

#ifdef USE_X86_PLACE
template<typename TargetType, typename TargetType_H>
int test_conv_results_nhwc(int group,
                           int input_num, int in_channels, int height, int width,
                           int out_channels, int kernel_h, int kernel_w,
                           int stride_h, int stride_w, int dilation_h, int dilation_w,
                           int pad_h, int pad_w, bool bias_term, bool with_relu,
                           SaberImplStrategy strategy, ImplEnum imp,bool is_unsigned=true) {

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
                     << " bias_term = " << (bias_term ? "true" : "false");

    float input_max=5.f;
    Shape input_nhwc({input_num,  height, width, in_channels}, Layout_NHWC);
    Shape input_nchw({input_num, in_channels, height, width}, Layout_NCHW);
    Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);
    int out_height = (pad_h * 2 + height - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int out_width = (pad_w * 2 + width - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    Shape output_nhwc({input_num, out_height, out_width, out_channels}, Layout_NHWC);
    Shape output_nchw({input_num, out_channels, out_height, out_width}, Layout_NCHW);

    // init input Tensor
    Tensor<TargetType> input_dev;
    Tensor<TargetType_H> input_host;
    Tensor<TargetType> input_dev_temp;
    input_dev.re_alloc(input_nhwc, AK_INT8);
    input_dev_temp.re_alloc(input_nchw, AK_INT8);
    input_host.re_alloc(input_nchw, AK_FLOAT);
    bool nothing_flag = false;
    std::string nothing_str = "";

    fill_tensor_rand(input_host,-input_max,input_max);
//    load_tensor_in_io_format(input_host,nothing_flag,nothing_str,"record+ConvEltwise+res2a_branch2c+in+0+1_64_56_56_+nchw+ak_float+0.txt");
    input_host.set_scale({input_max/127.f});
    utils::ScaleUtils::scale_fp32_int8(input_dev_temp,input_host);
    reorder_nhwc_nchw(input_dev_temp,input_dev);
    input_dev.set_scale(input_host.get_scale());


    // init weights Tensor
    Tensor<TargetType> weights_dev;
    Tensor<TargetType_H> weights_host;
    weights_dev.re_alloc(weights_s, AK_FLOAT);
    weights_host.re_alloc(weights_s, AK_FLOAT);

    fill_tensor_rand(weights_dev,-input_max,input_max);
//    load_tensor_in_io_format(weights_dev,nothing_flag,nothing_str,"record+weights+conv_eltwise+out+0+256_64_1_1_+nchw+ak_float+0.txt");
    weights_host.copy_from(weights_dev);


    Tensor<TargetType> bias_dev;
    Tensor<TargetType_H> bias_host;
    if (bias_term) {
        bias_dev.re_alloc(bias_s, AK_FLOAT);
        bias_host.re_alloc(bias_s, AK_FLOAT);
        fill_tensor_rand(bias_dev, -input_max, input_max);
//        fill_tensor_const(bias_dev, 0.f);
//        load_tensor_in_io_format(bias_dev,nothing_flag,nothing_str,"record+bias+conv_eltwise+out+0+1_256_1_1_+nchw+ak_float+0.txt");
        bias_host.copy_from(bias_dev);
    }
    Tensor<TargetType> output_load_temp_fp32(output_nchw,AK_FLOAT);
    Tensor<TargetType> output_load_temp_int8(output_nchw,AK_INT8);
//    fill_tensor_const(output_load_temp_fp32,0);
    fill_tensor_rand(output_load_temp_fp32,-input_max,input_max);
//    load_tensor_in_io_format(output_load_temp_fp32,nothing_flag,nothing_str,"record+pre_out+conv_eltwise+out+3+1_256_56_56_+nchw+ak_float+0.txt");
    Tensor<TargetType> output_dev(output_nhwc,AK_INT8);

    output_dev.set_scale({(in_channels*kernel_h*kernel_w*input_max)/127.f});

//    float elt_scale=0.019590;
    float elt_scale=input_max/127.f;
    Tensor<TargetType_H> output_host(output_nchw);
    Tensor<TargetType_H> check_host(output_nchw);
    check_host.copy_from(output_load_temp_fp32);
    output_load_temp_int8.set_scale({elt_scale});
    output_load_temp_fp32.set_scale({elt_scale});
    LOG(INFO)<<"out scale "<<output_load_temp_int8.get_scale().size();
    utils::ScaleUtils::scale_fp32_int8(output_load_temp_int8,output_load_temp_fp32);
    LOG(INFO)<<"out scale "<<output_load_temp_int8.get_scale().size();
    reorder_nhwc_nchw(output_load_temp_int8,output_dev);

    Context<TargetType> ctx1(0, 1, 1);
    EltwiseParam<TargetType> elt_param(Eltwise_sum,{1,1});

    ConvParam<TargetType> conv_param(group, pad_h, pad_w,
                                stride_h, stride_w,
                                dilation_h, dilation_w,
                                &weights_dev, &bias_dev);
    if (with_relu) {
        ActivationParam<TargetType> act_param(Active_relu);
        conv_param.activation_param = act_param;
        elt_param.activation_param=act_param;
    }
//    EltwiseParam<TargetType> elt_param(Eltwise_sum,{1,0.019590});
    conv_param.beta=elt_scale;
    conv_param.beta_type=AK_INT8;

    ConvEltwiseParam<TargetType> conv_elt_param(conv_param,elt_param);
    ConvEltwise<TargetType, AK_INT8> conv;
    std::vector<Tensor<TargetType>* > input_v;
    std::vector<Tensor<TargetType>* > output_v;
    input_v.push_back(&input_dev);
    output_v.push_back(&output_dev);
//    write_tensorfile(output_dev,"init_output",false);
    conv.compute_output_shape(input_v, output_v, conv_elt_param);


    conv.init(input_v, output_v, conv_elt_param, strategy, imp, ctx1);

    conv(input_v, output_v, conv_elt_param, ctx1);

    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
    output_v[0]->record_event(stream);
    output_v[0]->sync();
    reorder_nhwc_nchw(output_dev,output_host);

    conv_basic_check<TargetType_H>(input_host, check_host,
                                   (const float*)weights_host.data(), (const float*)bias_host.data(),
                                   group, kernel_w, kernel_h, stride_w, stride_h,
                                   dilation_w, dilation_h, pad_w, pad_h, bias_term,
                                   conv_elt_param.conv_param.activation_param.has_active,1.f);
//    print_tensor_valid(check_host);
    double max_ratio = 0.0;
    double max_diff = 0.0;
    //tensor_cmp_host((const float*)output_host.data(), (const float*)check_host.data(),
    //                check_host.valid_size(), max_ratio, max_diff);
    tensor_cmp_host_mlu((const float*)output_host.data(), (const float*)check_host.data(),
                        check_host.valid_size(), max_ratio, max_diff);

//    int count = count_diff((const float*)output_host.data(),
//                           (const float*)check_host.data(), check_host.valid_size(), 2e-1);
    if (max_ratio< 0.15) {
        //LOG(INFO) << " PASS!!! max_ratio = " << max_ratio << " max_diff = " << max_diff;
        write_tensorfile(output_host,"output_host");
        write_tensorfile(check_host,"check_host");
                LOG(INFO) << "PASS!!! ratio = " << max_ratio <<" in "<<output_host.valid_size();
        return 0;
    } else {
        write_tensorfile(output_dev,"output_dev",false);
        write_tensorfile(output_host,"output_host");
        write_tensorfile(check_host,"check_host");
//        write_tensorfile(weights_dev,"ori_weights.txt");
//        write_tensorfile(weights_host,"ori_weights2.txt");
//        write_tensorfile(output_dev, "ori_int8_output.txt");
//        write_tensorfile(output_host, "int8_output.txt");
//        write_tensorfile(check_host, "fp32_output.txt");
//        print_tensor_valid(output_host);
//        print_tensor_valid(check_host);
        //LOG(FATAL) << "FAIL!!! max_ratio = " << max_ratio << " max_diff = " << max_diff
                LOG(FATAL) << "FAIL!!! ratio = " << max_ratio<<" in "<<output_host.valid_size()<<","
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

#endif
template <typename TargetType, typename TargetType_H>
void test_conv_eltwise() {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
#endif

#if 0
    std::vector<int> kernel_h_v{1, 3};
    std::vector<int> kernel_w_v{1, 3};
    std::vector<int> pad_h_v{0, 1};
    std::vector<int> pad_w_v{0, 1};
    std::vector<int> stride_h_v{1};
    std::vector<int> stride_w_v{1};
    std::vector<int> dilation_h_v{1};
    std::vector<int> dilation_w_v{1};
    std::vector<int> group_v{1};
    std::vector<int> in_h_v{8};
    std::vector<int> in_w_v{8};
    std::vector<int> input_num_v{3};
    std::vector<int> input_channels_v{16};
    std::vector<int> output_channels_v{16};
    std::vector<bool> bias_term_v{true};
    std::vector<bool> with_relu_v{true};

    for (auto group : group_v)
    for (auto input_num : input_num_v)
    for (auto out_channels : output_channels_v)
    for (auto in_channels : input_channels_v)
    for (auto kernel_h : kernel_h_v)
    for (auto kernel_w : kernel_w_v)
    for (auto height : in_h_v)
    for (auto width : in_w_v)
    for (auto stride_h : stride_h_v)
    for (auto stride_w : stride_w_v)
    for (auto dilation_h : dilation_h_v)
    for (auto dilation_w : dilation_w_v)
    for (auto pad_h : pad_h_v)
    for (auto pad_w : pad_w_v)
    for (auto bias_term : bias_term_v)
    for (auto relu : with_relu_v) {

//#ifdef USE_CUDA
//        test_conv_results<NV, NVHX86>(group,
//                                      input_num,
//                                      in_channels,
//                                      height,
//                                      width,
//                                      out_channels,
//                                      kernel_h,
//                                      kernel_w,
//                                      stride_h, stride_w, dilation_h, dilation_w,
//                                      pad_h, pad_w, bias_term, relu,
//                                      SPECIFY,
//                                      VENDER_IMPL);
//
//#endif
#else
        {
        int group = 1;
        int input_num = 1;
        int in_channels = 64;
        int height = 56;
        int width = 56;
        int out_channels = 64;
        int kernel_h = 1;
        int kernel_w = 1;
        int stride_h = 1;
        int stride_w = 1;
        int dilation_h = 1;
        int dilation_w = 1;
        int pad_h = 0;
        int pad_w = 0;
        bool bias_term = true;
        bool with_relu = true;
#endif
#ifdef USE_X86_PLACE
        if (jit::mayiuse(jit::avx512_core_vnni)) {
//            test_conv_results<X86, X86>(group,
//                                        input_num,
//                                        in_channels,
//                                        height,
//                                        width,
//                                        out_channels,
//                                        kernel_h,
//                                        kernel_w,
//                                        stride_h, stride_w, dilation_h, dilation_w,
//                                        pad_h, pad_w, bias_term, relu,
//                                        SPECIFY,
//                                        SABER_IMPL);


        test_conv_results_nhwc<X86,X86>(group,
                                        input_num, in_channels,
                                        height, width,
                                        out_channels, kernel_h,
                                        kernel_w,
                                        stride_h, stride_w,
                                        dilation_h, dilation_w,
                                        pad_h, pad_w, bias_term,with_relu,
                                        SPECIFY, SABER_IMPL);
        }
#endif
    }
}

TEST(TestSaberFunc, test_saber_conv_results) {

#ifdef USE_CUDA
    test_conv_eltwise<NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    test_conv_eltwise<X86, X86>();
#endif

}
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
