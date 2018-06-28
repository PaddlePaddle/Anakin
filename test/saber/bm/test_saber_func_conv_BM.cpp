#include "core/context.h"
#include "funcs/conv.h"
#include "test_saber_func_BM.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

template <typename Tensor>
void print_tensor_shape(std::string name, Tensor &t0) {

            LOG(INFO) << name << " valid shape is ["
                      << t0.valid_shape()[0] << ", "
                      << t0.valid_shape()[1] << ", "
                      << t0.valid_shape()[2] << ", "
                      << t0.valid_shape()[3] << "].";

            LOG(INFO) << name << " real shape is ["
                      << t0.shape()[0] << ", "
                      << t0.shape()[1] << ", "
                      << t0.shape()[2] << ", "
                      << t0.shape()[3] << "].";

            LOG(INFO) << name << " offset is ["
                      << t0.offset()[0] << ", "
                      << t0.offset()[1] << ", "
                      << t0.offset()[2] << ", "
                      << t0.offset()[3] << "].";
}

//Round a / b to nearest higher integer value
inline int i_div_up(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


TEST(TestSaberFuncBM, test_conv_result) {

    int group = 1;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int kernel_h = 3;
    int kernel_w = 3;
    int out_channels = 1;
    
    int img_num = 1;
    int in_channels = 1;
    int img_h = 8;
    int img_w = 8;

    bool bias_term = true;

    LOG(INFO) << "conv param: ";
    LOG(INFO) << "  img_num = " << img_num;
    LOG(INFO) << "  in_channels = " << in_channels;
    LOG(INFO) << "  img_h = " << img_h;
    LOG(INFO) << "  img_w = " << img_w;
    LOG(INFO) << "  group = " << group;
    LOG(INFO) << "  pad_h = " << pad_h;
    LOG(INFO) << "  pad_w = " << pad_w;
    LOG(INFO) << "  stride_h = " << stride_h;
    LOG(INFO) << "  stride_w = " << stride_w;
    LOG(INFO) << "  dilation_h = " << dilation_h;
    LOG(INFO) << "  dilation_w = " << dilation_w;
    LOG(INFO) << "  kernel_h = " << kernel_h;
    LOG(INFO) << "  kernel_w = " << kernel_w;
    LOG(INFO) << "  out_channels = " << out_channels;

    Shape img_s(img_num, in_channels, img_h, img_w);
    Shape weights_s(out_channels, in_channels, kernel_h, kernel_w);
    Shape bias_s(1, out_channels, 1, 1);

    TensorHf4 img_host;
    TensorDf4 img_dev;
    
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = i;
    }

    img_dev.copy_from(img_host);
    
    TensorHf4 weights_host;
    TensorDf4 weights_dev;
    
    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);

    fill_tensor_host_const(weights_host, 1.f);
    weights_dev.copy_from(weights_host);

    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(bias_host, 1.f);
        bias_dev.copy_from(bias_host);
    }

    TensorHf4 output_host;
    TensorDf4 output_dev;

    // start Reshape & doInfer
    Context<BM> ctx1(0, 1, 1);
    
    ConvParam<TensorDf4> param(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Conv<BM, AK_BM, AK_BM, AK_BM, NCHW> conv;
    conv.compute_output_shape(input, output, param);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());

    LOG(INFO) << "regular start with group = " << group;
    // init assume output tensor has been reshpaed by user.
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);

    conv(input, output, param, ctx1);

    output_dev.sync();

    print_tensor_device(img_dev);
    print_tensor_device(output_dev);
}

TEST(TestSaberFuncBM, test_conv_param_change) {

    int group = 4;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int kernel_h = 3;
    int kernel_w = 3;
    int out_channels = 4;

    int img_num = 1;
    int in_channels = 4;
    int img_h = 64;
    int img_w = 64;

    bool bias_term = true;

    Shape img_s(img_num, in_channels, img_h, img_w);
    Shape weights_s(out_channels, in_channels, kernel_h, kernel_w);
    Shape bias_s(1, out_channels, 1, 1);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = i;
    }

    img_dev.copy_from(img_host);

    TensorHf4 weights_host;
    TensorDf4 weights_dev;

    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);

    fill_tensor_host_const(weights_host, 1.f);
    weights_dev.copy_from(weights_host);

    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(bias_host, 1.f);
        bias_dev.copy_from(bias_host);
    }

    TensorHf4 output_host;
    TensorDf4 output_dev;

    // start Reshape & doInfer
    Context<BM> ctx1(0, 1, 1);

    ConvParam<TensorDf4> param(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Conv<BM, AK_BM, AK_BM, AK_BM, NCHW> conv;
    conv.compute_output_shape(input, output, param);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());

    LOG(INFO) << "regular start with group = " << group;
    // init assume output tensor has been reshpaed by user.
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);

    conv(input, output, param, ctx1);

    param.group = 1;
    param.pad_h = 1;
    param.pad_w = 1;

    LOG(INFO) << " param changed start with group = " << param.group;
    conv(input, output, param, ctx1);
}

TEST(TestSaberFuncBM, test_conv_share_sub_tensor) {

    int group = 1;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int kernel_h = 3;
    int kernel_w = 3;
    int out_channels = 2;

    int img_num = 1;
    int in_channels = 2;
    int img_h = 8;
    int img_w = 8;

    bool bias_term = true;

    LOG(INFO) << "conv param: ";
    LOG(INFO) << " img_num = " << img_num;
    LOG(INFO) << " in_channels = " << in_channels;
    LOG(INFO) << " img_h = " << img_h;
    LOG(INFO) << " img_w = " << img_w;
    LOG(INFO) << " group = " << group;
    LOG(INFO) << " pad_h = " << pad_h;
    LOG(INFO) << " pad_w = " << pad_w;
    LOG(INFO) << " stride_h = " << stride_h;
    LOG(INFO) << " stride_w = " << stride_w;
    LOG(INFO) << " dilation_h = " << dilation_h;
    LOG(INFO) << " dilation_w = " << dilation_w;
    LOG(INFO) << " kernel_h = " << kernel_h;
    LOG(INFO) << " kernel_w = " << kernel_w;
    LOG(INFO) << " out_channels = " << out_channels;

    Shape img_s(img_num, in_channels, img_h, img_w);
    Shape weights_s(out_channels, in_channels, kernel_h, kernel_w);
    Shape bias_s(1, out_channels, 1, 1);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }

    img_dev.copy_from(img_host);

    Shape img_s_sub(img_num, in_channels, 4, 4);

    TensorDf4 t0;
    TensorDf4 t1;

    t0.share_sub_buffer(img_dev, img_s_sub, {0,0,0,0});
    t1.share_sub_buffer(img_dev, img_s_sub, {0,0,4,4});

    print_tensor_shape("t0", t0);
    print_tensor_shape("t1", t1);

    TensorHf4 weights_host;
    TensorDf4 weights_dev;

    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);

    fill_tensor_host_const(weights_host, 1.f);
    weights_dev.copy_from(weights_host);

    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(bias_host, 1.f);
        bias_dev.copy_from(bias_host);
    }

    TensorDf4 output_dev;

    // start Reshape & doInfer
    Context<BM> ctx1(0, 1, 1);
    Context<BM> ctx2(0, 2, 2);

    TensorDf4 out0;
    TensorDf4 out1;

    ConvParam<TensorDf4> param0(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);

    ConvParam<TensorDf4> param1(group, pad_h, pad_w,
                                stride_h, stride_w,
                                dilation_h, dilation_w,
                                &weights_dev, &bias_dev);

    std::vector<TensorDf4*> input0, input1;
    std::vector<TensorDf4*> output0, output1;

    input0.push_back(&t0);
    input1.push_back(&t1);

    output0.push_back(&out0);
    output1.push_back(&out1);

    output_dev.re_alloc(img_s);

    Conv<BM, AK_BM, AK_BM, AK_BM, NCHW> conv0;
    Conv<BM, AK_BM, AK_BM, AK_BM, NCHW> conv1;

    conv0.compute_output_shape(input0, output0, param0);
    conv1.compute_output_shape(input1, output1, param1);

    out0.share_sub_buffer(output_dev, output0[0]->valid_shape(),{0,0,0,0});
    out1.share_sub_buffer(output_dev, output1[0]->valid_shape(),{0,0,4,4});

    conv0.init(input0, output0, param0, SPECIFY, VENDER_IMPL, ctx1);
    conv1.init(input1, output1, param1, SPECIFY, VENDER_IMPL, ctx2);

    conv0(input0, output0, param0, ctx1);
    conv1(input1, output1, param1, ctx2);
}

TEST(TestSaberFuncBM, test_conv_fp32_speed_test) {

    int group = 1;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int kernel_h = 3;
    int kernel_w = 3;
    int out_channels = 128;

    int img_num = 64;
    int in_channels = 4;
    int img_h = 32;
    int img_w = 32;

    bool bias_term = true;

    LOG(INFO) << "conv param: ";
    LOG(INFO) << "  img_num = " << img_num;
    LOG(INFO) << "  in_channels = " << in_channels;
    LOG(INFO) << "  out_channels = " << out_channels;
    LOG(INFO) << "  img_h = " << img_h;
    LOG(INFO) << "  img_w = " << img_w;
    LOG(INFO) << "  group = " << group;
    LOG(INFO) << "  pad = " << pad_h;
    LOG(INFO) << "  stride = " << stride_h;
    LOG(INFO) << "  dilation = " << dilation_h;
    LOG(INFO) << "  kernel_h = " << kernel_h;
    LOG(INFO) << "  kernel_w = " << kernel_w;
    Shape img_s(img_num, in_channels, img_h, img_w);
    Shape weights_s(out_channels, in_channels, kernel_h, kernel_w);
    Shape bias_s(1, out_channels, 1, 1);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 1;
    }

    img_dev.copy_from(img_host);

    TensorHf4 weights_host;
    TensorDf4 weights_dev;

    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);

    fill_tensor_host_const(weights_host, 1.f);
    weights_dev.copy_from(weights_host);

    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(bias_host, 1.f);
        bias_dev.copy_from(bias_host);
    }

    TensorDf4 output_dev;

    // start Reshape & doInfer
    Context<BM> ctx1(0, 1, 1);

    ConvParam<TensorDf4> param(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Conv<BM, AK_BM, AK_BM, AK_BM, NCHW> conv;
    conv.compute_output_shape(input, output, param);

    output_dev.re_alloc(output[0]->shape());
    LOG(INFO) << "Output shape = [ " << output[0]->shape()[0] << " " << output[0]->shape()[1] << " " \
        << output[0]->shape()[2] << " " << output[0]->shape()[3] << "]";

    LOG(INFO) << "saber conv init";
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);

    /* conv(input, output, param, ctx1); */
    /* output_dev.sync(); */

    LOG(INFO) << "saber conv dispatch";
    SaberTimer<BM> t1;
    int ts = 100;
    t1.start(ctx1);
    for (int i = 0; i < ts; ++i) {
        conv(input, output, param, ctx1);
        output_dev.sync();
    }
    t1.end(ctx1);
    LOG(INFO) << "elapse time: " << t1.get_average_ms()/ts << " ms";
}

TEST(TestSaberFuncBM, test_conv_fp32_1x1_speed) {
    int img_num = 1;
    int kernel = 1;
    int out_channels = 128;
    int in_channels = 512;
    int img_h = 32;
    int img_w = 32;
    int pad = 0;
    int stride = 1;

    TensorDf4 weights;
    TensorDf4 bias;
    weights.re_alloc({out_channels, in_channels, 1, 1});
    bias.re_alloc({1, out_channels, 1, 1});

    TensorDf4 img, out;
    img.re_alloc({1, in_channels, img_h, img_w});

    fill_tensor_device_rand(weights, -1.f, 1.f);
    fill_tensor_device_rand(bias, -1.f, 1.f);
    fill_tensor_device_rand(img, -1.f, 1.f);

    LOG(INFO) << "conv param: ";
    LOG(INFO) << "  img_num: " << img_num;
    LOG(INFO) << "  kernel: " << kernel;
    LOG(INFO) << "  out_channels: " << out_channels;
    LOG(INFO) << "  in_channels: " << in_channels;
    LOG(INFO) << "  img_h: " << img_h;
    LOG(INFO) << "  img_w: " << img_w;
    LOG(INFO) << "  pad: " << pad;
    LOG(INFO) << "  stride: " << stride;

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img);
    output.push_back(&out);

    ConvParam<TensorDf4> conv_param(1, pad, pad,
                                    stride, stride,
                                    1, 1,
                                    &weights, &bias);
    Conv<BM, AK_BM, AK_BM, AK_BM, NCHW> conv;
    conv.compute_output_shape(input, output, conv_param);
    out.re_alloc(output[0]->shape());
    Context<BM> ctx1(0, 1, 1);
    conv.init(input, output, conv_param, SPECIFY, VENDER_IMPL, ctx1);

    SaberTimer<BM> t1;
    int ts = 100;
    t1.start(ctx1);
    for (int i = 0; i < ts; ++i) {
        conv(input, output, conv_param, ctx1);
        out.sync();
    }
    t1.end(ctx1);
    LOG(INFO) << "elapse time: " << t1.get_average_ms()/ts << " ms";
}

int main(int argc, const char** argv){
    Env<BM>::env_init();
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

