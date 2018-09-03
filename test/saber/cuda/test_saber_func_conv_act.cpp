#include "core/context.h"
#include "funcs/conv_act.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

int group = 1;
int pad_h = 1;
int pad_w = 1;
int stride_h = 1;
int stride_w = 1;
int dilation_h = 2;
int dilation_w = 2;

int kernel_h = 3;
int kernel_w = 3;
int g_out_channels = 48;

int img_num = 1;
int g_in_channels = 8;
int img_h = 16;
int img_w = 16;

bool g_bias_term = true;

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

Shape img_s(img_num, g_in_channels, img_h, img_w);
Shape weights_s(g_out_channels, g_in_channels, kernel_h, kernel_w);
Shape bias_s(1, g_out_channels, 1, 1);

TEST(TestSaberFuncNV, test_func_conv_relu_fusion) {

    TensorHf4 img_host;
    TensorDf4 img_dev;
    TensorHf4 weights_host;
    TensorDf4 weights_dev;
    TensorHf4 bias_host;
    TensorDf4 bias_dev;
    TensorHf4 output_host;
    TensorDf4 output_dev;
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }

    img_dev.copy_from(img_host);
    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);
    fill_tensor_host_const(weights_host, 1.f);
    weights_dev.copy_from(weights_host);

    if (g_bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);
        fill_tensor_host_const(bias_host, 1.f);
        bias_dev.copy_from(bias_host);
    }

    LOG(INFO) << "test conv + relu";

    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(event);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;

    Context<NV> ctx1(0, 1, 1);
    ConvParam<TensorDf4> conv_param(group, pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    &weights_dev, &bias_dev);

    ActivationParam<TensorDf4> active_param(Active_relu);

    ConvActiveParam<TensorDf4> param(conv_param, active_param);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    ConvAct<NV, AK_FLOAT> conv;
    conv.compute_output_shape(input, output, param);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());

    LOG(INFO) << "regular start with group = " << group;
    // init assume output tensor has been reshpaed by user.
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);

    conv(input, output, param, ctx1);

    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output[0]->record_event(cuda_stream);

    output_dev.sync();
    output_host.copy_from(output_dev);

    print_tensor_host(output_host);

    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_func_conv_bn_relu_fusion) {

    TensorHf4 img_host;
    TensorDf4 img_dev;
    TensorHf4 weights_host;
    TensorDf4 weights_dev;
    TensorHf4 bias_host;
    TensorDf4 bias_dev;
    TensorHf4 output_host;
    TensorDf4 output_dev;
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }

    img_dev.copy_from(img_host);
    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);
    fill_tensor_host_const(weights_host, 1.f);
    weights_dev.copy_from(weights_host);

    if (g_bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);
        fill_tensor_host_const(bias_host, 1.f);
        bias_dev.copy_from(bias_host);
    }

    LOG(INFO) << "test conv + bn + relu";

    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(event);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;

    Context<NV> ctx1(0, 1, 1);
    ConvParam<TensorDf4> conv_param(group, pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    &weights_dev, &bias_dev);

    ActivationParam<TensorDf4> active_param(Active_relu);
    typedef typename TensorDf4::Dtype dtype;

    std::vector<dtype> mean, variance;

    mean.resize(g_out_channels);
    variance.resize(g_out_channels);

    for (int i = 0; i < mean.size(); ++i) {
        mean[i] = 0.1f;
    }

    for (int i = 0; i < variance.size(); ++i) {
        variance[i] = 1.f;
    }

    BatchnormParam<TensorDf4> batchnorm_param(mean, variance, 1);

    ConvActiveParam<TensorDf4> param(conv_param, active_param, batchnorm_param);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    ConvAct<NV, AK_FLOAT> conv;
    conv.compute_output_shape(input, output, param);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());

    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);

    conv(input, output, param, ctx1);

    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output[0]->record_event(cuda_stream);

    output_dev.sync();

    output_host.copy_from(output_dev);

    print_tensor_host(output_host);

    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_func_conv_scale_relu_fusion) {

    TensorHf4 img_host;
    TensorDf4 img_dev;
    TensorHf4 weights_host;
    TensorDf4 weights_dev;
    TensorHf4 bias_host;
    TensorDf4 bias_dev;
    TensorHf4 output_host;
    TensorDf4 output_dev;
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }

    img_dev.copy_from(img_host);
    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);
    fill_tensor_host_const(weights_host, 1.f);
    weights_dev.copy_from(weights_host);

    if (g_bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);
        fill_tensor_host_const(bias_host, 1.f);
        bias_dev.copy_from(bias_host);
    }

    LOG(INFO) << "test conv + scale + relu";

    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(event);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;

    Context<NV> ctx1(0, 1, 1);
    ConvParam<TensorDf4> conv_param(group, pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    &weights_dev, &bias_dev);

    ActivationParam<TensorDf4> active_param(Active_relu);
    typedef typename TensorDf4::Dtype dtype;

    std::vector<dtype> scale_w, scale_b;

    scale_w.resize(g_out_channels);
    scale_b.resize(g_out_channels);

    for (int i = 0; i < scale_w.size(); ++i) {
        scale_w[i] = 0.1f;
    }

    for (int i = 0; i < scale_b.size(); ++i) {
        scale_b[i] = 0.1f;
    }

    ScaleParam<TensorDf4> scale_param(scale_w, scale_b, true);
    ConvActiveParam<TensorDf4> param(conv_param, active_param, scale_param);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    ConvAct<NV, AK_FLOAT> conv;
    conv.compute_output_shape(input, output, param);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());

    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);

    conv(input, output, param, ctx1);

    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output[0]->record_event(cuda_stream);

    output_dev.sync();

    output_host.copy_from(output_dev);

    print_tensor_host(output_host);

    CUDA_CHECK(cudaPeekAtLastError());
}


TEST(TestSaberFuncNV, test_func_conv_bn_scale_relu_fusion) {

    TensorHf4 img_host;
    TensorDf4 img_dev;
    TensorHf4 weights_host;
    TensorDf4 weights_dev;
    TensorHf4 bias_host;
    TensorDf4 bias_dev;
    TensorHf4 output_host;
    TensorDf4 output_dev;
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }

    img_dev.copy_from(img_host);
    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);
    fill_tensor_host_const(weights_host, 1.f);
    weights_dev.copy_from(weights_host);

    if (g_bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);
        fill_tensor_host_const(bias_host, 1.f);
        bias_dev.copy_from(bias_host);
    }

    LOG(INFO) << "test conv + bn + scale + relu";

    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(event);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;

    Context<NV> ctx1(0, 1, 1);
    ConvParam<TensorDf4> conv_param(group, pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    &weights_dev, &bias_dev);

    ActivationParam<TensorDf4> active_param(Active_relu);
    typedef typename TensorDf4::Dtype dtype;

    std::vector<dtype> mean, variance;
    std::vector<dtype> scale_w, scale_b;

    mean.resize(g_out_channels);
    variance.resize(g_out_channels);

    for (int i = 0; i < mean.size(); ++i) {
        mean[i] = 0.1f;
    }

    for (int i = 0; i < variance.size(); ++i) {
        variance[i] = 0.1f;
    }

    scale_w.resize(g_out_channels);
    scale_b.resize(g_out_channels);

    for (int i = 0; i < scale_w.size(); ++i) {
        scale_w[i] = 0.1f;
    }

    for (int i = 0; i < scale_b.size(); ++i) {
        scale_b[i] = 0.1f;
    }

    BatchnormParam<TensorDf4> batchnorm_param(mean, variance, 0.5);
    ScaleParam<TensorDf4> scale_param(scale_w,  scale_b, true);
    ConvActiveParam<TensorDf4> param(conv_param, active_param, batchnorm_param, scale_param);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    ConvAct<NV, AK_FLOAT> conv;
    conv.compute_output_shape(input, output, param);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());

    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);

    conv(input, output, param, ctx1);

    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output[0]->record_event(cuda_stream);

    output_dev.sync();

    output_host.copy_from(output_dev);
    print_tensor_host(output_host);

    CUDA_CHECK(cudaPeekAtLastError());
}
#if 0
>>> >>> > developing
TEST(TestSaberFuncNV, test_func_conv_bn_scale_relu_fusion_share_buffer) {

    TensorHf4 img_host;
    TensorDf4 img_dev;
    TensorHf4 weights_host;
    TensorDf4 weights_dev;
    TensorHf4 bias_host;
    TensorDf4 bias_dev;
    TensorHf4 output_host;
    TensorDf4 output_dev;
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }

    img_dev.copy_from(img_host);
    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);
    fill_tensor_host_const(weights_host, 1.f);
    weights_dev.copy_from(weights_host);

    if (g_bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);
        fill_tensor_host_const(bias_host, 1.f);
        bias_dev.copy_from(bias_host);
    }

    LOG(INFO) << "test conv + bn + scale + relu";

    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(event);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;

    Context<NV> ctx1(0, 1, 1);
    ConvParam<TensorDf4> conv_param(group, pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    &weights_dev, &bias_dev);

    ActivationParam<TensorDf4> active_param(Active_relu);
    typedef typename TensorDf4::Dtype dtype;

    std::vector<dtype> mean, variance;
    std::vector<dtype> scale_w, scale_b;

    mean.resize(g_out_channels);
    variance.resize(g_out_channels);

    for (int i = 0; i < mean.size(); ++i) {
        mean[i] = 0.1f;
    }

    for (int i = 0; i < variance.size(); ++i) {
        variance[i] = 0.1f;
    }

    scale_w.resize(g_out_channels);
    scale_b.resize(g_out_channels);

    for (int i = 0; i < scale_w.size(); ++i) {
        scale_w[i] = 0.1f;
    }

    for (int i = 0; i < scale_b.size(); ++i) {
        scale_b[i] = 0.1f;
    }

    TensorDf4 weights_dev1;
    weights_dev1.re_alloc(weights_dev.shape());
    weights_dev1.copy_from(weights_dev);

    TensorDf4 bias_dev1;
    bias_dev1.re_alloc(bias_dev.shape());
    bias_dev1.copy_from(bias_dev);

    BatchnormParam<TensorDf4> batchnorm_param(mean, variance, 0.5);
    ScaleParam<TensorDf4> scale_param(scale_w,  scale_b, true);
    ConvActiveParam<TensorDf4> param(conv_param, active_param, batchnorm_param, scale_param);

    ConvParam<TensorDf4> conv_param1(group, pad_h, pad_w,
                                     stride_h, stride_w,
                                     dilation_h, dilation_w,
                                     &weights_dev1, &bias_dev1);

    BatchnormParam<TensorDf4> batchnorm_param1(mean, variance, 0.5);
    ScaleParam<TensorDf4> scale_param1(scale_w,  scale_b, true);
    ConvActiveParam<TensorDf4> param1(conv_param1, active_param, batchnorm_param1, scale_param1);

    Shape img_s_sub(img_num, g_in_channels, 4, 4);

    TensorDf4 t0;
    TensorDf4 t1;

    TensorDf4 out0;
    TensorDf4 out1;

    t0.share_sub_buffer(img_dev, img_s_sub, {0, 0, 0, 0});
    t1.share_sub_buffer(img_dev, img_s_sub, {0, 0, 4, 4});

    std::vector<TensorDf4*> input0, input1;
    std::vector<TensorDf4*> output0, output1;

    Shape output_s(img_num, g_out_channels, 8, 8);

    output_dev.re_alloc(output_s);


    input0.push_back(&t0);
    input1.push_back(&t1);
    output0.push_back(&out0);
    output1.push_back(&out1);

    Conv_act<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> conv0;
    Conv_act<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> conv1;

    conv0.compute_output_shape(output0, input0, param);
    conv1.compute_output_shape(output1, input1, param1);

    out0.share_sub_buffer(output_dev, output0[0]->valid_shape(), {0, 0, 0, 0});
    out1.share_sub_buffer(output_dev, output1[0]->valid_shape(), {0, 0, 4, 4});

    conv0.init(input0, output0, param, SPECIFY, VENDER_IMPL, ctx1);
    conv1.init(input1, output1, param1, SPECIFY, VENDER_IMPL, ctx1);

    conv0(input0, output0, param, ctx1);
    conv1(input1, output1, param1, ctx1);

    cudaStream_t cuda_stream1 = ctx1.get_compute_stream();
    out0.record_event(cuda_stream1);
    cudaStream_t cuda_stream2 = ctx1.get_compute_stream();
    out1.record_event(cuda_stream2);

    out0.sync();
    out1.sync();

    print_tensor_device(output_dev);
    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_func_conv_relu_fusion_speed) {

    int group = 1;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int kernel_h = 3;
    int kernel_w = 3;
    int g_out_channels = 32;

    int img_num = 1;
    int g_in_channels = 16;
    int img_h = 240;
    int img_w = 720;

    bool g_bias_term = true;

    LOG(INFO) << " conv param: ";
    LOG(INFO) << " img_num = " << img_num;
    LOG(INFO) << " g_in_channels = " << g_in_channels;
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
    LOG(INFO) << " g_out_channels = " << g_out_channels;

    Shape img_s(img_num, g_in_channels, img_h, img_w);
    Shape weights_s(g_out_channels, g_in_channels, kernel_h, kernel_w);
    Shape bias_s(1, g_out_channels, 1, 1);

    TensorHf4 img_host;
    TensorDf4 img_dev;
    TensorHf4 weights_host;
    TensorDf4 weights_dev;
    TensorHf4 bias_host;
    TensorDf4 bias_dev;
    TensorHf4 output_host;
    TensorDf4 output_dev;
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }

    img_dev.copy_from(img_host);
    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);
    fill_tensor_host_const(weights_host, 1.f);
    weights_dev.copy_from(weights_host);

    if (g_bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);
        fill_tensor_host_const(bias_host, 1.f);
        bias_dev.copy_from(bias_host);
    }

    LOG(INFO) << "test conv + relu";

    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(event);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;

    Context<NV> ctx1(0, 1, 1);
    ConvParam<TensorDf4> conv_param(group, pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    &weights_dev, &bias_dev);

    ActivationParam<TensorDf4> active_param(Active_relu);

    ConvActiveParam<TensorDf4> param(conv_param, active_param);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Conv_act<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> conv;
    conv.compute_output_shape(output, input, param);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());

    LOG(INFO) << "regular start with group = " << group;
    // init assume output tensor has been reshpaed by user.
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);

    conv(input, output, param, ctx1);

    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output[0]->record_event(cuda_stream);
    output_dev.sync();

    SaberTimer<NV> t1;
    int ts = 100;

    for (int i = 0; i < ts; ++i) {
        t1.start(ctx1);
        conv(input, output, param, ctx1);
        output[0]->record_event(cuda_stream);
        output_dev.sync();
        t1.end(ctx1);
    }

    LOG(INFO) << "fp32 conv_act elapse time: " << t1.get_average_ms() << " ms";

    CUDA_CHECK(cudaPeekAtLastError());
}

#endif
int main(int argc, const char** argv) {

    Env<NV>::env_init();

    LOG(INFO) << " conv param: ";
    LOG(INFO) << " img_num = " << img_num;
    LOG(INFO) << " g_in_channels = " << g_in_channels;
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
    LOG(INFO) << " g_out_channels = " << g_out_channels;

    LOG(INFO) << "tensor initialization finished";

    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

