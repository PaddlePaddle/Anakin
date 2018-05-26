#include "core/context.h"
#include "funcs/conv.h"
#include "funcs/conv_act.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

#define NORM_TEST 0

#if 0
using namespace anakin::saber;

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

typedef Tensor<X86, AK_INT8, NCHW> TensorHI84;
typedef Tensor<NV, AK_INT8, NCHW> TensorDI84;

typedef Tensor<X86, AK_INT8, NCHW_C4> TensorHC4;
typedef Tensor<NV, AK_INT8, NCHW_C4> TensorDC4;

#if NORM_TEST
TEST(TestSaberFuncNV, test_conv_int8_nchw_c4) {

    int group = 1;
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
    int img_h = 8;
    int img_w = 8;

    bool bias_term = false;

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

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);

    // This is the origin nchw version shape, This is a 4D dimension
    Shape origin_img_s(img_num, in_channels, img_h, img_w);

    // This is the nchw_c4 version shape,
    // This is a 5D dimension with last dim 4
    Shape img_s(img_num, in_channels / 4, img_h, img_w, 4);
    Shape weights_s(out_channels, in_channels / 4, kernel_h, kernel_w, 4);

    // Bias shape is always nchw version
    Shape bias_s(1, out_channels, 1, 1);

    // Origin img dev, this is nchw version.
    TensorDf4 origin_img_dev;
    origin_img_dev.re_alloc(origin_img_s);
    fill_tensor_device_const(origin_img_dev, 2);

    // Origin output dev, this is nchw version
    TensorDf4 origin_output_dev;

    // nchw_c4 version Input tensor:
    TensorDC4 img_dev;
    img_dev.re_alloc(img_s);

    // nchw_c4 version Output tensor:
    TensorDC4 output_dev;

    // we need to transformer, one transform input and the other transform output.
    // Transformer support stride. Shared Tensor must use Transformer.
    // step 0 update weights, get scales
    float in_scale = 2.f;
    std::vector<float> weight_scale(out_channels);

    for (int i = 0; i < weight_scale.size(); ++i) {
        weight_scale[i] = 1.f;
    }

    DataTensorTransformHelper trans;

    TensorHC4 weights_host;
    TensorDC4 weights_dev;

    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);

    fill_tensor_host_const(weights_host, 1);
    weights_dev.copy_from(weights_host);

    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(bias_host, 1);
        bias_dev.copy_from(bias_host);
    }

    ConvParam<TensorDC4> param(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);

    std::vector<TensorDC4*> input;
    std::vector<TensorDC4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Conv<NV, AK_INT8, AK_INT8, AK_INT8, NCHW_C4, NCHW_C4, NCHW_C4> conv;
    conv.compute_output_shape(input, output,param);

    output_dev.re_alloc(output[0]->shape());
    Shape output_shape{output_dev.num(), output_dev.channel(),
                       output_dev.height(), output_dev.width()};
    origin_output_dev.re_alloc(output_shape);

    // tensors must have shapes
    trans.init(in_scale, weight_scale, ctx1);

    // step 1, transform TensorDf4 to TensorDC4 using DataTransoformHelper
    trans.transform(img_dev, origin_img_dev, ctx1);

    // step 2, using transformed tensor as input, calling nchw_c4 version conv_int8
    // init assume output tensor has been reshpaed by user.
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    conv(input, output, param, ctx1);
    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output[0]->record_event(cuda_stream);
    output_dev.sync();

    // step 3, after get TensorDC4 output, using DataTensorTransfromHelper
    // get TensorDf4 tensor. The next layer can stay origin layout

    trans.transform(origin_output_dev, output_dev, ctx1);

    print_tensor_device(origin_output_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_conv_int8_nchw_c4_share) {

    int group = 1;
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
    int img_h = 8;
    int img_w = 8;

    bool bias_term = false;

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

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);

    // This is the origin nchw version shape, This is a 4D dimension
    Shape origin_img_s(img_num, in_channels, img_h, img_w);

    Shape origin_img_s_shared(img_num, in_channels, img_h / 2, img_w / 2);

    Shape img_s(img_num, in_channels / 4, img_h / 2, img_w / 2, 4);
    // This is the nchw_c4 version shape,
    // This is a 5D dimension with last dim 4
    Shape weights_s(out_channels, in_channels / 4, kernel_h, kernel_w, 4);

    // Bias shape is always nchw version
    Shape bias_s(1, out_channels, 1, 1);

    // Origin img dev, this is nchw version.
    TensorDf4 origin_img_dev;
    origin_img_dev.re_alloc(origin_img_s);
    fill_tensor_device_const(origin_img_dev, 2);
    TensorDC4 output_dev;
    // Origin shared tensor
    TensorDf4 origin_shared_t0;
    TensorDf4 origin_shared_t1;
    origin_shared_t0.share_sub_buffer(origin_img_dev, origin_img_s_shared, {0, 0, 0, 0});
    origin_shared_t1.share_sub_buffer(origin_img_dev, origin_img_s_shared, {0, 0, 4, 4});

    // Origin output dev, this is nchw version
    TensorDf4 origin_output_dev;
    origin_output_dev.re_alloc(origin_img_s);

    // Origin output shared tensor
    TensorDf4 origin_shared_out0;
    TensorDf4 origin_shared_out1;

    // nchw_c4 version Input tensor:
    TensorDC4 img_dev0;
    TensorDC4 img_dev1;
    img_dev0.re_alloc(img_s);
    img_dev1.re_alloc(img_s);

    // nchw_c4 version Output tensor:
    TensorDC4 output_dev0;
    TensorDC4 output_dev1;

    // we need to transformer, one transform input and the other transform output.
    // Transformer support stride. Shared Tensor must use Transformer.
    // step 0
    float in_scale = 1.f;
    std::vector<float> weight_scale(out_channels);

    for (int i = 0; i < weight_scale.size(); ++i) {
        weight_scale[i] = 1.f;
    }

    DataTensorTransformHelper trans0;
    DataTensorTransformHelper trans1;

    TensorHC4 weights_host;
    TensorDC4 weights_dev;

    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);

    fill_tensor_host_const(weights_host, 1);
    weights_dev.copy_from(weights_host);

    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(bias_host, 1);
        bias_dev.copy_from(bias_host);
    }

    ConvParam<TensorDC4> param(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);

    std::vector<TensorDC4*> input0, input1;
    std::vector<TensorDC4*> output0, output1;

    // input_v, output_v must not be shared tensor.
    input0.push_back(&img_dev0);
    input1.push_back(&img_dev1);

    output0.push_back(&output_dev0);
    output1.push_back(&output_dev1);

    Conv<NV, AK_INT8, AK_INT8, AK_INT8, NCHW_C4, NCHW_C4, NCHW_C4> conv0;
    Conv<NV, AK_INT8, AK_INT8, AK_INT8, NCHW_C4, NCHW_C4, NCHW_C4> conv1;
    conv0.compute_output_shape(input0, output0, param);
    conv1.compute_output_shape(input1, output1, param);

    output_dev0.re_alloc(output_dev0.shape());
    output_dev1.re_alloc(output_dev1.shape());

    Shape output_shape0{output_dev0.num(), output_dev0.channel(),
                        output_dev0.height(), output_dev0.width()};

    Shape output_shape1{output_dev1.num(), output_dev1.channel(),
                        output_dev1.height(), output_dev1.width()};

    origin_shared_out0.share_sub_buffer(origin_output_dev, output_shape0,
    {0, 0, 0, 0});
    origin_shared_out1.share_sub_buffer(origin_output_dev, output_shape1,
    {0, 0, 4, 4});

    // tensors must have shapes
    trans0.init(in_scale, weight_scale, ctx1);
    trans1.init(in_scale, weight_scale, ctx1);

    // step 1, transform TensorDf4 to TensorDC4 using DataTransoformHelper
    trans0.transform(img_dev0, origin_shared_t0, ctx1);
    trans1.transform(img_dev1, origin_shared_t1, ctx1);

    // step 2, using transformed tensor as input, calling nchw_c4 version conv_int8
    // init assume output tensor has been reshpaed by user.
    conv0.init(input0, output0, param, SPECIFY, VENDER_IMPL, ctx1);
    conv1.init(input1, output1, param, SPECIFY, VENDER_IMPL, ctx1);

    //    print_tensor_device(img_dev0);
    //    print_tensor_device(img_dev1);
    cudaDeviceSynchronize();

    conv0(input0, output0, param, ctx1);
    conv1(input1, output1, param, ctx1);

    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output0[0]->record_event(cuda_stream);
    output1[0]->record_event(cuda_stream);
    output_dev0.sync();
    output_dev1.sync();

    // step 3, after get TensorDC4 output, using DataTensorTransfromHelper
    // get TensorDf4 tensor. The next layer can stay origin layout

    trans0.transform(origin_shared_out0, output_dev0, ctx1);
    trans1.transform(origin_shared_out1, output_dev1, ctx1);

    print_tensor_device(origin_output_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_conv_act_int8_nchw_c4_share) {

    int group = 1;
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

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);

    // This is the origin nchw version shape, This is a 4D dimension
    Shape origin_img_s(img_num, in_channels, img_h, img_w);

    Shape origin_img_s_shared(img_num, in_channels, img_h / 2, img_w / 2);

    Shape img_s(img_num, in_channels / 4, img_h / 2, img_w / 2, 4);
    // This is the nchw_c4 version shape,
    // This is a 5D dimension with last dim 4
    Shape weights_s(out_channels, in_channels / 4, kernel_h, kernel_w, 4);

    // Bias shape is always nchw version
    Shape bias_s(1, out_channels, 1, 1);

    // Origin img dev, this is nchw version.
    TensorDf4 origin_img_dev;
    origin_img_dev.re_alloc(origin_img_s);
    fill_tensor_device_const(origin_img_dev, 2);
    TensorDC4 output_dev;

    // Origin shared tensor
    TensorDf4 origin_shared_t0;
    TensorDf4 origin_shared_t1;

    origin_shared_t0.share_sub_buffer(origin_img_dev, origin_img_s_shared, {0, 0, 0, 0});
    origin_shared_t1.share_sub_buffer(origin_img_dev, origin_img_s_shared, {0, 0, 4, 4});

    // Origin output dev, this is nchw version
    TensorDf4 origin_output_dev;
    origin_output_dev.re_alloc(origin_img_s);

    // Origin output shared tensor
    TensorDf4 origin_shared_out0;
    TensorDf4 origin_shared_out1;

    // nchw_c4 version Input tensor:
    TensorDC4 img_dev0;
    TensorDC4 img_dev1;
    img_dev0.re_alloc(img_s);
    img_dev1.re_alloc(img_s);

    // nchw_c4 version Output tensor:
    TensorDC4 output_dev0;
    TensorDC4 output_dev1;

    // we need to transformer, one transform input and the other transform output.
    // Transformer support stride. Shared Tensor must use Transformer.
    // step 0 update weights, get scales
    float in_scale = 1.f;
    std::vector<float> weight_scale(out_channels);

    for (int i = 0; i < weight_scale.size(); ++i) {
        weight_scale[i] = 1.f;
    }

    DataTensorTransformHelper trans0;
    DataTensorTransformHelper trans1;

    TensorHC4 weights_host;
    TensorDC4 weights_dev;

    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);

    fill_tensor_host_const(weights_host, 1);
    weights_dev.copy_from(weights_host);

    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(bias_host, 1);
        bias_dev.copy_from(bias_host);
    }

    ConvParam<TensorDC4> conv_param(group, pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    &weights_dev, &bias_dev);
    ActivationParam<TensorDC4> act_param(Active_relu);

    ConvActiveParam<TensorDC4> param(conv_param, act_param);

    std::vector<TensorDC4*> input0, input1;
    std::vector<TensorDC4*> output0, output1;

    // input_v, output_v must not be shared tensor.
    input0.push_back(&img_dev0);
    input1.push_back(&img_dev1);

    output0.push_back(&output_dev0);
    output1.push_back(&output_dev1);

    Conv_act<NV, AK_INT8, AK_INT8, AK_INT8, NCHW_C4, NCHW_C4, NCHW_C4> conv0;
    Conv_act<NV, AK_INT8, AK_INT8, AK_INT8, NCHW_C4, NCHW_C4, NCHW_C4> conv1;
    conv0.compute_output_shape(input0, output0, param);
    conv1.compute_output_shape(input1, output1, param);

    output_dev0.re_alloc(output_dev0.shape());
    output_dev1.re_alloc(output_dev1.shape());

    Shape output_shape0{output_dev0.num(), output_dev0.channel(),
                        output_dev0.height(), output_dev0.width()};

    Shape output_shape1{output_dev1.num(), output_dev1.channel(),
                        output_dev1.height(), output_dev1.width()};

    origin_shared_out0.share_sub_buffer(origin_output_dev, output_shape0,
    {0, 0, 0, 0});
    origin_shared_out1.share_sub_buffer(origin_output_dev, output_shape1,
    {0, 0, 4, 4});

    // tensors must have shapes
    trans0.init(in_scale, weight_scale, ctx1);
    trans1.init(in_scale, weight_scale, ctx1);

    // step 1, transform TensorDf4 to TensorDC4 using DataTransoformHelper
    trans0.transform(img_dev0, origin_shared_t0, ctx1);
    trans1.transform(img_dev1, origin_shared_t1, ctx1);

    // step 2, using transformed tensor as input, calling nchw_c4 version conv_int8
    // init assume output tensor has been reshpaed by user.
    conv0.init(input0, output0, param, SPECIFY, VENDER_IMPL, ctx1);
    conv1.init(input1, output1, param, SPECIFY, VENDER_IMPL, ctx1);

    //    print_tensor_device(img_dev0);
    //    print_tensor_device(img_dev1);
    cudaDeviceSynchronize();

    conv0(input0, output0, param, ctx1);
    conv1(input1, output1, param, ctx1);

    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output0[0]->record_event(cuda_stream);
    output1[0]->record_event(cuda_stream);
    output_dev0.sync();
    output_dev1.sync();

    // step 3, after get TensorDC4 output, using DataTensorTransfromHelper
    // get TensorDf4 tensor. The next layer can stay origin layout

    trans0.transform(origin_shared_out0, output_dev0, ctx1);
    trans1.transform(origin_shared_out1, output_dev1, ctx1);

    print_tensor_device(origin_output_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_conv_act_int8_nchw_c4_speed_test) {

    int group = 1;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int kernel_h = 3;
    int kernel_w = 3;
    int out_channels = 32;

    int img_num = 1;
    int in_channels = 16;
    int img_h = 240;
    int img_w = 720;

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

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);

    // This is the origin nchw version shape, This is a 4D dimension
    Shape origin_img_s(img_num, in_channels, img_h, img_w);

    // This is the nchw_c4 version shape,
    // This is a 5D dimension with last dim 4
    Shape img_s(img_num, in_channels / 4, img_h, img_w, 4);
    Shape weights_s(out_channels, in_channels / 4, kernel_h, kernel_w, 4);

    // Bias shape is always nchw version
    Shape bias_s(1, out_channels, 1, 1);

    // Origin img dev, this is nchw version.
    TensorDf4 origin_img_dev;
    origin_img_dev.re_alloc(origin_img_s);
    fill_tensor_device_const(origin_img_dev, 2);

    // Origin output dev, this is nchw version
    TensorDf4 origin_output_dev;

    // nchw_c4 version Input tensor:
    TensorDC4 img_dev;
    img_dev.re_alloc(img_s);

    // nchw_c4 version Output tensor:
    TensorDC4 output_dev;

    // we need to transformer, one transform input and the other transform output.
    // Transformer support stride. Shared Tensor must use Transformer.
    // step 0 update weights, get scales
    float in_scale = 1.f;
    std::vector<float> weight_scale(out_channels);

    for (int i = 0; i < weight_scale.size(); ++i) {
        weight_scale[i] = 1.f;
    }

    DataTensorTransformHelper trans;

    TensorHC4 weights_host;
    TensorDC4 weights_dev;

    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);

    fill_tensor_host_const(weights_host, 1);
    weights_dev.copy_from(weights_host);

    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(bias_host, 1);
        bias_dev.copy_from(bias_host);
    }

    ConvParam<TensorDC4> conv_param(group, pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    &weights_dev, &bias_dev);
    ActivationParam<TensorDC4> act_param(Active_relu);
    ConvActiveParam<TensorDC4> param(conv_param, act_param);

    std::vector<TensorDC4*> input;
    std::vector<TensorDC4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Conv_act<NV, AK_INT8, AK_INT8, AK_INT8, NCHW_C4, NCHW_C4, NCHW_C4> conv;
    conv.compute_output_shape(input, output, param);

    output_dev.re_alloc(output[0]->shape());
    Shape output_shape{output_dev.num(), output_dev.channel(),
                       output_dev.height(), output_dev.width()};
    origin_output_dev.re_alloc(output_shape);

    // tensors must have shapes
    trans.init(in_scale, weight_scale, ctx1);

    SaberTimer<NV> in_t, out_t, infer_t;
    int ts = 100;
    cudaStream_t cuda_stream = ctx1.get_compute_stream();

    // step 1, transform TensorDf4 to TensorDC4 using DataTransoformHelper
    trans.transform(img_dev, origin_img_dev, ctx1);
    img_dev.record_event(cuda_stream);
    img_dev.sync();

    for (int i = 0; i < ts; ++i) {
        in_t.start(ctx1);
        trans.transform(img_dev, origin_img_dev, ctx1);
        img_dev.record_event(cuda_stream);
        img_dev.sync();
        in_t.end(ctx1);
    }

    LOG(INFO) << "in trans time: " << in_t.get_average_ms() << " ms";

    // step 2, using transformed tensor as input, calling nchw_c4 version conv_int8
    // init assume output tensor has been reshpaed by user.
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    conv(input, output, param, ctx1);

    for (int i = 0; i < ts; ++i) {
        infer_t.start(ctx1);
        conv(input, output, param, ctx1);
        output[0]->record_event(cuda_stream);
        output_dev.sync();
        infer_t.end(ctx1);
    }

    LOG(INFO) << "infer time: " << infer_t.get_average_ms() << " ms";

    // step 3, after get TensorDC4 output, using DataTensorTransfromHelper
    // get TensorDf4 tensor. The next layer can stay origin layout

    trans.transform(origin_output_dev, output_dev, ctx1);
    origin_output_dev.record_event(cuda_stream);
    origin_output_dev.sync();

    for (int i = 0; i < ts; ++i) {
        out_t.start(ctx1);
        trans.transform(origin_output_dev, output_dev, ctx1);
        origin_output_dev.record_event(cuda_stream);
        origin_output_dev.sync();
        out_t.end(ctx1);
    }

    LOG(INFO) << "out trans time: " << out_t.get_average_ms() << " ms";

    //    print_tensor_device(origin_output_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_results) {

    int group = 1;
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

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);

    // This is the origin nchw version shape, This is a 4D dimension
    Shape origin_img_s(img_num, in_channels, img_h, img_w);
    Shape img_s(img_num, in_channels / 4, img_h, img_w, 4);
    Shape origin_weights_s(out_channels, in_channels, kernel_h, kernel_w);
    Shape weights_s(out_channels, in_channels / 4, kernel_h, kernel_w, 4);

    // Bias shape is always nchw version
    Shape bias_s(1, out_channels, 1, 1);

    // Origin img dev, this is nchw version.
    TensorDf4 origin_img_dev;
    origin_img_dev.re_alloc(origin_img_s);
    fill_tensor_device_const(origin_img_dev, 2);
    //    fill_tensor_device_rand(origin_img_dev, 0, 0.5, ctx1.get_compute_stream());
    origin_img_dev.record_event(ctx1.get_compute_stream());
    origin_img_dev.sync();

    // Origin output dev, this is nchw version
    TensorDf4 origin_output_dev;

    // nchw_c4 version Input tensor:
    TensorDC4 img_dev;
    img_dev.re_alloc(img_s);

    // nchw_c4 version Output tensor:
    TensorDC4 output_dev;

    // we need to transformer, one transform input and the other transform output.
    // Transformer support stride. Shared Tensor must use Transformer.
    // step 0 update weights, get scales
    float in_scale = 0.5f;
    DataTensorTransformHelper trans(in_scale);

    TensorHf4 origin_weights_host;
    TensorDf4 origin_weights_dev;
    TensorHC4 weights_host;
    TensorDC4 weights_dev;

    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);
    origin_weights_host.re_alloc(origin_weights_s);
    origin_weights_dev.re_alloc(origin_weights_s);

    //    fill_tensor_host_const(origin_weights_host, 2);
    fill_tensor_host_rand(origin_weights_host, 0, 0.8);
    trans.convert_weights(weights_host, origin_weights_host, ctx1);

    weights_dev.copy_from(weights_host);
    origin_weights_dev.copy_from(origin_weights_host);

    TensorHf4 origin_bias_host;
    TensorDf4 origin_bias_dev;
    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        origin_bias_host.re_alloc(bias_s);
        origin_bias_dev.re_alloc(bias_s);
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(origin_bias_host, 1);
        trans.convert_bias(bias_host, origin_bias_host, ctx1);
        bias_dev.copy_from(bias_host);
        origin_bias_dev.copy_from(origin_bias_host);
    }

    ConvParam<TensorDC4> param(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);

    std::vector<TensorDC4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&origin_output_dev);

    Conv<NV, AK_INT8, AK_INT8, AK_FLOAT, NCHW_C4, NCHW_C4, NCHW> conv;
    conv.compute_output_shape(input, output, param);

    origin_output_dev.re_alloc(output[0]->shape());
    // tensors must have shapes
    trans.init(ctx1);

    // step 1, transform TensorDf4 to TensorDC4 using DataTransoformHelper
    trans.transform(img_dev, origin_img_dev, ctx1);
    img_dev.record_event(ctx1.get_compute_stream());
    img_dev.sync();

    // step 2, using transformed tensor as input, calling nchw_c4 version conv_int8
    // init assume output tensor has been reshpaed by user.
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    conv(input, output, param, ctx1);

    output[0]->record_event(ctx1.get_compute_stream());
    output_dev.sync();

    // step 3, after get TensorDC4 output, using DataTensorTransfromHelper
    // get TensorDf4 tensor. The next layer can stay origin layout

    trans.transform(origin_output_dev, origin_output_dev, ctx1);
    origin_output_dev.record_event(ctx1.get_compute_stream());
    origin_output_dev.sync();

    print_tensor_device(origin_output_dev);

    ConvParam<TensorDf4> origin_param(group, pad_h, pad_w,
                                      stride_h, stride_w,
                                      dilation_h, dilation_w,
                                      &origin_weights_dev, &origin_bias_dev);

    Conv<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> origin_conv;
    std::vector<TensorDf4*> origin_input;
    std::vector<TensorDf4*> origin_output;
    TensorDf4 check_results;
    origin_input.push_back(&origin_img_dev);
    origin_output.push_back(&check_results);

    origin_conv.compute_output_shape(origin_input, origin_output, origin_param);
    check_results.re_alloc(origin_output[0]->shape());

    origin_conv.init(origin_input, origin_output, origin_param, SPECIFY, VENDER_IMPL, ctx1);

    origin_conv(origin_input, origin_output, origin_param, ctx1);
    origin_output[0]->record_event(ctx1.get_compute_stream());

    check_results.sync();
    print_tensor_device(check_results);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_conv_act_results) {

    int group = 1;
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

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);

    // This is the origin nchw version shape, This is a 4D dimension
    Shape origin_img_s(img_num, in_channels, img_h, img_w);
    Shape img_s(img_num, in_channels / 4, img_h, img_w, 4);
    Shape origin_weights_s(out_channels, in_channels, kernel_h, kernel_w);
    Shape weights_s(out_channels, in_channels / 4, kernel_h, kernel_w, 4);

    // Bias shape is always nchw version
    Shape bias_s(1, out_channels, 1, 1);

    // Origin img dev, this is nchw version.
    TensorDf4 origin_img_dev;
    origin_img_dev.re_alloc(origin_img_s);
    fill_tensor_device_const(origin_img_dev, 2);
    //    fill_tensor_device_rand(origin_img_dev, 0, 0.5, ctx1.get_compute_stream());
    origin_img_dev.record_event(ctx1.get_compute_stream());
    origin_img_dev.sync();

    // Origin output dev, this is nchw version
    TensorDf4 origin_output_dev;

    // nchw_c4 version Input tensor:
    TensorDC4 img_dev;
    img_dev.re_alloc(img_s);

    // nchw_c4 version Output tensor:
    TensorDC4 output_dev;

    // we need to transformer, one transform input and the other transform output.
    // Transformer support stride. Shared Tensor must use Transformer.
    // step 0 update weights, get scales
    float in_scale = 0.5f;
    DataTensorTransformHelper trans(in_scale);

    TensorHf4 origin_weights_host;
    TensorDf4 origin_weights_dev;
    TensorHC4 weights_host;
    TensorDC4 weights_dev;

    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);
    origin_weights_host.re_alloc(origin_weights_s);
    origin_weights_dev.re_alloc(origin_weights_s);

    //    fill_tensor_host_const(origin_weights_host, 2);
    fill_tensor_host_rand(origin_weights_host, 0, 0.8);
    trans.convert_weights(weights_host, origin_weights_host, ctx1);

    weights_dev.copy_from(weights_host);
    origin_weights_dev.copy_from(origin_weights_host);

    TensorHf4 origin_bias_host;
    TensorDf4 origin_bias_dev;
    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        origin_bias_host.re_alloc(bias_s);
        origin_bias_dev.re_alloc(bias_s);
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(origin_bias_host, 1);
        trans.convert_bias(bias_host, origin_bias_host, ctx1);
        bias_dev.copy_from(bias_host);
        origin_bias_dev.copy_from(origin_bias_host);
    }

    ActivationParam<TensorDC4> act_param(Active_relu);
    ConvParam<TensorDC4> conv_param(group, pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    &weights_dev, &bias_dev);

    ConvActiveParam<TensorDC4> param(conv_param, act_param);
    std::vector<TensorDC4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&origin_output_dev);

    Conv_act<NV, AK_INT8, AK_INT8, AK_FLOAT, NCHW_C4, NCHW_C4, NCHW> conv;
    conv.compute_output_shape(input, output, param);

    origin_output_dev.re_alloc(output[0]->shape());
    // tensors must have shapes
    trans.init(ctx1);

    // step 1, transform TensorDf4 to TensorDC4 using DataTransoformHelper
    trans.transform(img_dev, origin_img_dev, ctx1);
    img_dev.record_event(ctx1.get_compute_stream());
    img_dev.sync();

    // step 2, using transformed tensor as input, calling nchw_c4 version conv_int8
    // init assume output tensor has been reshpaed by user.
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    conv(input, output, param, ctx1);
    output[0]->record_event(ctx1.get_compute_stream());
    output_dev.sync();

    // step 3, after get TensorDC4 output, using DataTensorTransfromHelper
    // get TensorDf4 tensor. The next layer can stay origin layout

    trans.transform(origin_output_dev, origin_output_dev, ctx1);
    origin_output_dev.record_event(ctx1.get_compute_stream());
    origin_output_dev.sync();
    print_tensor_device(origin_output_dev);

    ActivationParam<TensorDf4> origin_act_param(Active_relu);
    ConvParam<TensorDf4> origin_conv_param(group, pad_h, pad_w,
                                           stride_h, stride_w,
                                           dilation_h, dilation_w,
                                           &origin_weights_dev, &origin_bias_dev);
    ConvActiveParam<TensorDf4> origin_param(origin_conv_param, origin_act_param);

    Conv_act<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> origin_conv;
    std::vector<TensorDf4*> origin_input;
    std::vector<TensorDf4*> origin_output;
    TensorDf4 check_results;
    origin_input.push_back(&origin_img_dev);
    origin_output.push_back(&check_results);
    origin_conv.compute_output_shape(origin_input, origin_output, origin_param);
    check_results.re_alloc(origin_output[0]->shape());
    origin_conv.init(origin_input, origin_output, origin_param, SPECIFY, VENDER_IMPL, ctx1);

    origin_conv(origin_input, origin_output, origin_param, ctx1);
    origin_output[0]->record_event(ctx1.get_compute_stream());

    check_results.sync();
    print_tensor_device(check_results);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}
#else
void test_conv_int8_int8_speed(int img_num, int kernel, int out_channels,
                               int in_channels, int img_h, int img_w, int pad, int stride) {
    int group = 1;
    int pad_h = pad;
    int pad_w = pad;
    int stride_h = stride;
    int stride_w = stride;
    int dilation_h = 1;
    int dilation_w = 1;

    int kernel_h = kernel;
    int kernel_w = kernel;

    bool bias_term = false;
    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);

    // This is the origin nchw version shape, This is a 4D dimension
    Shape origin_img_s(img_num, in_channels, img_h, img_w);

    // This is the nchw_c4 version shape,
    // This is a 5D dimension with last dim 4
    Shape img_s(img_num, in_channels / 4, img_h, img_w, 4);
    Shape weights_s(out_channels, in_channels / 4, kernel_h, kernel_w, 4);

    // Bias shape is always nchw version
    Shape bias_s(1, out_channels, 1, 1);

    // Origin img dev, this is nchw version.
    TensorDf4 origin_img_dev;
    origin_img_dev.re_alloc(origin_img_s);
    fill_tensor_device_const(origin_img_dev, 2);

    // Origin output dev, this is nchw version
    TensorDf4 origin_output_dev;

    // nchw_c4 version Input tensor:
    TensorDC4 img_dev;
    img_dev.re_alloc(img_s);

    // nchw_c4 version Output tensor:
    TensorDC4 output_dev;

    // we need to transformer, one transform input and the other transform output.
    // Transformer support stride. Shared Tensor must use Transformer.
    // step 0 update weights, get scales
    float in_scale = 1.f;
    std::vector<float> weight_scale(out_channels);

    for (int i = 0; i < weight_scale.size(); ++i) {
        weight_scale[i] = 1.f;
    }

    DataTensorTransformHelper trans;

    TensorHC4 weights_host;
    TensorDC4 weights_dev;

    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);

    fill_tensor_host_const(weights_host, 1);
    weights_dev.copy_from(weights_host);

    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(bias_host, 1);
        bias_dev.copy_from(bias_host);
    }

    ConvParam<TensorDC4> param(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);

    std::vector<TensorDC4*> input;
    std::vector<TensorDC4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Conv<NV, AK_INT8, AK_INT8, AK_INT8, NCHW_C4, NCHW_C4, NCHW_C4> conv;
    conv.compute_output_shape(output, input, param);

    output_dev.re_alloc(output[0]->shape());
    Shape output_shape{output_dev.num(), output_dev.channel(),
                       output_dev.height(), output_dev.width()};
    origin_output_dev.re_alloc(output_shape);

    // tensors must have shapes
    trans.init(in_scale, weight_scale, ctx1);

    SaberTimer<NV> in_t, out_t, infer_t;
    int ts = 100;
    cudaStream_t cuda_stream = ctx1.get_compute_stream();

    // step 1, transform TensorDf4 to TensorDC4 using DataTransoformHelper
    trans.transform(img_dev, origin_img_dev, ctx1);
    img_dev.record_event(cuda_stream);
    img_dev.sync();

    for (int i = 0; i < ts; ++i) {
        in_t.start(ctx1);
        trans.transform(img_dev, origin_img_dev, ctx1);
        img_dev.record_event(cuda_stream);
        img_dev.sync();
        in_t.end(ctx1);
    }

    std::cout << in_t.get_average_ms() << ", ";

    // step 2, using transformed tensor as input, calling nchw_c4 version conv_int8
    // init assume output tensor has been reshpaed by user.
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    conv(input, output, param, ctx1);
    output[0]->record_event(cuda_stream);
    output_dev.sync();
    cudaDeviceSynchronize();

    for (int i = 0; i < ts; ++i) {
        infer_t.start(ctx1);
        conv(input, output, param, ctx1);
        output[0]->record_event(cuda_stream);
        output_dev.sync();
        infer_t.end(ctx1);
    }

    std::cout << infer_t.get_average_ms() << ", ";

    // step 3, after get TensorDC4 output, using DataTensorTransfromHelper
    // get TensorDf4 tensor. The next layer can stay origin layout

    trans.transform(origin_output_dev, output_dev, ctx1);
    origin_output_dev.record_event(cuda_stream);
    origin_output_dev.sync();

    for (int i = 0; i < ts; ++i) {
        out_t.start(ctx1);
        trans.transform(origin_output_dev, output_dev, ctx1);
        origin_output_dev.record_event(cuda_stream);
        origin_output_dev.sync();
        out_t.end(ctx1);
    }

    std::cout << out_t.get_average_ms() << ", ";

    //    print_tensor_device(origin_output_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}
void test_conv_int8_fp32_speed(int img_num, int kernel, int out_channels,
                               int in_channels, int img_h, int img_w, int pad, int stride) {
    int group = 1;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int kernel_h = kernel;
    int kernel_w = kernel;

    bool bias_term = false;

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);

    // This is the origin nchw version shape, This is a 4D dimension
    Shape origin_img_s(img_num, in_channels, img_h, img_w);

    // This is the nchw_c4 version shape,
    // This is a 5D dimension with last dim 4
    Shape img_s(img_num, in_channels / 4, img_h, img_w, 4);
    Shape weights_s(out_channels, in_channels / 4, kernel_h, kernel_w, 4);

    // Bias shape is always nchw version
    Shape bias_s(1, out_channels, 1, 1);

    // Origin img dev, this is nchw version.
    TensorDf4 origin_img_dev;
    origin_img_dev.re_alloc(origin_img_s);
    fill_tensor_device_const(origin_img_dev, 2);

    // Origin output dev, this is nchw version
    TensorDf4 origin_output_dev;

    // nchw_c4 version Input tensor:
    TensorDC4 img_dev;
    img_dev.re_alloc(img_s);

    // nchw_c4 version Output tensor:
    TensorDf4 output_dev;

    // we need to transformer, one transform input and the other transform output.
    // Transformer support stride. Shared Tensor must use Transformer.
    // step 0 update weights, get scales
    float in_scale = 1.f;
    std::vector<float> weight_scale(out_channels);

    for (int i = 0; i < weight_scale.size(); ++i) {
        weight_scale[i] = 1.f;
    }

    DataTensorTransformHelper trans;

    TensorHC4 weights_host;
    TensorDC4 weights_dev;

    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);

    fill_tensor_host_const(weights_host, 1);
    weights_dev.copy_from(weights_host);

    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(bias_host, 1);
        bias_dev.copy_from(bias_host);
    }

    ConvParam<TensorDC4> param(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);

    std::vector<TensorDC4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Conv<NV, AK_INT8, AK_INT8, AK_FLOAT, NCHW_C4, NCHW_C4, NCHW> conv;
    conv.compute_output_shape(output, input, param);

    output_dev.re_alloc(output[0]->shape());
    Shape output_shape{output_dev.num(), output_dev.channel(),
                       output_dev.height(), output_dev.width()};
    origin_output_dev.re_alloc(output_shape);

    // tensors must have shapes
    trans.init(in_scale, weight_scale, ctx1);

    SaberTimer<NV> in_t, out_t, infer_t;
    int ts = 100;
    cudaStream_t cuda_stream = ctx1.get_compute_stream();

    // step 1, transform TensorDf4 to TensorDC4 using DataTransoformHelper
    trans.transform(img_dev, origin_img_dev, ctx1);
    img_dev.record_event(cuda_stream);
    img_dev.sync();
    cudaDeviceSynchronize();

    for (int i = 0; i < ts; ++i) {
        in_t.start(ctx1);
        trans.transform(img_dev, origin_img_dev, ctx1);
        img_dev.record_event(cuda_stream);
        img_dev.sync();
        in_t.end(ctx1);
    }

    std::cout << in_t.get_average_ms() << ", ";

    // step 2, using transformed tensor as input, calling nchw_c4 version conv_int8
    // init assume output tensor has been reshpaed by user.
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    conv(input, output, param, ctx1);
    output[0]->record_event(cuda_stream);
    output_dev.sync();
    cudaDeviceSynchronize();

    for (int i = 0; i < ts; ++i) {
        infer_t.start(ctx1);
        conv(input, output, param, ctx1);
        output[0]->record_event(cuda_stream);
        output_dev.sync();
        infer_t.end(ctx1);
    }

    std::cout << infer_t.get_average_ms() << ", ";

    // step 3, after get TensorDC4 output, using DataTensorTransfromHelper
    // get TensorDf4 tensor. The next layer can stay origin layout

    trans.transform(origin_output_dev, output_dev, ctx1);
    origin_output_dev.record_event(cuda_stream);
    origin_output_dev.sync();
    cudaDeviceSynchronize();

    for (int i = 0; i < ts; ++i) {
        out_t.start(ctx1);
        trans.transform(origin_output_dev, output_dev, ctx1);
        origin_output_dev.record_event(cuda_stream);
        origin_output_dev.sync();
        out_t.end(ctx1);
    }

    std::cout << out_t.get_average_ms() << ", ";

    //    print_tensor_device(origin_output_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

void test_conv_fp32_speed(int img_num, int kernel, int out_channels,
                          int in_channels, int img_h, int img_w, int pad, int stride) {
    int group = 1;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int kernel_h = kernel;
    int kernel_w = kernel;

    bool bias_term = false;

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);

    // This is the origin nchw version shape, This is a 4D dimension
    Shape origin_img_s(img_num, in_channels, img_h, img_w);

    // This is the nchw_c4 version shape,
    // This is a 5D dimension with last dim 4
    Shape img_s(img_num, in_channels, img_h, img_w);
    Shape weights_s(out_channels, in_channels, kernel_h, kernel_w);

    // Bias shape is always nchw version
    Shape bias_s(1, out_channels, 1, 1);

    // Origin img dev, this is nchw version.
    TensorDf4 origin_img_dev;
    origin_img_dev.re_alloc(origin_img_s);
    fill_tensor_device_const(origin_img_dev, 2);

    // Origin output dev, this is nchw version
    TensorDf4 origin_output_dev;

    // nchw_c4 version Input tensor:
    TensorDf4 img_dev;
    img_dev.re_alloc(img_s);

    // nchw_c4 version Output tensor:
    TensorDf4 output_dev;

    // we need to transformer, one transform input and the other transform output.
    // Transformer support stride. Shared Tensor must use Transformer.
    // step 0 update weights, get scales
    float in_scale = 1.f;
    std::vector<float> weight_scale(out_channels);

    for (int i = 0; i < weight_scale.size(); ++i) {
        weight_scale[i] = 1.f;
    }

    DataTensorTransformHelper trans;

    TensorHf4 weights_host;
    TensorDf4 weights_dev;

    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);

    fill_tensor_host_const(weights_host, 1);
    weights_dev.copy_from(weights_host);

    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(bias_host, 1);
        bias_dev.copy_from(bias_host);
    }

    ConvParam<TensorDf4> param(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Conv<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> conv;
    conv.compute_output_shape(output, input, param);

    output_dev.re_alloc(output[0]->shape());
    Shape output_shape{output_dev.num(), output_dev.channel(),
                       output_dev.height(), output_dev.width()};
    origin_output_dev.re_alloc(output_shape);

    // tensors must have shapes
    trans.init(in_scale, weight_scale, ctx1);

    SaberTimer<NV> in_t, out_t, infer_t;
    int ts = 100;
    cudaStream_t cuda_stream = ctx1.get_compute_stream();

    // step 1, transform TensorDf4 to TensorDf4 using DataTransoformHelper
    //    trans.transform(img_dev, origin_img_dev, ctx1);
    //    img_dev.record_event(cuda_stream);
    //    img_dev.sync();
    //    cudaDeviceSynchronize();
    //
    //    for (int i = 0; i < ts; ++i) {
    //        in_t.start(ctx1);
    //        trans.transform(img_dev, origin_img_dev, ctx1);
    //        img_dev.record_event(cuda_stream);
    //        img_dev.sync();
    //        in_t.end(ctx1);
    //    }
    //    std::cout<<in_t.get_average_ms()<<", ";

    // step 2, using transformed tensor as input, calling nchw_c4 version conv_int8
    // init assume output tensor has been reshpaed by user.
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    conv(input, output, param, ctx1);
    output[0]->record_event(cuda_stream);
    output_dev.sync();
    cudaDeviceSynchronize();

    for (int i = 0; i < ts; ++i) {
        infer_t.start(ctx1);
        conv(input, output, param, ctx1);
        output[0]->record_event(cuda_stream);
        output_dev.sync();
        infer_t.end(ctx1);
    }

    std::cout << infer_t.get_average_ms() << ", ";

    // step 3, after get TensorDf4 output, using DataTensorTransfromHelper
    // get TensorDf4 tensor. The next layer can stay origin layout

    //    trans.transform(origin_output_dev, output_dev, ctx1);
    //    origin_output_dev.record_event(cuda_stream);
    //    origin_output_dev.sync();
    //    cudaDeviceSynchronize();
    //
    //    for (int i = 0; i < ts; ++i) {
    //        out_t.start(ctx1);
    //        trans.transform(origin_output_dev, output_dev, ctx1);
    //        origin_output_dev.record_event(cuda_stream);
    //        origin_output_dev.sync();
    //        out_t.end(ctx1);
    //    }
    //    std::cout<<out_t.get_average_ms()<<", ";

    //    print_tensor_device(origin_output_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_conv_int8_nchw_c4_speed_test) {

    //    std::cout<<"kernel_size, img_h, img_w, out_channels, in_channels, pad, stride,"
    //            " out_int8_in_trans, out_int8_op_time, out_int8_out_trans,"
    //            " out_fp32_in_trans, out_fp32_op_time, out_fp32_out_trans,"<<std::endl;

    std::cout << "img_num, kernel_size, img_h, img_w, out_channels, in_channels, pad, stride,"
              " optime," << std::endl;
    int pad = 0;
    int stride = 1;

    for (int img_n = 1; img_n < 32; img_n *= 2) {
        for (int k = 1; k <= 3; k += 2) { // kernel size
            for (int o = 64; o <= 128; o += 64) { // out_channels
                for (int i = 128; i <= 256; i += 64) { // in_channels
                    for (int h = 200; h <= 200; h += 400) { // img_h
                        for (int w = 300; w <= 300; w += 400) { // img_w
                            if (k == 3) {
                                pad = 1;
                            }

                            std::cout << img_n << ", " << k << ", " << h << ", " << w << ", "
                                      << o << ", " << i << ", " << pad << ", " << stride << ", ";
                            //                        test_conv_int8_int8_speed(img_n, k, o, i, h, w, 0, 1);
                            //                            test_conv_int8_fp32_speed(img_n, k, o, i, h, w, 0, 1);
                            test_conv_fp32_speed(img_n, k, o, i, h, w, 0, 1);
                            std::cout << std::endl;
                        }
                    }
                }
            }
        }
    }
}

#endif

#endif
int main(int argc, const char** argv) {
#if 0
    // initial logger
    //logger::init(argv[0]);
    Env<NV>::env_init();

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
#endif    
}

