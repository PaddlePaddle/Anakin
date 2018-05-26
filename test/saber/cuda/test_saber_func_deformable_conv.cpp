
#include "core/context.h"
#include "funcs/deformable_conv.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

TEST(TestSaberFuncNV, test_deformable_conv) {

    int group = 1;
    int pad_h = 2;
    int pad_w = 2;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 2;
    int dilation_w = 2;

    int kernel_h = 3;
    int kernel_w = 3;
    int out_channels = 6;

    int img_num = 1;
    int in_channels = 4;
    int img_h = 4;
    int img_w = 4;

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
    Shape offset_s(img_num, in_channels * 8, img_h, img_w);
    Shape weights_s(out_channels, in_channels, kernel_h, kernel_w);
    Shape bias_s(1, out_channels, 1, 1);

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);

    TensorDf4 img_dev;
    img_dev.re_alloc(img_s);
    fill_tensor_device_const(img_dev, 2.f);

    TensorDf4 offset_dev;
    offset_dev.re_alloc(offset_s);
    fill_tensor_device_const(offset_dev, 1.f);

    TensorDf4 weights_dev;
    weights_dev.re_alloc(weights_s);
    fill_tensor_device_const(weights_dev, 1.1f);

    TensorDf4 bias_dev;

    if (bias_term) {
        bias_dev.re_alloc(bias_s);
        fill_tensor_device_const(bias_dev, 1.f);
    }

    TensorDf4 output_dev;
    DeformableConvParam<TensorDf4> param(group, pad_h, pad_w,
                                         stride_h, stride_w,
                                         dilation_h, dilation_w,
                                         &weights_dev, &bias_dev);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    input.push_back(&offset_dev);

    output.push_back(&output_dev);

    DeformableConv<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> deformable_conv;
    deformable_conv.compute_output_shape(input, output, param);
    output_dev.re_alloc(output[0]->shape());

    deformable_conv.init(input, output, param, SPECIFY, SABER_IMPL, ctx1);
    deformable_conv(input, output, param, ctx1);

    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output[0]->record_event(cuda_stream);

    output_dev.sync();
    print_tensor_device(img_dev);
    print_tensor_device(offset_dev);
    print_tensor_device(output_dev);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

int main(int argc, const char** argv) {
    anakin::saber::Env<NV>::env_init();

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

