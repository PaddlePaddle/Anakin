#include "core/context.h"
#include "funcs/activation.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

template <typename Tensor>
void print_tensor_shape(std::string name, Tensor& t0) {

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

TEST(TestSaberFuncNV, test_func_constructor) {

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    int img_num = 1;
    int in_channels = 2;
    int img_h = 8;
    int img_w = 8;

    Shape img_s(img_num, in_channels, img_h, img_w);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = (float)(0.05 * (i & 0x1f) * -1);
    }

    img_dev.copy_from(img_host);
    TensorDf4 output_dev;

    // start Reshape & doInfer

    Context<NV> ctx1(0, 1, 1);

    ActivationParam<TensorDf4> param(Active_elu, 0.1f, 0.1f);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Activation<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> act;
    act.compute_output_shape(input, output, param);
    output_dev.re_alloc(output[0]->shape());

    // init assume output tensor has been reshpaed by user.
    act.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    act(input, output, param, ctx1);

    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output[0]->record_event(cuda_stream);
    output_dev.sync();
    print_tensor_device(output_dev);
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
}

TEST(TestSaberFuncNV, test_func_sub_tensor) {

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    int img_num = 1;
    int in_channels = 2;
    int img_h = 8;
    int img_w = 8;

    Shape img_s(img_num, in_channels, img_h, img_w);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = (float)(0.05 * (i & 0x1f) * -1);
    }

    img_dev.copy_from(img_host);
    Shape img_s_t0(img_num, in_channels, 4, 4);

    TensorDf4 t0;
    TensorDf4 t1;

    t0.share_sub_buffer(img_dev, img_s_t0, {0, 0, 0, 0});
    t1.share_sub_buffer(img_dev, img_s_t0, {0, 0, 4, 4});

    print_tensor_shape("t0", t0);
    print_tensor_shape("t1", t1);

    TensorDf4 output_dev;

    TensorDf4 out0;
    TensorDf4 out1;

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);
    Context<NV> ctx2(0, 2, 2);

    ActivationParam<TensorDf4> param1(Active_elu, 0.1f, 0.1f);
    ActivationParam<TensorDf4> param2(Active_elu, 0.1f, 0.1f);

    std::vector<TensorDf4*> input1, input2;
    std::vector<TensorDf4*> output1, output2;

    input1.push_back(&t0);
    input2.push_back(&t1);

    output1.push_back(&out0);
    output2.push_back(&out1);

    //FIXME where do I get img_s and all those shapes ????
    output_dev.re_alloc(img_s);

    out0.share_sub_buffer(output_dev, img_s_t0, {0, 0, 0, 0});
    out1.share_sub_buffer(output_dev, img_s_t0, {0, 0, 4, 4});

    print_tensor_shape("output_dev", output_dev);

    Activation<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> act1;
    Activation<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> act2;

    act1.compute_output_shape(output1, input1, param1);
    act2.compute_output_shape(output2, input2, param2);

    print_tensor_shape("out0", out0);
    print_tensor_shape("out1", out1);

    // init assume output tensor has been reshpaed by user.
    act1.init(input1, output1, param1, SPECIFY, SABER_IMPL, ctx1);
    act1(input1, output1, param1, ctx1);
    cudaStream_t cuda_stream = ctx1.get_compute_stream();
    output1[0]->record_event(cuda_stream);

    act2.init(input2, output2, param2, SPECIFY, SABER_IMPL, ctx2);
    act2(input2, output2, param2, ctx2);
    cudaStream_t cuda_stream2 = ctx2.get_compute_stream();
    output2[0]->record_event(cuda_stream2);

    out0.sync();
    out1.sync();
    print_tensor_device(output_dev);
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
}

int main(int argc, const char** argv) {
    Env<NV>::env_init();
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

