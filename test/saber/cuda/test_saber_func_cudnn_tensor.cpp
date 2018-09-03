#include "core/context.h"
#include "funcs/conv.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "funcs/impl/cuda/cudnn_helper.h"
#include "saber_types.h"
#include "saber/funcs/timer.h"
#include <vector>

using namespace anakin::saber;

template<typename inTensorH, typename inTensorD, typename outTensorH, typename outTensorD>
void test_cudnn_tensor() {

    Context<NV> ctx1(0, 1, 1);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;

    int out_channels = 3;
    int out_h = 1200;
    int out_w = 1200;

    int img_num = 1;
    int in_channels = 3;
    int img_h = 1200;
    int img_w = 1200;

    Shape img_s(img_num, in_channels, img_h, img_w);
    Shape out_s(img_num, out_channels, out_h, out_w);

    inTensorH img_host;
    inTensorD img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }

    img_dev.copy_from(img_host);
    //    print_tensor_device(img_dev, ctx1.get_compute_stream());
    img_dev.record_event(ctx1.get_compute_stream());

    img_dev.sync();

    outTensorH output_host;
    outTensorD output_dev;

    output_host.re_alloc(out_s);
    output_dev.re_alloc(out_s);

    LOG(INFO) << "input shape: " << img_num
              << ", " << in_channels
              << ", " << img_h
              << ", " << img_w;

    LOG(INFO) << "output shape: " << output_dev.num()
              << ", " << output_dev.channel()
              << ", " << output_dev.height()
              << ", " << output_dev.width();

    typedef typename inTensorD::Dtype dtype_in;
    typedef typename outTensorD::Dtype dtype_out;

    cudnnHandle_t _handle;
    cudnnTensorDescriptor_t _input_decs;
    cudnnTensorDescriptor_t _output_decs;

    cudaStream_t cuda_stream;
    cuda_stream = ctx1.get_compute_stream();

    CUDNN_CHECK(cudnnCreate(&_handle));
    CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));

    cudnnCreateTensorDescriptor(&_input_decs);
    cudnnCreateTensorDescriptor(&_output_decs);

    cudnnSetTensor4dDescriptor(_input_decs, CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT, img_num, in_channels, img_h, img_w);

    cudnnSetTensor4dDescriptor(_output_decs, CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT, img_num, out_channels, out_h, out_w);

    const void* x_data = img_dev.data();
    void* y_data = output_dev.mutable_data();

    const float alpha = 1.f;
    const int beta = 0.f;

    CUDNN_CHECK(cudnnTransformTensor(_handle, &alpha,
                                     _input_decs, x_data,
                                     &beta, _output_decs, y_data));

    SaberTimer<NV> t1;
    t1.start(ctx1);
    CUDNN_CHECK(cudnnTransformTensor(_handle, &alpha,
                                     _input_decs, x_data,
                                     &beta, _output_decs, y_data));
    output_dev.record_event(ctx1.get_compute_stream());
    output_dev.sync();
    t1.end(ctx1);
    LOG(INFO) << t1.get_average_ms();
    output_host.re_alloc(out_s);

    output_host.copy_from(output_dev);

    output_dev.sync();

    //    LOG(INFO)<<"=====================img========================";
    //    print_tensor_host(img_host);
    //
    //    LOG(INFO)<<"=====================out========================";
    //    print_tensor_host(output_host);

    CUDA_CHECK(cudaPeekAtLastError());

    CUDNN_CHECK(cudnnDestroy(_handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(_input_decs));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(_output_decs));
}

TEST(TestSaberFuncNV, test_func_cudnn_tensor_sh) {

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef Tensor<X86, AK_INT8, NCHW> TensorHINT8;
    typedef Tensor<NV, AK_INT8, NCHW> TensorDINT8;

    test_cudnn_tensor<TensorHf4, TensorDf4, TensorHf4, TensorDf4>();

    //    test_cudnn_tensor<TensorHINT8, TensorDINT8, TensorHINT8, TensorDINT8>();

    //    test_cudnn_tensor<TensorHINT8, TensorDINT8>();

}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env<NV>::env_init();

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

