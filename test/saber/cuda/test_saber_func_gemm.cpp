
#include "core/context.h"
#include "funcs/conv.h"
#include "funcs/conv_act.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"
#include <vector>

//#include "cublas.h"

using namespace anakin::saber;

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

cublasHandle_t  cublas_handle;
void caffe_gemm(const int M, const int N, const int K,\
					 const float alpha, const float* A,\
					 const float* B, const float beta, float* C) {
    int lda = K;
    int ldb = N;
    cublasOperation_t cu_trans_a = CUBLAS_OP_N;
    cublasOperation_t cu_trans_b = CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemm(cublas_handle,
                             cu_trans_b, cu_trans_a,
                             N, M, K, &alpha,
                             B, ldb, A, lda,
                             &beta, C, N));
}

TEST(TestSaberFuncNV, test_gemm) {
    int img_num = 1;
    int kernel = 1;

//    int out_channels = 32;
//    int in_channels = 128;
//    int img_h = 52;
//    int img_w = 112;
//    int out_channels = 64;
//    int in_channels = 256;
//    int img_h = 26;
//    int img_w = 56;
    int out_channels = 128;
    int in_channels = 512;
    int img_h = 13;
    int img_w = 28;
//    int out_channels = 512;
//    int in_channels = 128;
//    int img_h = 13;
//    int img_w = 28;

    int pad = 0;
    int stride = 1;
    Context<NV> ctx1(0, 1, 1);

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, ctx1.get_compute_stream()));

    TensorDf4 weights;
    weights.re_alloc({out_channels, in_channels, 1, 1});

    TensorDf4 img;
    img.re_alloc({1, in_channels, img_h, img_w});

    TensorDf4 out;
    out.re_alloc({1, out_channels, img_h, img_w});
    TensorDf4 out_gemm;
    out_gemm.re_alloc({1, out_channels, img_h, img_w});

    fill_tensor_device_rand(weights, 1.f, 2.f);
    fill_tensor_device_rand(img, 1.f, 2.f);

    LOG(INFO) << "img_num: " << img_num;
    LOG(INFO) << "kernel: " << kernel;
    LOG(INFO) << "out_channels: " << out_channels;
    LOG(INFO) << "in_channels: " << in_channels;
    LOG(INFO) << "img_h: " << img_h;
    LOG(INFO) << "img_w: " << img_w;
    LOG(INFO) << "pad: " << pad;
    LOG(INFO) << "stride: " << stride;

    TensorDf4 bias;

    bias.re_alloc({1,out_channels, 1, 1});
    fill_tensor_device_const(bias, 0);
    std::vector<TensorDf4*> input_v;
    std::vector<TensorDf4*> output_gemm_v, output_v;
    float alpha = 1.0f;
    float beta = 0.f;
    cudaStream_t  cuda_stream  =ctx1.get_compute_stream();
    ker_gemm_32x32x32_NN_bias_relu(out_channels, img_h * img_w, in_channels,
                                        alpha,  weights.data(),
                                        beta,  img.data(),
                                        out_gemm.mutable_data(), bias.data(),
                                   cuda_stream);
    caffe_gemm(out_channels, img_h * img_w, in_channels,\
					 1.f, weights.data(),\
					 img.data(), 0.f, out.mutable_data());

    cudaDeviceSynchronize();
    SaberTimer<NV> t1;
    int ts = 100;

    for (int i = 0; i < ts; ++i) {
        t1.start(ctx1);
        caffe_gemm(out_channels, img_h * img_w, in_channels,\
					 1.f, weights.data(),\
					 img.data(), 0.f, out.mutable_data());
        ker_gemm_32x32x32_NN_bias_relu(out_channels, img_h * img_w, in_channels,
                                       alpha,  weights.data(),
                                       beta,  img.data(),
                                       out_gemm.mutable_data(), bias.data(),
                                       cuda_stream);
        out_gemm.record_event(ctx1.get_compute_stream());
        out_gemm.sync();
        t1.end(ctx1);
    }
    LOG(INFO) << "elapse time: " << t1.get_average_ms() << " ms";

    cudaDeviceSynchronize();

    TensorHf4 out_host;
    TensorHf4 out_gemm_host;
    out_host.re_alloc(out.shape());
    out_host.copy_from(out);

    out_gemm_host.re_alloc(out_gemm.shape());
    out_gemm_host.copy_from(out_gemm);
    double max_r, max_d;
    tensor_cmp_host(out_host.data(), out_gemm_host.data(), out_host.size(), max_r, max_d);
            LOG(INFO) << "cmp result: max_r = " << max_r << " max_d = " << max_d;
}

void test_conv_fp32_speed(std::vector<TensorDf4*> &inputs, std::vector<TensorDf4*> &outputs,
                          TensorDf4 &weights, int kernel_size, int stride, int pad,
                          int in_channel, int out_channel, TensorDf4 &bias,
                          anakin::saber::ImplEnum impl) {

    ConvParam<TensorDf4> conv_param(1, pad, pad,
                                    stride, stride,
                                    1, 1,
                                    &weights, &bias);
    ActivationParam<TensorDf4> act_param(Active_relu);
    ConvActiveParam<TensorDf4> param(conv_param, act_param);
    ConvAct<NV, AK_FLOAT> conv;
    conv.compute_output_shape(inputs, outputs, param);
    outputs[0]->re_alloc(outputs[0]->shape());
    Context<NV> ctx1(0, 1, 1);

    SABER_CHECK(conv.init(inputs, outputs, param, SPECIFY, impl, ctx1));

    conv(inputs, outputs, param, ctx1);
    outputs[0]->record_event(ctx1.get_compute_stream());
    outputs[0]->sync();

//    cudaDeviceSynchronize();
//
//    SaberTimer<NV> t1;
//    int ts = 100;
//    for (int i = 0; i < ts; ++i) {
//        t1.start(ctx1);
//        conv(inputs, outputs, param, ctx1);
//        outputs[0]->record_event(ctx1.get_compute_stream());
//        outputs[0]->sync();
//        t1.end(ctx1);
//    }
//            LOG(INFO)<<"elapse time: "<<t1.get_average_ms()<<" ms";

    cudaDeviceSynchronize();
}

TEST(TestSaberFuncNV, test_gemm_res) {
    int img_num = 1;
    int kernel = 1;

//    int out_channels = 32;
//    int in_channels = 128;
//    int img_h = 52;
//    int img_w = 112;
//    int out_channels = 64;
//    int in_channels = 256;
//    int img_h = 26;
//    int img_w = 56;
    int out_channels = 35;
    int in_channels = 37;
    int img_h = 9;
    int img_w = 7;
//    int out_channels = 512;
//    int in_channels = 128;
//    int img_h = 13;
//    int img_w = 28;

    int pad = 0;
    int stride = 1;
    Context<NV> ctx1(0, 1, 1);

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, ctx1.get_compute_stream()));

    TensorDf4 weights;
    weights.re_alloc({out_channels, in_channels, 1, 1});

    TensorDf4 img;
    img.re_alloc({1, in_channels, img_h, img_w});

    TensorDf4 out_cudnn;
    out_cudnn.re_alloc({1, out_channels, img_h, img_w});

    TensorDf4 out;
    out.re_alloc({1, out_channels, img_h, img_w});
    TensorDf4 out_gemm;
    out_gemm.re_alloc({1, out_channels, img_h, img_w});

    fill_tensor_device_rand(weights, 1.f, 2.f);
    fill_tensor_device_rand(img, 1.f, 2.f);

    LOG(INFO) << "img_num: " << img_num;
    LOG(INFO) << "kernel: " << kernel;
    LOG(INFO) << "out_channels: " << out_channels;
    LOG(INFO) << "in_channels: " << in_channels;
    LOG(INFO) << "img_h: " << img_h;
    LOG(INFO) << "img_w: " << img_w;
    LOG(INFO) << "pad: " << pad;
    LOG(INFO) << "stride: " << stride;

    TensorDf4 bias;

    bias.re_alloc({1, out_channels, 1, 1});
//    fill_tensor_device_const(bias, 0.0f);
    fill_tensor_device_rand(bias, 1.f, 2.f);
    std::vector<TensorDf4*> input_v;
    std::vector<TensorDf4*> output_v;
    std::vector<TensorDf4*> output_gemm_v;

    input_v.push_back(&img);
    output_v.push_back(&out_cudnn);
    output_gemm_v.push_back(&out_gemm);

    cudaStream_t  cuda_stream = ctx1.get_compute_stream();

    caffe_gemm(out_channels, img_h * img_w, in_channels,\
					 1.f, weights.data(),\
					 img.data(), 0.f, out.mutable_data());

    test_conv_fp32_speed(input_v, output_v,
                         weights, kernel, stride, pad,
                         in_channels, out_channels, bias,
                         VENDER_IMPL);

    test_conv_fp32_speed(input_v, output_gemm_v,
                         weights, kernel, stride, pad,
                         in_channels, out_channels, bias,
                         SABER_IMPL);

    cudaDeviceSynchronize();
    SaberTimer<NV> t1;
    int ts = 100;

    for (int i = 0; i < ts; ++i) {
        t1.start(ctx1);

        test_conv_fp32_speed(input_v, output_gemm_v,
                             weights, kernel, stride, pad,
                             in_channels, out_channels, bias,
                             SABER_IMPL);

        caffe_gemm(out_channels, img_h * img_w, in_channels,\
					 1.f, weights.data(),\
					 img.data(), 0.f, out.mutable_data());

        test_conv_fp32_speed(input_v, output_v,
                             weights, kernel, stride, pad,
                             in_channels, out_channels, bias,
                             VENDER_IMPL);

        out_gemm.record_event(ctx1.get_compute_stream());
        out_gemm.sync();
        t1.end(ctx1);
    }

    cudaDeviceSynchronize();

    TensorHf4 out_host;
    TensorHf4 out_gemm_host;
    TensorHf4 out_cudnn_host;
    out_host.re_alloc(out.shape());
    out_host.copy_from(out);

    out_gemm_host.re_alloc(out_gemm.shape());
    out_gemm_host.copy_from(out_gemm);

    out_cudnn_host.re_alloc(out_cudnn.shape());
    out_cudnn_host.copy_from(out_cudnn);
    double max_r0;
    double max_d0;
    double max_r1;
    double max_d1;
    double max_r2;
    double max_d2;

    tensor_cmp_host(out_host.data(),
                    out_cudnn_host.data(), out_host.size(), max_r0, max_d0);
    LOG(INFO) << "cmp cublas cudnn result: max_r0 = "
              << max_r0 << " max_d0 = " << max_d0;

    tensor_cmp_host(out_host.data(),
                    out_gemm_host.data(), out_host.size(), max_r1, max_d1);
    LOG(INFO) << "cmp cublas gemm result: max_r1 = "
              << max_r1 << " max_d1 = " << max_d1;

    tensor_cmp_host(out_cudnn_host.data(),
                    out_gemm_host.data(), out_gemm_host.size(), max_r2, max_d2);
    LOG(INFO) << "cmp cudnn gemm result: max_r2 = "
              << max_r2 << " max_d2 = " << max_d2;
//    print_tensor_host(out_host);
//    print_tensor_host(out_cudnn_host);
//    print_tensor_host(out_gemm_host);
}

int main(int argc, const char** argv){
    anakin::saber::Env<NV>::env_init();

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

