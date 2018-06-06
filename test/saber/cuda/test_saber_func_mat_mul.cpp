#include "core/context.h"
#include "funcs/mat_mul.h"
#include "test_saber_func_fc_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;
typedef TargetWrapper<NV> API;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef TensorDf4::Dtype ftype;

void batch_gemm_compute(const TensorHf4& tin, const TensorHf4& weight, TensorHf4& output) {
    int M = tin.height();
    int K = weight.height();
    int N = weight.width();
    int B = tin.num() * tin.channel();
    float* out_ptr = output.mutable_data();
    float* in_ptr = tin.mutable_data();
    float* wei_ptr = weight.mutable_data();
    for (int b = 0; b < B; b++)
    {
        float* optr = out_ptr + b * M * N;
        float* iptr = in_ptr + b * M * K;
        float* wptr = wei_ptr + b * K * N;
        for (int i = 0; i < M; ++i) {
            float* pdout = optr + i * N;
            const float* pdin = iptr + i * K;

            for (int j = 0; j < N; ++j) {

                for (int l = 0; l < K; ++l) {
                    pdout[j] += pdin[l] * wptr[l * N + j];
                }
            }
        }

    }
}

void batch_transpos(const TensorHf4& input, TensorHf4& output)
{
    int M = input.height();
    int N = input.width();
    int B = input.num() * input.channel();
    float* data_in = input.data();
    float* data_out = output.mutable_data();
    for (int b = 0; b < B; b++)
    {
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                data_out[b*M*N + n*M + m] = data_in[b*M*N + m*N+n];
            }
        }
    }
}

void swap_NM(int& M, int& N)
{
    int tmp = M;
    M = N;
    N = tmp;
}

TEST(TestSaberFuncFcNV, test_func_fc_NV) {

    int test_iter = 100;
    int N0 = 1;
    int C0 = 2;
    int H0 = 1024;
    int W0 = 1024;

    int N1 = N0;
    int C1 = C0;
    int H1 = W0;
    int W1 = 1024;

    bool T0 = false;
    bool T1 = false;

    if (T0) swap_NM(H0, W0);
    if (T1) swap_NM(H1, W1);

    Shape shape_out_should;
    {
        int N = std::max(N0, N1);
        int C = std::max(C0, C1);
        int H = T0 ? W0 : H0;
        int W = T1 ? W1 : H1;
        std::cout << " Shape out should be " << N << " " << C << " " << H << " " << W << std::endl;
    }

    Shape shape_in0(N0, C0, H0, W0);
    Shape shape_in1(N1, C1, H1, W1);

    TensorDf4 in0_dev(shape_in0);
    TensorHf4 in0_host(shape_in0);
    TensorDf4 in1_dev(shape_in1);
    TensorHf4 in1_host(shape_in1);

    TensorDf4 out0;

    fill_tensor_host_const(in0_host, 1.f);
    fill_tensor_host_const(in1_host, 1.f);

    in0_dev.copy_from(in0_host);
    in1_dev.copy_from(in1_host);

    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;

    inputs.push_back(&in0_dev);
    inputs.push_back(&in1_dev);

    outputs.push_back(&out0);

    // start Reshape & doInfer
    Context<NV> ctx_dev(0, 1, 1);

    MatMulParam<TensorDf4> param(T0, T1);

    MatMul<NV, AK_FLOAT> matmul;

    SABER_CHECK(matmul.compute_output_shape(inputs, outputs, param));
    outputs[0]->re_alloc(outputs[0]->valid_shape());
    Shape shape_out = outputs[0]->valid_shape();
    LOG(INFO) << "Compute output shape = " << shape_out[0] << " " << shape_out[1] << " " << shape_out[2] << " " << shape_out[3];
    SABER_CHECK(matmul.init(inputs, outputs, param, SPECIFY, SABER_IMPL, ctx_dev));

    SaberTimer<NV> t1;
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        SABER_CHECK(matmul(inputs, outputs, param, ctx_dev));
        outputs[0]->record_event(ctx_dev.get_compute_stream());
        outputs[0]->sync();
    }

    CUDA_POST_KERNEL_CHECK;
    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    LOG(INFO) << "total time: " << ts << " avg time: " << ts / test_iter << " avg GFLOPS: " << 2*N0*C0*((H0*W0*W1)/(ts/test_iter))*1e-6;
    //print_tensor_device(*outputs[0]);

    TensorHf4 thout(shape_out);
    fill_tensor_host_const(thout, 0.f);

    TensorHf4 cpu_in0(shape_in0);
    TensorHf4 cpu_in1(shape_in1);
    if (T0)
    {
       cpu_in0.re_alloc({shape_in0[0],shape_in0[1],shape_in0[3],shape_in0[2]});
       batch_transpos(in0_host, cpu_in0);
    }else{
        cpu_in0.copy_from(in0_host);
    }
    if (T1)
    {
       cpu_in1.re_alloc({shape_in1[0],shape_in1[1],shape_in1[3],shape_in1[2]}); 
       batch_transpos(in1_host, cpu_in1);
    }else{
        cpu_in1.copy_from(in1_host);
    }
    batch_gemm_compute(cpu_in0, cpu_in1, thout);
    //print_tensor_host(thout);

    TensorHf4 thout_d(shape_out);
    thout_d.copy_from(*outputs[0]);
    double max_ratio = 0;
    double max_diff = 0;
    tensor_cmp_host(thout.data(), thout_d.data(), thout.valid_size(), max_ratio, max_diff);

    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    CHECK_LE(fabs(max_ratio), 1.0e-6) << "error result";

}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env<NV>::env_init();
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

