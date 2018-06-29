#include "core/context.h"
#include "saber/funcs/layer_norm.h"
#include "test/saber/cuda/test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

template <typename dtype>
void layernorm_cpu(Shape shin, const dtype* scale, const bool flag_scale, \
    const dtype* bias, const bool flag_bias, int axis, float eps, \
    const dtype* src, dtype* dst) {

    int inner_size = shin.count(axis);
    int outer_size = shin.count() / inner_size;

    for (int i = 0; i < outer_size; ++i) {
        dtype mean = 0;
        dtype std = 0;
        const dtype* src_ptr = src + i * inner_size;
        dtype* dst_ptr = dst + i * inner_size;
        for (int j = 0; j < inner_size; ++j) {
            mean += src_ptr[j];
        }
        mean /= inner_size;
        for (int j = 0; j < inner_size; ++j) {
            std += (src_ptr[j] - mean) * (src_ptr[j] - mean);
        }
        std = std / inner_size;
        //printf("std pre: %.6f\n", std);
        std = 1.f / (sqrtf(std) + eps);
        //printf("mean: %.6f, std: %.6f\n", mean, std);
        for (int j = 0; j < inner_size; ++j) {
            dst_ptr[j] = (flag_scale? scale[j] : 1) * (src_ptr[j] - mean) * std + (flag_bias? bias[j] : 0);
        }
    }
}


TEST(TestSaberFuncNV, test_func_normalize_without_roi_NV) {


    typedef TargetWrapper<NV> API;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    typedef TensorDf4::Dtype dtype;

    int test_iter = 1;
    float eps = 1e-6f;
    int axis = 2;

    int w_in = 128;
    int h_in = 128;
    int ch_in = 64;
    int num_in = 4;

    Shape shape_in{num_in, ch_in, h_in, w_in};
    Shape shape_out = shape_in;
    int inner_size = shape_in.count(axis);
    int outer_size = shape_in.count() / inner_size;

    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;

    Tensor<X86, AK_FLOAT, NCHW> thsca, thbias;
    TensorDf4 tdsca, tdbias;

    thsca.re_alloc(Shape(1, 1, 1, inner_size));
    thbias.re_alloc(Shape(1, 1, 1, inner_size));
    tdsca.re_alloc(Shape(1, 1, 1, inner_size));
    tdbias.re_alloc(Shape(1, 1, 1, inner_size));

    //fill_tensor_host_rand(thsca, -1.f, 1.f);
    //fill_tensor_host_rand(thbias, -1.f, 1.f);
    fill_tensor_host_const(thsca, 1.f);
    fill_tensor_host_const(thbias, 1.f);

    tdsca.copy_from(thsca);
    tdbias.copy_from(thbias);

    LayerNormParam<TensorDf4> param(axis, eps, &tdsca, &tdbias);

    LOG(INFO) << "create layer norm param";

    //! create input output tensor
    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in), thout(shape_in);

    fill_tensor_host_rand(thin, -1.f, 1.f);
    //fill_tensor_host_const(thin, 1.f);

    TensorDf4 tdin, tdout;
    tdin.re_alloc(shape_in);
    SABER_CHECK(tdin.copy_from(thin));
    CUDA_POST_KERNEL_CHECK;
    input_dev_4d.push_back(&tdin);
    output_dev_4d.push_back(&tdout);
    //print_tensor_device(tdin);
    //cudaDeviceSynchronize();
    //CUDA_POST_KERNEL_CHECK;
    //! create process contex
    Context<NV> ctx_dev(0, 1, 1);

    //! create normalize class
    LayerNorm<NV, AK_FLOAT> norm;
    LOG(INFO) << "layernorm compute ouput shape";
    SABER_CHECK(norm.compute_output_shape(input_dev_4d, output_dev_4d, param));
    //LOG(INFO) << "re-alloc tensor buffer";
    output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape());
    Shape va_sh = tdout.valid_shape();
    LOG(INFO) << "shape out 4d: " << va_sh[0] << ", " << va_sh[1] << ", " << \
              va_sh[2] << ", " << va_sh[3];
    CHECK_EQ(va_sh == shape_out, true) << "compute output shape error";

    LOG(INFO) << "layernorm initialization";
    SABER_CHECK(norm.init(input_dev_4d, output_dev_4d, param, SPECIFY, SABER_IMPL, ctx_dev));

    LOG(INFO) << "layernorm compute";
    //! compute result by cpu funcs
    layernorm_cpu(shape_in, thsca.data(), true, thbias.data(), true, axis, eps, thin.data(), thout.mutable_data());
    //print_tensor_host(thout);
    SaberTimer<NV> t1;
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        SABER_CHECK(norm(input_dev_4d, output_dev_4d, param, ctx_dev));
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
    }

    CUDA_POST_KERNEL_CHECK;
    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    LOG(INFO) << "total time: " << ts << ", avg time: " << ts / test_iter;
    //print_tensor_device(*output_dev_4d[0]);
    //cudaDeviceSynchronize();

    Tensor<X86, AK_FLOAT, NCHW> th_result(shape_in);
    th_result.copy_from(tdout);
    double max_ratio = 0;
    double max_diff = 0;
    tensor_cmp_host(thout.data(), th_result.data(), thout.valid_size(), max_ratio, max_diff);
    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    //CHECK_LE(fabs(max_ratio), 1.0e-6) << "error result";
    LOG(INFO) << "\n";
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env<NV>::env_init();
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

