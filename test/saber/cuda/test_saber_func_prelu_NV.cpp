
#include "core/context.h"
#include "funcs/prelu.h"
#include "test_saber_func_prelu_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncPreluNV, test_func_prelu_without_roi_NV) {

    typedef TargetWrapper<NV> API;

    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef TensorDf4::Dtype dtype;

    int test_iter = 1000;

    int w_in = 10;
    int h_in = 2;
    int ch_in = 2;
    int num_in = 1;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out = shape_in;
    Shape sh_slope{1, 1, 1, ch_in};
    Tensor<X86, AK_FLOAT, NCHW> th_slope(sh_slope);
    TensorDf4 tslop(sh_slope);

    for (int i = 0; i < ch_in; ++i) {
        th_slope.mutable_data()[i] = 0.1f * (i + 1);
    }

    tslop.copy_from(th_slope);

    PreluParam<TensorDf4> param_shared(true, &tslop);
    PreluParam<TensorDf4> param(false, &tslop);

    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d1, output_dev_4d2;

    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in);

    for (int i = 0; i < thin.size(); ++i) {
        thin.mutable_data()[i] = -i + thin.size() / 2 / ch_in;
    }

    TensorDf4 tdin, tdout1, tdout2;
    tdin.re_alloc(shape_in);
    tdin.copy_from(thin);
    input_dev_4d.push_back(&tdin);

    // start Reshape & doInfer
    Context<NV> ctx_dev(0, 1, 1);

    Prelu<NV, AK_FLOAT> prelu_dev1;
    Prelu<NV, AK_FLOAT> prelu_dev2;

    LOG(INFO) << "shape out 4d: " << shape_out[0] << ", " << shape_out[1] << ", " << \
              shape_out[2] << ", " << shape_out[3];

    output_dev_4d1.push_back(&tdout1);
    output_dev_4d2.push_back(&tdout2);
    SABER_CHECK(prelu_dev1.compute_output_shape(input_dev_4d, output_dev_4d1, param));
    SABER_CHECK(prelu_dev2.compute_output_shape(input_dev_4d, output_dev_4d2, param_shared));

    LOG(INFO) << "re-alloc tensor buffer";
    output_dev_4d1[0]->re_alloc(output_dev_4d1[0]->valid_shape());
    output_dev_4d2[0]->re_alloc(output_dev_4d2[0]->valid_shape());
    Shape va_sh = tdout1.valid_shape();
    LOG(INFO) << "shape out 4d: " << va_sh[0] << ", " << va_sh[1] << ", " << \
              va_sh[2] << ", " << va_sh[3];
    va_sh = tdout2.valid_shape();
    LOG(INFO) << "shape out 4d: " << va_sh[0] << ", " << va_sh[1] << ", " << \
              va_sh[2] << ", " << va_sh[3];
    CHECK_EQ(tdout1.valid_shape() == shape_out, true) << "compute output shape error";
    CHECK_EQ(tdout2.valid_shape() == shape_out, true) << "compute output shape error";

    LOG(INFO) << "prelu initialization";
    SABER_CHECK(prelu_dev1.init(input_dev_4d, output_dev_4d1, param, RUNTIME, SABER_IMPL, ctx_dev));
    SABER_CHECK(prelu_dev2.init(input_dev_4d, output_dev_4d2, param_shared, RUNTIME, SABER_IMPL,
                                ctx_dev));

    LOG(INFO) << "prelu compute";
    SaberTimer<NV> t1;
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        SABER_CHECK(prelu_dev1(input_dev_4d, output_dev_4d1, param, ctx_dev));
        output_dev_4d1[0]->record_event(ctx_dev.get_compute_stream());
        SABER_CHECK(prelu_dev2(input_dev_4d, output_dev_4d2, param_shared, ctx_dev));
        output_dev_4d2[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d1[0]->sync();
        output_dev_4d2[0]->sync();
    }

    CUDA_POST_KERNEL_CHECK;
    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    LOG(INFO) << "total time: " << ts << "avg time: " << ts / test_iter;
    print_tensor_device(*output_dev_4d1[0]);
    print_tensor_device(*output_dev_4d2[0]);
    cudaDeviceSynchronize();
}

TEST(TestSaberFuncPreluNV, test_func_prelu_ROI_NV) {

    typedef TargetWrapper<NV> API;

    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef TensorDf4::Dtype dtype;

    int test_iter = 1000;

    int w_in = 10;
    int h_in = 4;
    int ch_in = 2;
    int num_in = 1;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_in_roi{num_in, ch_in, h_in / 2, w_in / 2};
    Shape off1{0, 0, 0, 0};
    Shape off2{0, 0, 2, 5};
    Shape shape_out = shape_in_roi;

    Shape sh_slope{1, 1, 1, ch_in};
    Tensor<X86, AK_FLOAT, NCHW> th_slope(sh_slope);
    TensorDf4 tslop(sh_slope);

    for (int i = 0; i < ch_in; ++i) {
        th_slope.mutable_data()[i] = 0.1f * (i + 1);
    }

    tslop.copy_from(th_slope);

    PreluParam<TensorDf4> param_shared(true, &tslop);
    PreluParam<TensorDf4> param(false, &tslop);

    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;

    std::vector<TensorDf4*> in_4d1, in_4d2;
    std::vector<TensorDf4*> out_4d1, out_4d2;

    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in);

    for (int i = 0; i < thin.size(); ++i) {
        thin.mutable_data()[i] = -i + thin.size() / 2 / ch_in;
    }

    TensorDf4 tdin, tdin_roi1, tdin_roi2, tdout, tdout_roi1, tdout_roi2;
    tdin.re_alloc(shape_in);
    tdout.re_alloc(shape_in);
    tdin.copy_from(thin);
    tdin_roi1.share_sub_buffer(tdin, shape_in_roi, off1);
    tdin_roi2.share_sub_buffer(tdin, shape_in_roi, off2);
    in_4d1.push_back(&tdin_roi1);
    in_4d2.push_back(&tdin_roi2);
    out_4d1.push_back(&tdout_roi1);
    out_4d2.push_back(&tdout_roi2);

    // start Reshape & doInfer
    Context<NV> ctx_dev(0, 1, 1);

    Prelu<NV, AK_FLOAT> prelu_dev1;
    Prelu<NV, AK_FLOAT> prelu_dev2;

    LOG(INFO) << "shape out 4d: " << shape_out[0] << ", " << shape_out[1] << ", " << \
              shape_out[2] << ", " << shape_out[3];

    prelu_dev1.compute_output_shape(in_4d1, out_4d1, param);
    prelu_dev2.compute_output_shape(in_4d2, out_4d2, param_shared);

    LOG(INFO) << "re-alloc tensor buffer";
    out_4d1[0]->share_sub_buffer(tdout, shape_in_roi, off1);
    out_4d2[0]->share_sub_buffer(tdout, shape_in_roi, off2);

    CHECK_EQ(out_4d1[0]->valid_shape() == shape_out, true) << "compute shape error";

    LOG(INFO) << "prelu initialization";
    prelu_dev1.init(in_4d1, out_4d1, param, SPECIFY, SABER_IMPL, ctx_dev);
    prelu_dev2.init(in_4d2, out_4d2, param_shared, SPECIFY, SABER_IMPL, ctx_dev);

    LOG(INFO) << "prelu compute";
    SaberTimer<NV> t1;
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        prelu_dev1(in_4d1, out_4d1, param, ctx_dev);
        out_4d1[0]->record_event(ctx_dev.get_compute_stream());
        prelu_dev2(in_4d2, out_4d2, param_shared, ctx_dev);
        out_4d2[0]->record_event(ctx_dev.get_compute_stream());
        out_4d1[0]->sync();
        out_4d2[0]->sync();
    }

    CUDA_POST_KERNEL_CHECK;
    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    printf("total time : %.4f, avg time : %.4f\n", ts, ts / test_iter);
    print_tensor_device(tdout);
    cudaDeviceSynchronize();
    TensorDf4 troi(out_4d1[0]->valid_shape());
    troi.copy_from(*out_4d1[0]);
    print_tensor_device(troi);
    cudaDeviceSynchronize();
    troi.copy_from(*out_4d2[0]);
    print_tensor_device(troi);
    CUDA_POST_KERNEL_CHECK;
    cudaDeviceSynchronize();
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env<NV>::env_init();
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

