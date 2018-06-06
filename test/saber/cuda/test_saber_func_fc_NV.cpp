
#include "core/context.h"
#include "funcs/fc.h"
#include "test_saber_func_fc_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;
typedef TargetWrapper<NV> API;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef TensorDf4::Dtype ftype;

void fc_compute(const TensorHf4& tin, const TensorHf4& weight, \
                const TensorHf4& bias, TensorHf4& tout) {

    int m = tin.num();
    int k = tin.valid_size() / m;
    int n = weight.valid_size() / k;
    bool bias_term = bias.valid_size() > 0;

    const float* din = tin.data();
    const float* w = weight.data();
    float* dout = tout.mutable_data();

    for (int i = 0; i < m; ++i) {
        float* pdout = dout + i * n;
        const float* pdin = din + i * k;

        for (int j = 0; j < n; ++j) {
            if (bias_term) {
                pdout[j] = bias.data()[j];
            } else {
                pdout[j] = 0;
            }

            for (int l = 0; l < k; ++l) {
                pdout[j] += pdin[l] * w[l * n + j];
            }
        }
    }
}

TEST(TestSaberFuncFcNV, test_func_fc_NV) {

    int test_iter = 100;
    int w_in = 7;
    int h_in = 7;
    int ch_in = 512;
    int num_in = 1;

    int num_out = 4096;
    int axis = 1;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out = {num_in, num_out, 1, 1};

    Shape sh_w{1, 1, w_in* h_in * ch_in, num_out};
    TensorDf4 weight(sh_w);
    Shape sh_b{1, 1, 1, num_out};
    TensorDf4 bias(sh_b);
    fill_tensor_device_const(weight, 1.f);
    fill_tensor_device_const(bias, 1.f);

    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    TensorDf4 tdin;
    TensorDf4 tdout;
    tdin.re_alloc(shape_in);
    fill_tensor_device_const(tdin, 1.f);
    input_dev_4d.push_back(&tdin);
    output_dev_4d.push_back(&tdout);

    // start Reshape & doInfer
    Context<NV> ctx_dev(0, 1, 1);

    FcParam<TensorDf4> param(&weight, &bias, num_out, axis);

    Fc<NV, AK_FLOAT> fc;

    LOG(INFO) << "shape out 4d: " << shape_out[0] << ", " << shape_out[1] << ", " << \
              shape_out[2] << ", " << shape_out[3];

    SABER_CHECK(fc.compute_output_shape(input_dev_4d, output_dev_4d, param));

    LOG(INFO) << "re-alloc tensor buffer";
    output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape());
    Shape va_sh = tdout.valid_shape();
    LOG(INFO) << "shape out 4d: " << va_sh[0] << ", " << va_sh[1] << ", " << \
              va_sh[2] << ", " << va_sh[3];
    CHECK_EQ(tdout.valid_shape() == shape_out, true) << "compute output shape error";

    LOG(INFO) << "FC initialization";
    SABER_CHECK(fc.init(input_dev_4d, output_dev_4d, param, SPECIFY, VENDER_IMPL, ctx_dev));

    LOG(INFO) << "FC compute";
    SaberTimer<NV> t1;
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        SABER_CHECK(fc(input_dev_4d, output_dev_4d, param, ctx_dev));
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
        //cudaDeviceSynchronize();
    }

    CUDA_POST_KERNEL_CHECK;
    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    LOG(INFO) << "total time: " << ts << "avg time: " << ts / test_iter;
    //print_tensor_device(*output_dev_4d[0]);
    //cudaDeviceSynchronize();

    //! check result
    TensorHf4 thin(shape_in);
    TensorHf4 thout(shape_out);
    TensorHf4 thw(sh_w);
    TensorHf4 thb(sh_b);
    thin.copy_from(tdin);
    thw.copy_from(weight);
    thb.copy_from(bias);
    fc_compute(thin, thw, thb, thout);
    //print_tensor_host(thout);

    TensorHf4 thout_d(shape_out);
    thout_d.copy_from(tdout);
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

