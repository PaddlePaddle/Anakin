
#include "core/context.h"
#include "funcs/deconv_act.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;
typedef TargetWrapper<X86> X86_API;
typedef TargetWrapper<NV> NV_API;
typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;


void test_deconv(std::vector<TensorDf4*>& tin, \
                 int ch_out, int kernel, int stride, int pad, int dila, int group, bool bias) {

    int test_iter = 100;
    double to = 0;
    SaberTimer<NV> t1;

    Context<NV> ctx1(0, 1, 1);

    TensorDf4 tdout_cudnn;
    TensorDf4 tdout_saber;

    std::vector<TensorDf4*> tvout_cudnn;
    std::vector<TensorDf4*> tvout_saber;

    tvout_cudnn.push_back(&tdout_cudnn);
    tvout_saber.push_back(&tdout_saber);

    int num = tin[0]->num();
    int chin = tin[0]->channel();
    int hin = tin[0]->height();
    int win = tin[0]->width();

    LOG(INFO) << "conv param: ";
    LOG(INFO) << " img_num = " << num;
    LOG(INFO) << " in_channels = " << chin;
    LOG(INFO) << " img_h = " << hin;
    LOG(INFO) << " img_w = " << win;
    LOG(INFO) << " group = " << group;
    LOG(INFO) << " pad = " << pad;
    LOG(INFO) << " stride = " << stride;
    LOG(INFO) << " dilation = " << dila;
    LOG(INFO) << " kernel = " << kernel;
    LOG(INFO) << " out_channels = " << ch_out;

    int kernel_extent_h = dila * (kernel - 1) + 1;
    int hout = (hin - 1) * stride + kernel_extent_h - 2 * pad;
    int kernel_extent_w = dila * (kernel - 1) + 1;
    int wout = (win - 1) * stride + kernel_extent_w - 2 * pad;

    Shape shape_out{num, ch_out, hout, wout};

    Shape shw{ch_out, chin / group, kernel, kernel};
    Shape shb{1, ch_out, 1, 1};
    TensorDf4 pweiht(shw);
    TensorDf4 pbias(shb);
    fill_tensor_device_const(pweiht, 1.f);
    fill_tensor_device_const(pbias, 1.f);

    TensorDf4* bias_ptr = nullptr;

    if (bias) {
        bias_ptr = &pbias;
    }

    ConvParam<TensorDf4> conv_param(group, pad, pad,
                                    stride, stride,
                                    dila, dila,
                                    &pweiht, bias_ptr);
    ActivationParam<TensorDf4> active_param(Active_relu);
    ConvActiveParam<TensorDf4> param(conv_param, active_param);

    DeconvAct<NV, AK_FLOAT> deconv_cudnn;
    DeconvAct<NV, AK_FLOAT> deconv_saber;

    deconv_cudnn.compute_output_shape(tin, tvout_cudnn, param);
    deconv_saber.compute_output_shape(tin, tvout_saber, param);
    Shape sh_out_cudnn = tvout_cudnn[0]->valid_shape();
    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
              << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_cudnn, true) << "compute output shape error";
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_cudnn[0]->re_alloc(shape_out);
    tvout_saber[0]->re_alloc(shape_out);

    //! init
    LOG(INFO) << "cudnn impl init";
    SABER_CHECK(deconv_cudnn.init(tin, tvout_cudnn, param, SPECIFY, VENDER_IMPL, ctx1));
    LOG(INFO) << "saber impl init";
    SABER_CHECK(deconv_saber.init(tin, tvout_saber, param, SPECIFY, SABER_IMPL, ctx1));

    //! compute
    LOG(INFO) << "cudnn impl compute";
    to = 0;
    t1.start(ctx1);

    for (int i = 0; i < test_iter; ++i) {
        deconv_cudnn(tin, tvout_cudnn, param, ctx1);
        tvout_cudnn[0]->record_event(ctx1.get_compute_stream());
        tvout_cudnn[0]->sync();
        //to += t1.get_average_ms();
    }

    t1.end(ctx1);
    LOG(INFO) << "cudnn deconv running time: " << t1.get_average_ms() / test_iter;
    //print_tensor_device(*tvout_cudnn[0]);

    LOG(INFO) << "saber impl compute";
    to = 0;
    t1.clear();
    t1.start(ctx1);

    for (int i = 0; i < test_iter; ++i) {
        deconv_saber(tin, tvout_saber, param, ctx1);
        tvout_saber[0]->record_event(ctx1.get_compute_stream());
        tvout_saber[0]->sync();
        //to += t1.get_average_ms();
    }

    t1.end(ctx1);
    LOG(INFO) << "saber deconv running time: " << t1.get_average_ms() / test_iter;
    //print_tensor_device(*tvout_saber[0]);

    double max_ratio = 0;
    double max_diff = 0;
    TensorHf4 tout1(shape_out);
    TensorHf4 tout2(shape_out);
    tout1.copy_from(*tvout_cudnn[0]);
    tout2.copy_from(*tvout_saber[0]);
    tensor_cmp_host(tout1.data(), tout2.data(), tout1.valid_size(), max_ratio, max_diff);
    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;

}

TEST(TestSaberFuncNV, test_func_deconv_act) {

    int num = 1;
    int chin = 64;
    int hin = 112;
    int win = 112;

    int group = chin;
    int pad = 1;
    int stride = 2;
    int dilation = 1;
    int kernel = 3;
    int chout = chin;

    bool bias_term = true;

    Shape shape_in(num, chin, hin, win);

    TensorHf4 th0;
    TensorDf4 tdin;

    th0.re_alloc(shape_in);
    tdin.re_alloc(shape_in);

    for (int i = 0; i < th0.size(); ++i) {
        th0.mutable_data()[i] = 0x7f & i;
    }

    tdin.copy_from(th0);

    std::vector<TensorDf4*> tin;
    tin.push_back(&tdin);

    test_deconv(tin, chout, kernel, stride, pad, dilation, group, bias_term);
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env<NV>::env_init();
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

