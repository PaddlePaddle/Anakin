
#include "core/context.h"
#include "funcs/conv_act.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

int img_num = 1;
int chin = 1024;
int img_h = 7;
int img_w = 7;

int group = chin;
int pad_h = 1;
int pad_w = 1;
int stride_h = 1;
int stride_w = 1;
int dilation_h = 1;
int dilation_w = 1;

int kernel_h = 3;
int kernel_w = 3;
int chout = chin;


bool g_bias_term = true;

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

Shape shin(img_num, chin, img_h, img_w);
Shape shw(chout, chin / group, kernel_h, kernel_w);
Shape shb(1, chout, 1, 1);

TEST(TestSaberFuncNV, test_func_conv_relu_fusion) {

    int warm_up = 10;
    int iter = 100;

    SaberTimer<NV> t1;
    SaberTimer<NV> t2;

    TensorDf4 tdin(shin);
    TensorDf4 wdev(shw);
    TensorDf4 bdev(shb);
    TensorDf4 tdout;

    fill_tensor_device_rand(wdev, 0.f, 1.f);
    fill_tensor_device_rand(bdev, 0.f, 1.f);
    fill_tensor_device_rand(tdin, -128.f, 127.f);

    LOG(INFO) << "test depthwise conv + relu";

    Context<NV> ctx1(0, 1, 1);
    ConvParam<TensorDf4> conv_param(group, pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    &wdev, &bdev);

    ActivationParam<TensorDf4> active_param(Active_relu);

    ConvActiveParam<TensorDf4> param(conv_param, active_param);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&tdin);
    output.push_back(&tdout);

    ConvAct<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> conv_saber;
    ConvAct<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> conv_vender;
    conv_saber.compute_output_shape(input, output, param);

    output[0]->re_alloc(output[0]->valid_shape());

    LOG(INFO) << "regular start with group = " << group;
    // init assume output tensor has been reshpaed by user.
    conv_saber.init(input, output, param, SPECIFY, SABER_IMPL, ctx1);
    conv_vender.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);


    for (int i = 0; i < warm_up; ++i) {
        conv_saber(input, output, param, ctx1);
    }

    t1.start(ctx1);

    for (int i = 0; i < iter; ++i) {
        conv_saber(input, output, param, ctx1);
        output[0]->record_event(ctx1.get_compute_stream());
        output[0]->sync();
    }

    t1.end(ctx1);

    TensorHf4 th1(tdout.valid_shape());
    th1.copy_from(tdout);

    for (int i = 0; i < warm_up; ++i) {
        conv_vender(input, output, param, ctx1);
    }

    t2.start(ctx1);

    for (int i = 0; i < iter; ++i) {
        conv_vender(input, output, param, ctx1);
        output[0]->record_event(ctx1.get_compute_stream());
        output[0]->sync();
    }

    t2.end(ctx1);

    TensorHf4 th2(tdout.valid_shape());
    th2.copy_from(tdout);

    LOG(INFO) << "depthwise conv, saber impl: " << t1.get_average_ms() / iter << ", cudnn impl: " <<
              t2.get_average_ms() / iter;

    double max_err;
    double max_ratio;
    tensor_cmp_host(th1.data(), th2.data(), th1.valid_size(), max_ratio, max_err);
    LOG(INFO) << "max error: " << max_err << ", ratio: " << max_ratio;
    CHECK_EQ(max_ratio < 1e-6f, true) << "check result, error";

    CUDA_CHECK(cudaPeekAtLastError());
}

int main(int argc, const char** argv) {

    Env<NV>::env_init();

    LOG(INFO) << " conv param: ";
    LOG(INFO) << " img_num = " << img_num;
    LOG(INFO) << " g_in_channels = " << chin;
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
    LOG(INFO) << " g_out_channels = " << chout;

    LOG(INFO) << "tensor initialization finished";

    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

