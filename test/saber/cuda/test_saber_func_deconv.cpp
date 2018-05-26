
#include "core/context.h"
#include "funcs/deconv.h"
#include "funcs/deconv_act.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "funcs/funcs_utils.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;
typedef TargetWrapper<X86> X86_API;
typedef TargetWrapper<NV> NV_API;
typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

#if 0
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

    Deconv<NV, AK_FLOAT> deconv_cudnn;
    Deconv<NV, AK_FLOAT> deconv_saber;

    deconv_cudnn.compute_output_shape(tvout_cudnn, tin, conv_param);
    deconv_saber.compute_output_shape(tvout_saber, tin, conv_param);
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
    SABER_CHECK(deconv_cudnn.init(tin, tvout_cudnn, conv_param, SPECIFY, VENDER_IMPL, ctx1));
    LOG(INFO) << "saber impl init";
    SABER_CHECK(deconv_saber.init(tin, tvout_saber, conv_param, SPECIFY, SABER_IMPL, ctx1));

    //! compute
    LOG(INFO) << "cudnn impl compute";
    to = 0;
    t1.start(ctx1);

    for (int i = 0; i < test_iter; ++i) {
        deconv_cudnn(tin, tvout_cudnn, conv_param, ctx1);
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
        deconv_saber(tin, tvout_saber, conv_param, ctx1);
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
    int chin = 48;
    int hin = 320;
    int win = 320;

    int group = chin;
    int pad = 1;
    int stride = 2;
    int dilation = 1;
    int kernel = 4;
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
#endif

template <bool print_res, bool speed_check>
void test_deconv_results(std::vector<TensorDf4*>& inputs, std::vector<TensorDf4*>& outputs,
                         TensorDf4& weights, int kernel_size, int stride, int pad,
                         int in_channel, int out_channel, TensorDf4& bias,
                         anakin::saber::ImplEnum impl) {

    ConvParam<TensorDf4> conv_param(1, pad, pad,
                                    stride, stride,
                                    1, 1,
                                    &weights, &bias);
    Deconv<NV, AK_FLOAT> deconv;
    deconv.compute_output_shape(inputs, outputs, conv_param);
    outputs[0]->re_alloc(outputs[0]->shape());
    Context<NV> ctx1(0, 1, 1);

    SABER_CHECK(deconv.init(inputs, outputs, conv_param, SPECIFY, impl, ctx1));

    deconv(inputs, outputs, conv_param, ctx1);
    outputs[0]->record_event(ctx1.get_compute_stream());
    outputs[0]->sync();

    if (print_res) {
        LOG(INFO) << "Print Res";
        print_tensor_device(*outputs[0]);
    }

    cudaDeviceSynchronize();

    if (speed_check) {
        SaberTimer<NV> t1;
        int ts = 20;

        for (int i = 0; i < ts; ++i) {
            t1.start(ctx1);
            deconv(inputs, outputs, conv_param, ctx1);
            outputs[0]->record_event(ctx1.get_compute_stream());
            outputs[0]->sync();
            t1.end(ctx1);
        }

        LOG(INFO) << "elapse time: " << t1.get_average_ms() << " ms";
    }

    cudaDeviceSynchronize();
}

template <bool print_res, bool speed_check>
void test_deconv_act_results(std::vector<TensorDf4*>& inputs, std::vector<TensorDf4*>& outputs,
                             TensorDf4& weights, int kernel_size, int stride, int pad,
                             int in_channel, int out_channel, TensorDf4& bias,
                             anakin::saber::ImplEnum impl) {

    ConvParam<TensorDf4> conv_param(1, pad, pad,
                                    stride, stride,
                                    1, 1,
                                    &weights, &bias);
    ActivationParam<TensorDf4> act_param(Active_relu);
    ConvActiveParam<TensorDf4> param(conv_param, act_param);
    DeconvAct<NV, AK_FLOAT> deconv;
    deconv.compute_output_shape(inputs, outputs, param);
    outputs[0]->re_alloc(outputs[0]->shape());
    Context<NV> ctx1(0, 1, 1);

    SABER_CHECK(deconv.init(inputs, outputs, param, SPECIFY, impl, ctx1));

    deconv(inputs, outputs, param, ctx1);
    outputs[0]->record_event(ctx1.get_compute_stream());
    outputs[0]->sync();

    if (print_res) {
        LOG(INFO) << "Print Res";
        print_tensor_device(*outputs[0]);
    }

    cudaDeviceSynchronize();

    if (speed_check) {
        SaberTimer<NV> t1;
        int ts = 20;

        for (int i = 0; i < ts; ++i) {
            t1.start(ctx1);
            deconv(inputs, outputs, param, ctx1);
            outputs[0]->record_event(ctx1.get_compute_stream());
            outputs[0]->sync();
            t1.end(ctx1);
        }

        LOG(INFO) << "elapse time: " << t1.get_average_ms() << " ms";
    }

    cudaDeviceSynchronize();
}


TEST(TestSaberFuncNV, test_func_self_deconv) {
    int img_n = 8;
    int kernel_size = 4;
    int pad = 1;
    int stride = 2;
    int img_h = 320;
    int img_w = 320;
    int in_channel = 48;
    int out_channel = 12; // this actully the channel dim
    bool bias_term = true;
    const bool print_res = false;
    const bool speed_check = true;

    LOG(INFO) << " img_n: " << img_n;
    LOG(INFO) << " kernel_size: " << kernel_size;
    LOG(INFO) << " pad: " << pad;
    LOG(INFO) << " stride: " << stride;
    LOG(INFO) << " img_h: " << img_h;
    LOG(INFO) << " img_w: " << img_w;
    LOG(INFO) << " in_channel: " << in_channel;
    LOG(INFO) << " out_channel: " << out_channel;
    LOG(INFO) << " bias: " << (bias_term ? "true" : "false");
    TensorHf4 img_host;
    TensorDf4 img_dev;
    std::vector<TensorDf4*> inputs;

    TensorHf4 weights_origin_host;
    TensorDf4 weights_cudnn;

    TensorHf4 weights_transform_host;
    TensorDf4 weights_transform_dev;

    Shape img_shape(img_n, in_channel, img_h, img_w);
    Shape weights_shape(in_channel, out_channel, kernel_size, kernel_size);

    img_host.re_alloc(img_shape);
    img_dev.re_alloc(img_shape);

    fill_tensor_host_rand(img_host, -2.f, 2.f);
    img_dev.copy_from(img_host);
    inputs.push_back(&img_dev);

    weights_origin_host.re_alloc(weights_shape);
    weights_cudnn.re_alloc(weights_shape);

    weights_transform_host.re_alloc(weights_shape);
    weights_transform_dev.re_alloc(weights_shape);

    //    for (int i = 0; i < weights_origin_host.size(); ++i) {
    //        weights_origin_host.mutable_data()[i] = (float)i * 0.02;
    //    }
    fill_tensor_host_rand(weights_origin_host, -1.f, 1.f);
    weights_cudnn.copy_from(weights_origin_host);
    Shape cudnn_weights_shape{out_channel, in_channel, kernel_size, kernel_size};
    // caffe transform weights into this kind of form !!!
    weights_cudnn.reshape(cudnn_weights_shape);

    // transform
    // PAY ATTENTION!!!!
    // The shape of weights is suppose to be {in_channel, out_channel, kernel_size, kernel_size};
    // but caffe is reshaped their shape as {out, in, kernel_size, kernel_size}
    // so this need to reshaped like caffe as {out_channel, in_channel, kernel_size, kernel_size}
    // The param of transform_weights_deconv:
    // int in_channel  : the in_channel of the img(where loop running on!)
    //                   this param must be the seam with img in_channel,
    //
    // int out_channel : the real output filter num(as much as you can, this is the proto param)
    //
    // const float *
    //     weights_src : the real data is orgnized as
    //                   (in_channel, out_channel, kernel_size, kernel_size)
    // const float *
    //     XX_out      : the output data is orgnized as
    //                   (out_channel, in_channel, kernel_size, kernel_size)
    //                   just like normal convolution weights
    if (false) {
        int offset = weights_origin_host.size() / 4;
        float* trans_w = weights_transform_host.mutable_data();
        scale_weight_deconv_w4x4<4, true>(trans_w + 0 * offset,
                                          trans_w + 1 * offset,
                                          trans_w + 2 * offset,
                                          trans_w + 3 * offset,
                                          weights_origin_host.data(),
                                          in_channel, out_channel);

        weights_transform_dev.copy_from(weights_transform_host);
        // seam as caffe
        weights_transform_dev.reshape(cudnn_weights_shape);
    }

    //    weights_transform_dev.copy_from(weights_origin_host);
    // done transform

    //print_tensor_host(weights_origin_host);
    //    print_tensor_host(weights_transform_host);
    TensorDf4 bias;
    Shape bias_shape(1, out_channel, 1, 1);

    if (bias_term) {
        bias.re_alloc(bias_shape);
        fill_tensor_device_const(bias, 1);
    }

    TensorDf4 cudnn_out_tensor;
    TensorDf4 saber_out_tensor;
    std::vector<TensorDf4*> cudnn_outputs;
    std::vector<TensorDf4*> saber_outputs;
    cudnn_outputs.push_back(&cudnn_out_tensor);
    saber_outputs.push_back(&saber_out_tensor);

    LOG(INFO) << "start cudnn impl";
    test_deconv_results<print_res, speed_check>(inputs, cudnn_outputs, weights_cudnn,
            kernel_size, stride, pad, in_channel, out_channel, bias,
            anakin::saber::VENDER_IMPL);

    LOG(INFO) << "start saber impl";
    test_deconv_results<print_res, speed_check>(inputs, saber_outputs, weights_cudnn,
            kernel_size, stride, pad, in_channel, out_channel, bias,
            anakin::saber::SABER_IMPL);

    if (!print_res) {
        double max_ratio = 0;
        double max_diff = 0;
        TensorHf4 t_cudnn_out(cudnn_out_tensor.shape());
        TensorHf4 t_saber_out(saber_out_tensor.shape());
        t_cudnn_out.copy_from(cudnn_out_tensor);
        t_saber_out.copy_from(saber_out_tensor);
        tensor_cmp_host(t_cudnn_out.data(), t_saber_out.data(),
                        t_cudnn_out.valid_size(), max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    }
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env<NV>::env_init();
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

