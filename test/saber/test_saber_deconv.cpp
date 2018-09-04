
#include "core/context.h"
#include "funcs/deconv.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "tensor_op.h"
#include "funcs/funcs_utils.h"
#include "saber_types.h"
#include <vector>
#include "debug.h"
using namespace anakin::saber;

#ifdef NVIDIA_GPU
typedef Tensor<NVHX86> TensorHf4;
typedef Tensor<NV> TensorDf4;


template <bool print_res, bool speed_check>
void test_deconv_results(std::vector<TensorDf4*>& inputs, std::vector<TensorDf4*>& outputs,
                         TensorDf4& weights, int kernel_size, int stride, int pad,
                         int in_channel, int out_channel, TensorDf4& bias,
                         anakin::saber::ImplEnum impl) {

    ConvParam<NV> conv_param(1, pad, pad,
                             stride, stride,
                             1, 1,
                             &weights, &bias);
    Deconv<NV, AK_FLOAT> deconv;
    deconv.compute_output_shape(inputs, outputs, conv_param);
    outputs[0]->re_alloc(outputs[0]->shape(),AK_FLOAT);
    Context<NV> ctx1(0, 1, 1);

    SABER_CHECK(deconv.init(inputs, outputs, conv_param, SPECIFY, impl, ctx1));

    deconv(inputs, outputs, conv_param, ctx1);
    outputs[0]->record_event(ctx1.get_compute_stream());
    outputs[0]->sync();


    if (print_res) {
        LOG(INFO) << "Print Res";
        print_tensor(*outputs[0]);
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



TEST(TestSaberFunc, test_func_self_deconv) {
    Env<NV>::env_init();
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

    Shape img_shape({img_n, in_channel, img_h, img_w});
    Shape weights_shape({in_channel, out_channel, kernel_size, kernel_size});

    img_host.re_alloc(img_shape,AK_FLOAT);
    img_dev.re_alloc(img_shape,AK_FLOAT);

    fill_tensor_rand(img_host, -2.f, 2.f);
    img_dev.copy_from(img_host);
    inputs.push_back(&img_dev);

    weights_origin_host.re_alloc(weights_shape,AK_FLOAT);
    weights_cudnn.re_alloc(weights_shape,AK_FLOAT);

    weights_transform_host.re_alloc(weights_shape,AK_FLOAT);
    weights_transform_dev.re_alloc(weights_shape,AK_FLOAT);


    fill_tensor_rand(weights_origin_host, -1.f, 1.f);
    weights_cudnn.copy_from(weights_origin_host);
    Shape cudnn_weights_shape({out_channel, in_channel, kernel_size, kernel_size});
    // caffe transform weights into this kind of form !!!
    weights_cudnn.reshape(cudnn_weights_shape);
    TensorDf4 bias;
    Shape bias_shape({1, out_channel, 1, 1});

    if (bias_term) {
        bias.re_alloc(bias_shape,AK_FLOAT);
        fill_tensor_const(bias, 1);
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
//        write_tensorfile(t_cudnn_out,"t_cudnn_out");
//        write_tensorfile(t_saber_out,"t_saber_out");
        tensor_cmp_host(static_cast<float*>(t_cudnn_out.data()), static_cast<float*>(t_saber_out.data()),
                        t_cudnn_out.valid_size(), max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    }
}
#endif
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

