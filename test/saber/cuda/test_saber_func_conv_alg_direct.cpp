#include "core/context.h"
#include "funcs/conv.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>
#include <tuple>

using namespace anakin::saber;

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

// Vector saves w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride
std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int>>
problem_set = {
    // dump
    std::make_tuple(32, 9, 3, 2, 63, 7, 7, 3, 3, 2, 2),
#if 1
    std::make_tuple(224, 224, 3, 2, 64, 7, 7, 3, 3, 2, 2),
    std::make_tuple(90, 30, 256, 1, 128, 1, 1, 0, 0, 1, 1),
    std::make_tuple(360, 120, 64, 1, 32, 1, 1, 0, 0, 1, 1),
    std::make_tuple(180, 60, 128, 1, 64, 1, 1, 0, 0, 1, 1),
    std::make_tuple(90, 30, 256, 1, 128, 1, 1, 0, 0, 1, 1),
    std::make_tuple(45, 15, 512, 1, 256, 1, 1, 0, 0, 1, 1),
    std::make_tuple(45, 15, 512, 1, 80, 1, 1, 0, 0, 1, 1),
    std::make_tuple(45, 15, 512, 1, 20, 1, 1, 0, 0, 1, 1),
    std::make_tuple(45, 15, 512, 1, 30, 1, 1, 0, 0, 1, 1),
    // from deepbench
    std::make_tuple(45, 15, 512, 1, 64, 1, 1, 0, 0, 1, 1),
    std::make_tuple(90, 30, 256, 1, 512, 2, 2, 0, 0, 2, 2),
    std::make_tuple(224, 224, 15, 2, 64, 7, 7, 3, 3, 2, 2),
    std::make_tuple(224, 224, 3, 2, 64, 7, 7, 3, 3, 1, 1),
    std::make_tuple(30, 30, 3, 2, 64, 7, 7, 3, 3, 1, 1),
    std::make_tuple(20, 20, 3, 2, 4, 7, 7, 3, 3, 2, 2),
    std::make_tuple(14, 14, 512, 1, 48, 5, 5, 2, 2, 1, 1),
    std::make_tuple(7, 7, 832, 1, 256, 1, 1, 0, 0, 1, 1),
    std::make_tuple(7, 7, 832, 1, 128, 5, 5, 2, 2, 1, 1),
    std::make_tuple(28, 28, 192, 2, 32, 5, 5, 2, 2, 1, 1),
#endif
    //  densebox
    //std::make_tuple(768, 272,  32, 3, 48, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(768, 272,  32, 1, 48, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(768, 272,  48, 1, 48, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(768, 272,  64, 1, 64, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(384, 136,  64, 1, 64, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(384, 136,  80, 1, 80, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(192, 68,  80, 1, 80, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(192, 68,  96, 1, 96, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(96, 34,  96, 1, 96, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(48, 17,  128, 1, 128, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(192, 68,  24, 1, 18, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(192, 68,  18, 1, 36, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(192, 68,  36, 1, 18, 3, 3, 1, 1, 1, 1),
    //  traffic lights
    //std::make_tuple( 32,  32,  32, 1, 64, 3, 3, 1, 1, 1, 1),
    //  cnn
    //std::make_tuple(256, 256,  96, 1, 48, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(256, 256,  48, 1, 48, 3, 3, 1, 1, 1, 1),
    //std::make_tuple(128, 128, 128, 1, 64, 3, 3, 1, 1, 1, 1),
    //std::make_tuple( 32,  32, 256, 1, 128, 3, 3, 1, 1, 1, 1),
    // depth
    //std::make_tuple(256,  48, 160, 1, 160, 3, 3, 1, 1, 1, 1),
    // more ...
};



TEST(TestSaberFuncNV, test_conv_fp32_conv_direct) {

    int n, h, w, group;
    int k, c, r, s;
    int pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w;

    for (const auto& problem : problem_set) {
        std::tie(w, h, c, n, k, s, r, pad_w, pad_h, stride_w, stride_h) = problem;

        dilate_h = 1;
        dilate_w = 1;
        group    = 1;

        int num_repeats = 1;

        Shape img_dim(n, c, h, w);
        Shape flt_dim(k, c, r, s);
        Shape bias_dim(1, k, 1, 1);

        TensorHf4 img_h;
        TensorDf4 img_d;
        TensorHf4 cudnn_out_h;
        TensorHf4 saber_out_h;
        {
            img_h.re_alloc(img_dim);
            img_d.re_alloc(img_dim);
            fill_tensor_host_rand(img_h, -1.0f, 1.0f);
            //fill_tensor_host_const(img_h, 1.f);
            img_d.copy_from(img_h);
        }

        TensorHf4 flt_h;
        TensorDf4 flt_d;
        {
            flt_h.re_alloc(flt_dim);
            flt_d.re_alloc(flt_dim);
            fill_tensor_host_rand(flt_h, -1.0f, 1.0f);
            //fill_tensor_host_const(flt_h, 1.f);
            flt_d.copy_from(flt_h);
        }

        TensorHf4 bias_h;
        TensorDf4 bias_d;

        if (1) {
            bias_h.re_alloc(bias_dim);
            bias_d.re_alloc(bias_dim);

            fill_tensor_host_rand(bias_h, -1.0f, 1.0f);
            //fill_tensor_host_const(bias_h, 0.f);
            bias_d.copy_from(bias_h);
        }

        TensorDf4 saber_out_d;
        TensorDf4 cudnn_out_d;

        Context<NV> ctx1(0, 1, 1);
        ConvParam<TensorDf4> param(group, pad_h, pad_w,
                                   stride_h, stride_w,
                                   dilate_h, dilate_w,
                                   &flt_d, &bias_d, 1.0f, 0.0f);

        std::vector<TensorDf4*> inputs;
        std::vector<TensorDf4*> cudnn_outputs;
        std::vector<TensorDf4*> saber_outputs;

        inputs.push_back(&img_d);
        saber_outputs.push_back(&saber_out_d);
        cudnn_outputs.push_back(&cudnn_out_d);

        Conv<NV, AK_FLOAT> saber_conv;
        saber_conv.compute_output_shape(inputs, saber_outputs, param);
        saber_out_d.re_alloc(saber_outputs[0]->shape());
        saber_out_h.re_alloc(saber_outputs[0]->shape());
        {
            cudaDeviceSynchronize();
            saber_conv.init(inputs, saber_outputs, param, SPECIFY, SABER_IMPL, ctx1);
            cudaDeviceSynchronize();
            saber_conv(inputs, saber_outputs, param, ctx1);
            cudaDeviceSynchronize();
            saber_out_h.copy_from(*saber_outputs[0]);
            cudaDeviceSynchronize();
        }

        Conv<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> cudnn_conv;
        cudnn_conv.compute_output_shape(inputs, cudnn_outputs, param);
        cudnn_out_d.re_alloc(cudnn_outputs[0]->shape());
        cudnn_out_h.re_alloc(cudnn_outputs[0]->shape());
        {
            cudaDeviceSynchronize();
            cudnn_conv.init(inputs, cudnn_outputs, param, SPECIFY, VENDER_IMPL, ctx1);
            cudaDeviceSynchronize();
            cudnn_conv(inputs, cudnn_outputs, param, ctx1);
            cudaDeviceSynchronize();
            cudnn_out_h.copy_from(*cudnn_outputs[0]);
            cudaDeviceSynchronize();
        }

        std::cout << "Running Error: "
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
        std::cout << "cudnn out size = " << cudnn_outputs[0]->size() << std::endl;
        std::cout << "saber out size = " << saber_outputs[0]->size() << std::endl;
        {
            bool pass = true;
            int  error_count = 0;
            float thresh = 0.00001f;
            //int out_num = saber_outputs[0]->size();
            int out_num = cudnn_outputs[0]->size();

            for (int i = 0; i < out_num; i++) {
                float cudnn_var = cudnn_out_h.data(0)[i];
                float saber_var = saber_out_h.data(0)[i];
                float delta = std::abs(saber_var - cudnn_var);
                float quote = std::min(std::abs(saber_var), std::abs(cudnn_var));
                quote = quote == 0 ? 1.0f : quote;
                delta = delta / quote;

                if (delta > thresh) {
                    pass = false;
                    std::cout << "Fail in check: i = " << i << std::endl;
                    printf("cudnn[%d] = %.6f, saber[%d] = %.6f, delta = %.6f\n", \
                           i, cudnn_var, i, saber_var, delta);
                    error_count += 1;

                    if (error_count == 10) {
                        break;
                    }
                }
            }

            if (pass) {
                std::cout << "Check Conv PASS ... " << std::endl;
            } else {
                std::cout << "Check Conv FAIL ... " << std::endl;
            }
        }
    }
}


int main(int argc, const char** argv) {
    anakin::saber::Env<NV>::env_init();

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

