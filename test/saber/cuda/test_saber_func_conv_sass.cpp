
#include "core/context.h"
#include "funcs/conv.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>
#include <map>
using namespace anakin::saber;

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
#if 0
std::vector<int> param0 {1, 3, 480, 1440, 16};
std::vector<int> param1{1, 16, 240, 720, 32};
std::vector<int> param2{1, 32, 120, 360, 64};
std::vector<int> param3{1, 64, 60, 180, 128};
std::vector<int> param4{1, 128, 30, 90, 256};
std::vector<int> param5{1, 256, 15, 45, 512};
std::vector<int> param6{1, 512, 15, 45, 512};
std::vector<int> param7{1, 1024, 15, 45, 512};
#else
std::vector<int> param0 {3, 31, 483, 1440, 116};
std::vector<int> param1{3, 161, 247, 720, 31};
std::vector<int> param2{3, 321, 129, 360, 63};
std::vector<int> param3{3, 641, 63, 180, 127};
std::vector<int> param4{3, 128, 16, 16, 128};
std::vector<int> param5{3, 128, 63, 181, 128};
std::vector<int> param6{3, 228, 15, 45, 256};
std::vector<int> param7{3, 1021, 15, 45, 514};
#endif
std::vector<int> param_test {2, 32, 16, 16, 128};
std::map<std::string, std::vector<int>> param_map;


TEST(TestSaberFuncNV, test_conv_fp32_conv_sass) {

#if 1
    param_map.insert(std::pair<std::string, std::vector<int>>("conv0", param0));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv1", param1));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv2", param2));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv3", param3));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv4", param4));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv5", param5));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv6", param6));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv7", param7));
#else
    param_map.insert(std::pair<std::string, std::vector<int>>("param_test", param_test));
#endif

    for (int jj = 0; jj < 20; jj ++) {
        for (auto iter = param_map.begin(); iter != param_map.end(); iter++) {

            int img_num = iter->second[0];
            int in_channels = iter->second[1];
            int img_h = iter->second[2];
            int img_w = iter->second[3];
            int out_channel = iter->second[4];

            int group = 1;
            int pad_h = 1;
            int pad_w = 1;
            int stride_h = 1;
            int stride_w = 1;
            int dilation_h = 1;
            int dilation_w = 1;

            int kernel_h = 3;
            int kernel_w = 3;
            bool bias_term = true;

            Shape img_s(img_num, in_channels, img_h, img_w);
            Shape weights_s(out_channel, in_channels, kernel_h, kernel_w);
            Shape bias_s(1, out_channel, 1, 1);

            TensorHf4 img_host;
            TensorHf4 result_cudnn;
            TensorHf4 result_saber;
            TensorDf4 img_dev;

            img_host.re_alloc(img_s);
            img_dev.re_alloc(img_s);
            fill_tensor_host_rand(img_host, -1.0, 1.0f);
            //fill_tensor_host_const(img_host,1.f);
            img_dev.copy_from(img_host);

            TensorHf4 weights_host;
            TensorDf4 weights_dev;

            weights_host.re_alloc(weights_s);
            weights_dev.re_alloc(weights_s);

            fill_tensor_host_rand(weights_host, -1.0, 1.0f);
            //fill_tensor_host_const(weights_host,1.f);
            weights_dev.copy_from(weights_host);

            TensorHf4 bias_host;
            TensorDf4 bias_dev;

            if (bias_term) {
                bias_host.re_alloc(bias_s);
                bias_dev.re_alloc(bias_s);

                //fill_tensor_host_const(bias_host,0.f);
                fill_tensor_host_rand(bias_host, -1.0, 1.0f);
                bias_dev.copy_from(bias_host);
            }

            TensorDf4 output_dev;
            TensorDf4 output_dev1;

            // start Reshape & doInfer
            Context<NV> ctx1(0, 1, 1);

            ConvParam<TensorDf4> param(group, pad_h, pad_w,
                                       stride_h, stride_w,
                                       dilation_h, dilation_w,
                                       &weights_dev, &bias_dev, 1.0f, 0.0f);

            std::vector<TensorDf4*> input;
            std::vector<TensorDf4*> output;
            std::vector<TensorDf4*> output1;

            input.push_back(&img_dev);
            output.push_back(&output_dev);

            Conv<NV, AK_FLOAT> conv;
            conv.compute_output_shape(input, output, param);
            output_dev.re_alloc(output[0]->shape());
            std::cout << "output shape = " << output[0]->shape()[0] << " " << output[0]->shape()[1] << " "
                      << output[0]->shape()[2] << " " << output[0]->shape()[3] << std::endl;
            result_cudnn.re_alloc(output[0]->shape());
            cudaDeviceSynchronize();
            conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
            cudaDeviceSynchronize();
            conv(input, output, param, ctx1);
            cudaDeviceSynchronize();
            output_dev.sync();
            result_cudnn.copy_from(*output[0]);
            cudaDeviceSynchronize();

            Conv<NV, AK_FLOAT> conv_saber;
            output1.push_back(&output_dev1);
            output_dev1.re_alloc(output[0]->shape());
            conv_saber.compute_output_shape(input, output, param);
            result_saber.re_alloc(output[0]->shape());
            conv_saber.init(input, output1, param, SPECIFY, SABER_IMPL, ctx1);
            cudaDeviceSynchronize();
            conv_saber(input, output1, param, ctx1);
            cudaDeviceSynchronize();
            output_dev1.sync();
            result_saber.copy_from(*output1[0]);
            cudaDeviceSynchronize();
            std::cout << "In saber" << std::endl;
            //print_tensor_device(output_dev1);
            CUDA_CHECK(cudaPeekAtLastError());

            bool ok = true;

            for (int i = 0; i < output[0]->size(); i++) {
                float err = result_saber.data(0)[i] - result_cudnn.data(0)[i];

                if (abs(err) > 0.001) {
                    ok = false;
                    std::cout << "Failed index = " << i << std::endl;
                    break;
                }
            }

            if (!ok) {
                std::cout << iter->first << " Check Failed" << std::endl;
            } else {
                std::cout << iter->first << " Check Pass" << std::endl;
            }
        }

    }
}

int main(int argc, const char** argv) {
    anakin::saber::Env<NV>::env_init();

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

