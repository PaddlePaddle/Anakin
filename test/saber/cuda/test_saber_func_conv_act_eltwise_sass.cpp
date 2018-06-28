
#include "core/context.h"
#include "funcs/conv_act.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>
#include <map>
#include <algorithm>
using namespace anakin::saber;



typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

#if 0
std::vector<int> param0 {3, 3, 480, 1440, 16, 1, 3};
std::vector<int> param1{3, 16, 240, 720, 32, 1, 3};
std::vector<int> param2{3, 32, 120, 360, 64, 1, 3};
std::vector<int> param3{3, 64, 60, 180, 128, 1, 3};
std::vector<int> param4{3, 128, 30, 90, 128, 1, 3};
std::vector<int> param5{3, 256, 15, 45, 512, 1, 3};
std::vector<int> param6{3, 512, 15, 45, 512, 1, 3};
std::vector<int> param7{3, 1024, 15, 45, 512, 1, 3};
#else
std::vector<int> param_test {1, 256, 16, 16, 512, 0, 1};
std::vector<int> param_test1 {1, 16, 240, 720, 32, 0, 1};
std::vector<int> param_test2 {1, 32, 120, 360, 64, 0, 1};
std::vector<int> param_test3 {1, 64, 60, 180, 128, 0, 1};
std::vector<int> param_test4 {1, 512, 15, 45, 512, 0, 1};
std::vector<int> param_test5 {1, 256, 16, 16, 512, 1, 3};
#endif
std::map<std::string, std::vector<int>> param_map;

//std::vector<EltwiseType> elt = {Eltwise_prod, Eltwise_sum, Eltwise_max};
std::vector<EltwiseType> elt = {Eltwise_sum};


TEST(TestSaberFuncNV, test_func_conv_relu_elt_fusion) {
#if 0
    param_map.insert(std::pair<std::string, std::vector<int>>("conv0", param0));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv1", param1));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv2", param2));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv3", param3));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv4", param4));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv5", param5));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv6", param6));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv7", param7));
#else
    param_map.insert(std::pair<std::string, std::vector<int>>("conv1", param_test1));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv2", param_test2));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv3", param_test3));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv4", param_test4));
    param_map.insert(std::pair<std::string, std::vector<int>>("conv5", param_test5));
#endif

//    for (int jj = 0; jj < 20; jj ++) {
        for (int a = 0; a < elt.size(); a++) {
            EltwiseType elt_type = elt[a];

            for (auto iter = param_map.begin(); iter != param_map.end(); iter++) {

                int img_num = iter->second[0];
                int in_channels = iter->second[1];
                int img_h = iter->second[2];
                int img_w = iter->second[3];
                int out_channel = iter->second[4];

                int group = 1;
                int pad_h = iter->second[5];
                int pad_w = iter->second[5];
                int stride_h = 1;
                int stride_w = 1;
                int dilation_h = 1;
                int dilation_w = 1;

                int kernel_h = iter->second[6];
                int kernel_w = iter->second[6];
                bool bias_term = true;

                Shape img_s(img_num, in_channels, img_h, img_w);
                Shape weights_s(out_channel, in_channels, kernel_h, kernel_w);
                Shape bias_s(1, out_channel, 1, 1);
                TensorHf4 img_host;
                TensorHf4 output_host;
                TensorHf4 result_cudnn;
                TensorHf4 result_saber;
                TensorDf4 img_dev;

                img_host.re_alloc(img_s);
                img_dev.re_alloc(img_s);
                fill_tensor_host_rand(img_host, -1.0f, 1.0f);
                //fill_tensor_host_const(img_host,1.f);
                img_dev.copy_from(img_host);

                TensorHf4 weights_host;
                TensorDf4 weights_dev;

                weights_host.re_alloc(weights_s);
                weights_dev.re_alloc(weights_s);

                fill_tensor_host_rand(weights_host, -1.0f, 1.0f);
                //fill_tensor_host_const(weights_host,1.f);
                weights_dev.copy_from(weights_host);

                TensorHf4 bias_host;
                TensorDf4 bias_dev;

                if (bias_term) {
                    bias_host.re_alloc(bias_s);
                    bias_dev.re_alloc(bias_s);

                    //fill_tensor_host_const(bias_host,0.f);
                    fill_tensor_host_rand(bias_host, -1.0f, 1.0f);
                    bias_dev.copy_from(bias_host);
                }

                TensorDf4 output_dev;
                TensorDf4 output_dev1;


                EltwiseParam<TensorDf4> elt_param(elt_type);
                // start Reshape & doInfer
                Context<NV> ctx1(0, 1, 1);

                ConvParam<TensorDf4> conv_param(group, pad_h, pad_w,
                                                stride_h, stride_w,
                                                dilation_h, dilation_w,
                                                &weights_dev, &bias_dev, 1.0f, 0.0f);

                ActivationParam<TensorDf4> active_param(Active_relu);

                ConvActiveParam<TensorDf4> param(conv_param, active_param, elt_param);

                std::vector<TensorDf4*> input;
                std::vector<TensorDf4*> output;
                std::vector<TensorDf4*> output1;

                input.push_back(&img_dev);
                output.push_back(&output_dev);

                ConvAct<NV, AK_FLOAT> conv;
                conv.compute_output_shape(input, output, param);
                output_dev.re_alloc(output[0]->shape());

                std::cout << "kernel = "<<iter->second[6]<<" output shape = " << output[0]->shape()[0] << " " << output[0]->shape()[1] << " "
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
                std::cout<<"calling saber conv !!!"<<std::endl;
                ConvAct<NV, AK_FLOAT> conv_saber;
                output1.push_back(&output_dev1);
                output_dev1.re_alloc(output[0]->shape());
                output_host.re_alloc(output[0]->shape());
                fill_tensor_host_rand(output_host, -1.0f, 1.0f);
                output_dev1.copy_from(output_host);
                output_dev1.sync();
                conv_saber.compute_output_shape(input, output, param);
                result_saber.re_alloc(output[0]->shape());
                conv_saber.init(input, output1, param, SPECIFY, SABER_IMPL, ctx1);
                cudaDeviceSynchronize();
                conv_saber(input, output1, param, ctx1);
                cudaDeviceSynchronize();
                output_dev1.sync();
                result_saber.copy_from(*output1[0]);
                cudaDeviceSynchronize();
                std::cout << "In saber Eltwise_mode = " << elt_type << std::endl;
                //print_tensor_device(output_dev1);
                CUDA_CHECK(cudaPeekAtLastError());

                bool ok = true;

                for (int i = 0; i < output[0]->size(); i++) {
                    float err;

                    if (elt_type == Eltwise_prod) {
                        err = result_saber.data()[i] - (result_cudnn.data()[i] * output_host.data()[i]);
                    } else if (elt_type == Eltwise_sum) {
                        err = result_saber.data()[i] - (result_cudnn.data()[i] + output_host.data()[i]);
                    } else if (elt_type == Eltwise_max) {
                        err = result_saber.data()[i] - std::max(result_cudnn.data()[i], output_host.data()[i]);
                    }

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
//    }
}

int main(int argc, const char** argv) {

    Env<NV>::env_init();
    LOG(INFO) << "tensor initialization finished";

    InitTest();

    RUN_ALL_TESTS(argv[0]);

    return 0;
}

