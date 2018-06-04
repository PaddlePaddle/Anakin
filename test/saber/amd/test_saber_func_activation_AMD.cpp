/* Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <vector>
#include "saber/core/context.h"
#include "saber/funcs/activation.h"
#include "test_saber_func_AMD.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_funcs_param.h"
#include "saber/saber_types.h"
#include <vector>
using namespace anakin::saber;

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<AMD, AK_FLOAT, NCHW> TensorDf4;
typedef TargetWrapper<AMD> API;

template <typename Tensor>
void print_tensor_shape(std::string name, Tensor &t0) {

            LOG(INFO) << name << " valid shape is ["
                      << t0.valid_shape()[0] << ", "
                      << t0.valid_shape()[1] << ", "
                      << t0.valid_shape()[2] << ", "
                      << t0.valid_shape()[3] << "].";

            LOG(INFO) << name << " real shape is ["
                      << t0.shape()[0] << ", "
                      << t0.shape()[1] << ", "
                      << t0.shape()[2] << ", "
                      << t0.shape()[3] << "].";

            LOG(INFO) << name << " offset is ["
                      << t0.offset()[0] << ", "
                      << t0.offset()[1] << ", "
                      << t0.offset()[2] << ", "
                      << t0.offset()[3] << "].";
}

void test_active(std::vector<TensorDf4*> &inputs, std::vector<TensorDf4*> &outputs,
                         int stride, int pad, TensorDf4 &bias,
                         anakin::saber::ImplEnum impl, int warm_iter, int iter, Context<AMD> &ctx1, SaberTimer<AMD> &t_device) {

    ActivationParam<TensorDf4> param(Active_relu);

    Activation<AMD, AK_FLOAT> activation;
    activation.compute_output_shape(inputs, outputs, param);
    //MUST USE "valid_shape()"
    outputs[0]->re_alloc(outputs[0]->valid_shape());

    SABER_CHECK(activation.init(inputs, outputs, param, SPECIFY, impl, ctx1));

    for(int i=0; i < warm_iter; i++)
        activation(inputs, outputs, param, ctx1);
    clFinish(ctx1.get_compute_stream());

    for(int i=0 ;i < iter; i++) {
        t_device.start(ctx1);
        activation(inputs, outputs, param, ctx1);
        t_device.end(ctx1);
     }

    clFlush(ctx1.get_compute_stream());
    clFinish(ctx1.get_compute_stream());
}

typedef struct ProblemConfigType
{
    std::string ConfigName;
    int N;
    int W, H;
    int C, K;
    float NegSlope; //for conv_relu
}T_ProblemConfig;

void Activation_Host(TensorHf4* out_host, TensorHf4& in_host, T_ProblemConfig* problemConfig)
{
    int negSlope = problemConfig->NegSlope;
    int size_in = problemConfig->N * problemConfig->C * problemConfig->H * problemConfig->W;

    for (int i = 0; i < size_in; i++)
    {
        out_host->mutable_data()[i] = in_host.data()[i] > 0 ? in_host.data()[i] : in_host.data()[i] * negSlope;
    }
}

TEST(TestSaberFuncAMD, test_activation) {

    std::list<T_ProblemConfig*> problemConfigList;
    T_ProblemConfig * problemConfig;

    TensorDf4 in;
    TensorHf4 in_host;
    TensorDf4 bias;
    TensorDf4 out;
    TensorHf4 out_host;        //to store the result from device
    TensorHf4 out_golden_host; //to store the result from cpu calculation
/*
    // ======================================================================
    // problem config conv12:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Relu12";
    problemConfig->N = 1;    //batch
    problemConfig->W = 224;  //width
    problemConfig->H = 224;  //height
    problemConfig->C = 64;    //channel
    problemConfig->K = 64;   //kernels
    problemConfig->NegSlope = 2;
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv22:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Relu22";
    problemConfig->N = 1;    //batch
    problemConfig->W = 112;  //width
    problemConfig->H = 112;  //height
    problemConfig->C = 128;    //channel
    problemConfig->K = 128;   //kernels
    problemConfig->NegSlope = 2;
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv33:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Relu33";
    problemConfig->N = 1;    //batch
    problemConfig->W = 56;  //width
    problemConfig->H = 56;  //height
    problemConfig->C = 256;    //channel
    problemConfig->K = 256;   //kernels
    problemConfig->NegSlope = 2;
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv43:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Relu43";
    problemConfig->N = 1;    //batch
    problemConfig->W = 28;  //width
    problemConfig->H = 28;  //height
    problemConfig->C = 512;    //channel
    problemConfig->K = 512;   //kernels
    problemConfig->NegSlope = 2;
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv53:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Relu53";
    problemConfig->N = 1;    //batch
    problemConfig->W = 14;  //width
    problemConfig->H = 14;  //height
    problemConfig->C = 512;    //channel
    problemConfig->K = 512;   //kernels
    problemConfig->NegSlope = 2;
    problemConfigList.push_back(problemConfig);
*/

    // ======================================================================
    // problem config conv53:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "4096C";
    problemConfig->N = 1;    //batch
    problemConfig->W = 1;  //width
    problemConfig->H = 1;  //height
    problemConfig->C = 4096;    //channel
    problemConfig->K = 4096;   //kernels
    problemConfig->NegSlope = 2;
    problemConfigList.push_back(problemConfig);
    Context<AMD> ctx1(0, 1, 1);

    API::stream_t amd_cstream = ctx1.get_compute_stream();

    LOG(INFO) << "Total " << problemConfigList.size() << " problems...";


    SaberTimer<AMD> t_device;
    SaberTimer<AMD> t_host;

    //Begin loop
    for (auto p : problemConfigList)
    {
        //TODO: get the problem and solve it...
        LOG(INFO) << "Problem: " << p->ConfigName;

        //allocate input buffer
        in.re_alloc({p->N, p->C, p->H, p->W});
        in_host.re_alloc({p->N, p->C, p->H, p->W});

        //assign default value to input buffer
        fill_tensor_device_rand(in, -1.f, 1.f);
        in_host.copy_from(in);

        std::vector<TensorDf4*> input_v, output_v;

        input_v.push_back(&in);
        output_v.push_back(&out);

        //wait for device ready
        clFlush(amd_cstream);
        clFinish(amd_cstream);

        test_active(input_v, output_v,
                         1, 1,
                         bias, SABER_IMPL, warm_iter, iter, ctx1, t_device);

        print_tensor_shape("in", in);
        print_tensor_shape("out", out);

        //wait for device ready
        clFlush(amd_cstream);
        clFinish(amd_cstream);

        out_host.re_alloc(out.shape());
        out_host.copy_from(out);

        //test result in host side.
        t_host.start(ctx1);
        out_golden_host.re_alloc(out.shape());
        Activation_Host(&out_golden_host, in_host, p);
        t_host.end(ctx1);

        //LOG(INFO) << "PRINT DEVICE TENSOR: img";
        //print_tensor_device(img);
        //sleep(2);

        //LOG(INFO) << "PRINT DEVICE TENSOR: out";
        print_tensor_device(out);
        //sleep(2);

        //LOG(INFO) << "PRINT HOST TENSOR: out";
        print_tensor_host(out_golden_host);
        //sleep(2);

        double max_r, max_d;
        tensor_cmp_host(out_host.data(), out_golden_host.data(), out_host.size(), max_r, max_d);
        LOG(INFO) << "ConfigName = " << p->ConfigName << "cmp result: max_r = " << max_r << " max_d = " << max_d;
        LOG(INFO) << "cmp elapse time: device( best : " <<t_device.get_best_ms() <<" ms , average : " << t_device.get_average_ms() << " ms) : host(" << t_host.get_average_ms() << " ms)";
    }
}
int main(int argc, const char** argv) {
    anakin::saber::Env<AMD>::env_init();
    anakin::saber::Env<X86>::env_init();
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

