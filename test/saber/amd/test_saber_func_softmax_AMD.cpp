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
#include "saber/core/context.h"
#include "saber/funcs/softmax.h"
#include "test_saber_func_AMD.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_funcs_param.h"
#include "saber_types.h"
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

void test_softmax(std::vector<TensorDf4*> &inputs, std::vector<TensorDf4*> &outputs,
                         anakin::saber::ImplEnum impl, int warm_iter, int iter, Context<AMD> ctx1, SaberTimer<AMD> &timer) {

    int noused = 0;
    SoftmaxParam<TensorDf4> param(noused);

    Softmax<AMD, AK_FLOAT> softmax;
    softmax.compute_output_shape(inputs, outputs, param);
    //MUST USE "valid_shape()"
    outputs[0]->re_alloc(outputs[0]->valid_shape());

    SABER_CHECK(softmax.init(inputs, outputs, param, SPECIFY, impl, ctx1));

    for(int i=0; i<warm_iter; i++) {
        softmax(inputs, outputs, param, ctx1);
    }
    clFinish(ctx1.get_compute_stream());

    for(int i=0; i<iter; i++) {
        timer.start(ctx1);
        softmax(inputs, outputs, param, ctx1);
        timer.end(ctx1);
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
}T_ProblemConfig;

#define FLT_MAX 3.402823466e+38f
void Softmax_Host(TensorHf4* out_host, TensorHf4* in_host, T_ProblemConfig* problemConfig)
{
    printf("softmax host.\n");

    int batch = problemConfig->N;
    int channel = problemConfig->C;
    int width = problemConfig->W;
    int height = problemConfig->H;

    int tensor_size = width * height * channel * batch;
    int grid_size = width * height * batch;
    int stride_c = width * height;

    std::vector<float> channel_max(grid_size, -FLT_MAX);
    out_host->copy_from(*in_host);

    //print_tensor_host(*out_host);
    for (int i = 0; i < batch; i++)
    {
        for (int s = 0; s < height * width; s++)
        {
            for (int j = 0; j < channel; j++)
            {
                channel_max[i * height * width + s] =
                    std::max(out_host->mutable_data()[(i * channel + j) * height * width + s],
                    channel_max[i * height * width + s]);
            }

            for (int j = 0; j < channel; j++)
            {
                out_host->mutable_data()[(i * channel + j) * height * width + s] -= channel_max[i * height * width + s];
                out_host->mutable_data()[(i * channel + j) * height * width + s] = std::exp(out_host->mutable_data()[(i * channel + j) * height * width + s]);
            }

            channel_max[i * height * width + s] = 0.0;
            for (int j = 0; j < channel; j++)
            {
                channel_max[i * height * width + s] += out_host->mutable_data()[(i * channel + j) * height * width + s];
            }

            for (int j = 0; j < channel; j++)
            {
                out_host->mutable_data()[(i * channel + j) * height * width + s] /= channel_max[i * height * width + s];
            }
        }
    }
}

static void Verify(TensorHf4& h_out, TensorHf4& out_ref, int size)
{
    LOG(INFO) << "softmax verify";

    float diff = 0;
    for (int i = 0; i < size; i++)
    {
        diff += (out_ref.data()[i] - h_out.data()[i])*(out_ref.data()[i] - h_out.data()[i]);
    }
        diff /= size;

    LOG(INFO) << "mean err: " << diff;
    if (diff > 1e-6)
    {
        LOG(INFO) << "err: " << diff;
        LOG(INFO) << "verify failed!";
    } else {
        LOG(INFO) << "verify success.";
    }
}

TEST(TestSaberFuncAMD, test_vgg_softmax) {

    std::list<T_ProblemConfig*> problemConfigList;
    T_ProblemConfig * problemConfig;

    TensorDf4 in;
    TensorHf4 in_host;
    TensorDf4 bias;
    TensorDf4 out;
    TensorHf4 out_host;        //to store the result from device
    TensorHf4 out_golden_host; //to store the result from cpu calculation

    // ======================================================================
    // problem config Softmax53:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Softmax53";
    problemConfig->N = 4;    //batch
    problemConfig->W = 14;  //width
    problemConfig->H = 14;  //height
    problemConfig->C = 512;    //channel
    problemConfig->K = 512;   //kernels
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
        //fill_tensor_device_const(in, 0.3);
        fill_tensor_host_rand(in_host, 0.f, 1.f);
        //fill_tensor_device_rand(in);
        in.copy_from(in_host);

        std::vector<TensorDf4*> input_v, output_v;

        input_v.push_back(&in);
        output_v.push_back(&out);

        //wait for device ready
        clFlush(amd_cstream);
        clFinish(amd_cstream);

        test_softmax(input_v, output_v, SABER_IMPL, warm_iter, iter, ctx1, t_device);

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
        Softmax_Host(&out_golden_host, &in_host, p);
        t_host.end(ctx1);

        //LOG(INFO) << "PRINT DEVICE TENSOR: in";
        //print_tensor_device(in);
        //LOG(INFO) << "PRINT HOST TENSOR: in";
        //print_tensor_host(in_host);
        //sleep(2);

        //LOG(INFO) << "PRINT DEVICE TENSOR: out";
        //print_tensor_device(out);
        //sleep(2);

        //LOG(INFO) << "PRINT HOST TENSOR: out_golden";
        //print_tensor_host(out_golden_host);
        //sleep(2);

        double max_r, max_d;
        tensor_cmp_host(out_host.data(), out_golden_host.data(), out_host.size(), max_r, max_d);
        LOG(INFO) << "ConfigName = " << p->ConfigName << "cmp result: max_r = " << max_r << " max_d = " << max_d;
        LOG(INFO) << "cmp elapse time: device(" << t_device.get_best_ms() << " ms) : host(" << t_host.get_average_ms() << " ms)";

        Verify(out_host, out_golden_host, out_host.size());
    }
}
int main(int argc, const char** argv){
    anakin::saber::Env<AMD>::env_init();

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

