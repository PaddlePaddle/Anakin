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
#include "saber/funcs/conv.h"
#include "test_saber_func_AMD.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_funcs_param.h"
#include "saber_types.h"
#include <vector>
using namespace anakin::saber;

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<AMD, AK_FLOAT, NCHW> TensorDf4;
typedef TargetWrapper<AMD> API;
typedef struct ScoreTypde
{
    double ElapsedMilliSec;
    double ElapsedMilliSecBest;
    double TFlops;
    double Calculation;
    double TheoryElapsedMilliSec;
    double perf;
}T_Score;

typedef struct StaticsTypde
{
    std::string ConfigName;
    T_Score score;
    double max_r;
    double max_d;;
}T_Statics;

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

void test_conv_fp32_speed(std::vector<TensorDf4*> &inputs, std::vector<TensorDf4*> &outputs,
                         TensorDf4 &weights, int stride, int pad, TensorDf4 &bias,
                         anakin::saber::ImplEnum impl, int warm_iter, int iter, Context<AMD> ctx1, T_Score& score) {

    ConvParam<TensorDf4> conv_param(1, pad, pad,
                                   stride, stride,
                                   1, 1,
                                   &weights, &bias);
    Conv<AMD, AK_FLOAT> conv;
    conv.compute_output_shape(inputs, outputs, conv_param);
    //MUST USE "valid_shape()"
    outputs[0]->re_alloc(outputs[0]->valid_shape());

    SABER_CHECK(conv.init(inputs, outputs, conv_param, SPECIFY, impl, ctx1));

    for(int i =0 ; i < warm_iter; i++)
        conv(inputs, outputs, conv_param, ctx1);
    clFinish(ctx1.get_compute_stream());

    SaberTimer<AMD> t1;
    for(int i =0 ; i < iter ; i++) {
        t1.start(ctx1);
        conv(inputs, outputs, conv_param, ctx1);
        t1.end(ctx1);
    }

    score.ElapsedMilliSec = t1.get_average_ms();
    score.ElapsedMilliSecBest = t1.get_best_ms();

    clFlush(ctx1.get_compute_stream());
    clFinish(ctx1.get_compute_stream());
}

#ifdef TEST_CONV_1X1
TEST(TestSaberFuncAMD, test_conv_fp32_1x1_speed) {
    int img_num = 1;
    int kernel = 1;

    int out_channels = 64;
    int in_channels = 192;
    int img_h = 28;
    int img_w = 28;
    int batch = 16;
    int pad = 0;
    int stride = 1;
    Context<AMD> ctx1(0, 1, 1);

    API::stream_t amd_cstream = ctx1.get_compute_stream();

    TensorDf4 weights;
    TensorHf4 weights_host;
    weights.re_alloc({out_channels, in_channels, 1, 1});
    weights_host.re_alloc({out_channels, in_channels, 1, 1});

    TensorDf4 img;
    TensorHf4 img_host;
    img.re_alloc({batch, in_channels, img_h, img_w});
    img_host.re_alloc({batch, in_channels, img_h, img_w});

    TensorDf4 out;
    TensorHf4 out_h;
    out.re_alloc({batch, out_channels, img_h, img_w});
    out_h.re_alloc({batch, out_channels, img_h, img_w});

    fill_tensor_device_rand(weights, -50.f, 50.f);
    fill_tensor_device_rand(img, -50.f, 50.f);
    //fill_tensor_device_const(weights, 2.f);
    //fill_tensor_host_const(weights_host, 2.f);
    //fill_tensor_device_const(img, 1.f);
    //fill_tensor_host_const(img_host, 1.f);
    print_tensor_shape("weights", weights);
    fill_tensor_device_const(out, 3.f);
    //print_tensor_device(weights);
    print_tensor_shape("img", img);
    //print_tensor_device(img);
    print_tensor_shape("out", out);
    //print_tensor_device(out);

    LOG(INFO) << "img_num: " << img_num;
    LOG(INFO) << "kernel: " << kernel;
    LOG(INFO) << "out_channels: " << out_channels;
    LOG(INFO) << "in_channels: " << in_channels;
    LOG(INFO) << "img_h: " << img_h;
    LOG(INFO) << "img_w: " << img_w;
    LOG(INFO) << "pad: " << pad;
    LOG(INFO) << "stride: " << stride;

    TensorDf4 bias;

    std::vector<TensorDf4*> input_v, output_v;

    input_v.push_back(&img);
    output_v.push_back(&out);
    clFlush(amd_cstream);
    clFinish(amd_cstream);

    test_conv_fp32_speed(input_v, output_v,
                         weights, kernel, stride, pad,
                         in_channels, out_channels, bias,
                         SABER_IMPL);

    clFlush(amd_cstream);
    clFinish(amd_cstream);

    TensorHf4 out_host;
    //TensorHf4 out_gemm_host;
    out_host.re_alloc(out.shape());
    out_host.copy_from(out);

    LOG(INFO) << "PRINT TENSOR: weights";
    //print_tensor_device(weights);
    LOG(INFO) << "PRINT TENSOR: img";
    //print_tensor_device(img);
    LOG(INFO) << "PRINT TENSOR: out";
    print_tensor_device(out);
    
    //out_gemm_host.re_alloc(out_gemm.shape());
    //out_gemm_host.copy_from(out_gemm);
    //double max_r, max_d;
    //tensor_cmp_host(out_host.data(), out_gemm_host.data(), out_host.size(), max_r, max_d);
    //LOG(INFO) << "cmp result: max_r = " << max_r << " max_d = " << max_d;
}
#endif

typedef struct ProblemConfigType
{
    std::string ConfigName;
    int N;
    int W, H;
    int C, K;
    int X, Y;
    int PadW, PadH;
}T_ProblemConfig;
    
void ConvFwd3x3_Host(TensorHf4* out_host, TensorHf4& in_host, TensorHf4& weights_host, T_ProblemConfig* problemConfig)
{
    LOG(INFO) << "conv fwd 3x3 host.";
    //int c = problemConfig->C, w = problemConfig->W, h = problemConfig->H, k = problemConfig->K;
    int u = 1; // stride height
    int v = 1; // stride width
    int dilation_h = 1;
    int dilation_w = 1;

    int batch_size = problemConfig->N;
    int channel_in = problemConfig->C;
    int feature_out = problemConfig->K;
    int width_in = problemConfig->W;
    int height_in = problemConfig->H;
    int width_wei = problemConfig->X;
    int height_wei = problemConfig->Y;
    int pad_w = problemConfig->PadW;
    int pad_h = problemConfig->PadH;
    int width_out = width_in + pad_w * 2 - width_wei + 1;
    int height_out = height_in + pad_h * 2 - height_wei + 1;    // output size
    //int size_in, size_wei, size_out;
    int stride_n_in = width_in * height_in * channel_in;   // stride for differetn batch of input
    int stride_c_in = width_in * height_in;                // stride for differetn channel in the same batch of input
    int stride_k_wei = width_wei * height_wei * channel_in;// stride for differetn feature of weight
    int stride_c_wei = width_wei * height_wei;             // stride for differetn channel in the same feature of weight
    int stride_n_out = out_host->width() * out_host->height() * feature_out;// stride for differetn bathc of output
    int stride_k_out = out_host->width() * out_host->height();  // stride for differetn feature in the same batch of output

    //TensorHf4 weights_host, in_host;
    //weights_host.re_alloc({feature_out, channel_in, height_wei, width_wei});
    //in_host.re_alloc({batch_size, channel_in, width_in, height_in});
    //Fill the data into memory
    //fill_tensor_host_const(in_host, 1.f);
    //fill_tensor_host_const(weights_host, 2.f);

    for (int o = 0; o < batch_size; o++)             // for batch size
    {
        for (int w = 0; w < feature_out; w++)        // for output features
        { 
            for (int i = 0; i < height_out; i++)     // for output heigh
            { 
                int in_off_h = i * u;
                for (int j = 0; j < width_out; j++)  // for output width
                {
                    float acc = 0;
                    int in_off_w = j * v;
                    for (int k = 0; k < channel_in; k++)        // sum input channels
                    {
                        for (int x = 0; x < height_wei; x++)        // for filter heigh
                        {
                            int in_x = in_off_h - pad_h + x * dilation_h;

                            if (in_x >= 0 && in_x < height_in)
                            {
                                for (int y = 0; y < width_wei; y++)    // for filter width
                                {
                                    int in_y = in_off_w - pad_w + y * dilation_w;// start idx of input a line for conv

                                    if (in_y >= 0 && in_y < width_in)
                                    {
                                        acc +=
                                            in_host.data()[o * stride_n_in + k * stride_c_in + in_x * width_in + in_y] *
                                            weights_host.data()[w * stride_k_wei + k * stride_c_wei + x * width_wei + y];
                                    }
                                }
                            }
                        }
                    }
                    out_host->mutable_data()[o * stride_n_out + w * stride_k_out + i * width_out + j] = acc;
                }
            }
        }
    }
}

TEST(TestSaberFuncAMD, test_vgg_conv_3x3) {

    std::list<T_ProblemConfig*> problemConfigList;
    std::list<T_Statics*> staticsList;
    T_ProblemConfig * problemConfig;
    T_Statics * statics;

    TensorDf4 weights;
    TensorHf4 weights_host;
    TensorDf4 in;
    TensorHf4 in_host;
    TensorDf4 bias;
    TensorDf4 out;
    TensorHf4 out_host;        //to store the result from device
    TensorHf4 out_golden_host; //to store the result from cpu calculation

    // ======================================================================
    // problem config conv11:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv11";
    problemConfig->N = 1;    //batch
    problemConfig->W = 224;  //width
    problemConfig->H = 224;  //height
    problemConfig->C = 3;    //channel
    problemConfig->K = 64;   //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv12:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv12";
    problemConfig->N = 1;    //batch
    problemConfig->W = 224;  //width
    problemConfig->H = 224;  //height
    problemConfig->C = 64;   //channel
    problemConfig->K = 64;   //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv21:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv21";
    problemConfig->N = 1;    //batch
    problemConfig->W = 112;  //width
    problemConfig->H = 112;  //height
    problemConfig->C = 64;   //channel
    problemConfig->K = 128;  //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv22:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv22";
    problemConfig->N = 1;    //batch
    problemConfig->W = 112;  //width
    problemConfig->H = 112;  //height
    problemConfig->C = 128;  //channel
    problemConfig->K = 128;  //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv31:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv31";
    problemConfig->N = 1;    //batch
    problemConfig->W = 56;   //width
    problemConfig->H = 56;   //height
    problemConfig->C = 128;  //channel
    problemConfig->K = 256;  //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv32:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv32";
    problemConfig->N = 1;    //batch
    problemConfig->W = 56;   //width
    problemConfig->H = 56;   //height
    problemConfig->C = 256;  //channel
    problemConfig->K = 256;  //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv33:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv32";
    problemConfig->N = 1;    //batch
    problemConfig->W = 56;   //width
    problemConfig->H = 56;   //height
    problemConfig->C = 256;  //channel
    problemConfig->K = 256;  //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv41:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv41";
    problemConfig->N = 1;    //batch
    problemConfig->W = 28;   //width
    problemConfig->H = 28;   //height
    problemConfig->C = 256;  //channel
    problemConfig->K = 512;  //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv42:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv42";
    problemConfig->N = 1;    //batch
    problemConfig->W = 28;   //width
    problemConfig->H = 28;   //height
    problemConfig->C = 512;  //channel
    problemConfig->K = 512;  //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv43:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv43";
    problemConfig->N = 1;    //batch
    problemConfig->W = 28;   //width
    problemConfig->H = 28;   //height
    problemConfig->C = 512;  //channel
    problemConfig->K = 512;  //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv51:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv51";
    problemConfig->N = 1;    //batch
    problemConfig->W = 14;   //width
    problemConfig->H = 14;   //height
    problemConfig->C = 512;  //channel
    problemConfig->K = 512;  //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv52:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv52";
    problemConfig->N = 1;    //batch
    problemConfig->W = 14;   //width
    problemConfig->H = 14;   //height
    problemConfig->C = 512;  //channel
    problemConfig->K = 512;  //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
    problemConfigList.push_back(problemConfig);

    // ======================================================================
    // problem config conv53:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv53";
    problemConfig->N = 1;    //batch
    problemConfig->W = 14;   //width
    problemConfig->H = 14;   //height
    problemConfig->C = 512;  //channel
    problemConfig->K = 512;  //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
    problemConfigList.push_back(problemConfig);

    Context<AMD> ctx1(0, 1, 1);

    API::stream_t amd_cstream = ctx1.get_compute_stream();

    LOG(INFO) << "Total " << problemConfigList.size() << " problems...";

    //Check the Device's max frequency
    Device<AMD> dev = Env<AMD>::cur_env()[0];
    cl_device_id id = dev.get_device();
    unsigned int ComputeUnitNum;
    unsigned int ProcessingElementNum;
    unsigned int tmpCoreFreq;
    double CoreFreq;
    double Fp32Flops;
#define     GPU_SIMD_NUM_PER_CU     (4)
#define     GPU_ALU_NUM_PER_SIMD    (16)
    // compute unite
    clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &ComputeUnitNum, NULL);
    ProcessingElementNum = ComputeUnitNum * GPU_SIMD_NUM_PER_CU * GPU_ALU_NUM_PER_SIMD;

    // clock frequency
    clGetDeviceInfo(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &tmpCoreFreq, NULL);
    CoreFreq = tmpCoreFreq * 1e6;
    Fp32Flops = ProcessingElementNum * CoreFreq * 2;

    //Begin loop
    for (auto p : problemConfigList)
    {
        //auto p = *problemConfigList.begin();
        //TODO: get the problem and solve it...
        LOG(INFO) << "Problem: " << p->ConfigName;
        statics = new T_Statics();

        //allocate weights buffer
        weights.re_alloc({p->K, p->C, p->Y, p->X});
        weights_host.re_alloc({p->K, p->C, p->Y, p->X});

        //allocate input buffer
        in.re_alloc({p->N, p->C, p->H, p->W});
        in_host.re_alloc({p->N, p->C, p->H, p->W});

        //assign default value to weights and input buffer
        //fill_tensor_device_const(weights, 2.f);
        //fill_tensor_device_const(in, 1.f);
        fill_tensor_device_rand(weights, -50.f, 50.f);
        fill_tensor_device_rand(in, -50.f, 50.f);

        std::vector<TensorDf4*> input_v, output_v;

        input_v.push_back(&in);
        output_v.push_back(&out);

        //wait for device ready
        clFlush(amd_cstream);
        clFinish(amd_cstream);

        test_conv_fp32_speed(input_v, output_v,
                 weights, 1, 1,
                 bias, SABER_IMPL, warm_iter, iter, ctx1, statics->score);

        print_tensor_shape("weights", weights);
        print_tensor_shape("in", in);
        print_tensor_shape("out", out);

        //wait for device ready
        clFlush(amd_cstream);
        clFinish(amd_cstream);

        out_host.re_alloc(out.shape());
        out_host.copy_from(out);

        out_golden_host.re_alloc(out.shape());
        in_host.copy_from(in);
        weights_host.copy_from(weights);
        ConvFwd3x3_Host(&out_golden_host, in_host, weights_host, p);

        //LOG(INFO) << "PRINT DEVICE TENSOR: weights";
        //print_tensor_device(weights);
        //sleep(2);

        //LOG(INFO) << "PRINT DEVICE TENSOR: img";
        //print_tensor_device(img);
        //sleep(2);

        //LOG(INFO) << "PRINT DEVICE TENSOR: out";
        //print_tensor_device(out);
        //sleep(2);

        //LOG(INFO) << "PRINT HOST TENSOR: out";
        //print_tensor_host(out_golden_host);
        //sleep(2);

        //To generate the report
        //double max_r, max_d;
        statics->ConfigName = p->ConfigName;
        tensor_cmp_host(out_host.data(), out_golden_host.data(), out_host.size(), statics->max_r, statics->max_d);
        //LOG(INFO) << "ConfigName = " << p->ConfigName << "cmp result: max_r = " << max_r << " max_d = " << max_d;
        //CHECK_LE(fabs(statics->max_r), 1.0e-6) << "error result";

        statics->score.Calculation = p->W * p->H * p->C * p->N * p->K * p->X * p->Y;

        statics->score.TheoryElapsedMilliSec = statics->score.Calculation / Fp32Flops * 1000;
        statics->score.TFlops = statics->score.Calculation / statics->score.ElapsedMilliSec / 1e9;
        statics->score.perf =  statics->score.TheoryElapsedMilliSec / statics->score.ElapsedMilliSec * 100;

        staticsList.push_back(statics);
    }

    LOG(INFO) << "GPU_CORE_FREQ_HZ: " << CoreFreq << " =================================";
    for (auto s : staticsList) {
        LOG(INFO) << "-----ConfigName:            " << s->ConfigName << "-----";
        LOG(INFO) << "     max_r:                 " << s->max_r;
        LOG(INFO) << "     max_d:                 " << s->max_d;
        LOG(INFO) << "     ElapsedMilliSec(Avg.): " << s->score.ElapsedMilliSec;
        LOG(INFO) << "     ElapsedMilliSec(Best): " << s->score.ElapsedMilliSecBest;
        LOG(INFO) << "     TFlops:                " << s->score.TFlops;
        LOG(INFO) << "     Calculation:           " << s->score.Calculation;
        LOG(INFO) << "     TheoryElapsedMilliSec: " << s->score.TheoryElapsedMilliSec;
        LOG(INFO) << "     perfs:                 " << s->score.perf;
        LOG(INFO) << " ";
    }
    LOG(INFO) << "===========================================================";
}

int main(int argc, const char** argv){
    anakin::saber::Env<AMD>::env_init();

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

