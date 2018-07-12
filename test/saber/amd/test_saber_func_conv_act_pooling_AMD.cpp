/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

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
#include "saber/funcs/conv_act_pooling.h"
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

typedef struct ProblemConfigType
{
#define MLO_POOLING_OP_MAX 0
#define MLO_POOLING_OP_AVE 1
#define FLT_MAX 3.402823466e+38f

    std::string ConfigName;
    int N;
    int W, H;
    int C, K;
    int X, Y;
    int PadW, PadH;

    //relu config
    float NegSlope;

    // pooling config
    int PoolMethod;
    int PoolPadW, PoolPadH;
    int PoolStrideW, PoolStrideH;
    int PoolSizeW, PoolSizeH;

}T_ProblemConfig;

void test_conv_active_pooling(T_ProblemConfig* problemConfig, std::vector<TensorDf4*> &inputs, std::vector<TensorDf4*> &outputs,
                         TensorDf4 &weights, int stride, int pad, int slope, TensorDf4 &bias,
                         anakin::saber::ImplEnum impl, int warm_iter, int iter, Context<AMD> ctx1, T_Score& score) {

    ConvParam<TensorDf4> conv_param(1, pad, pad,
                                   stride, stride,
                                   1, 1,
                                   &weights, &bias);
    ActivationParam<TensorDf4> active_param(Active_relu, slope);
    PoolingParam<TensorDf4> pooling_param(problemConfig->PoolSizeH, problemConfig->PoolSizeW, problemConfig->PoolPadH, 
                                          problemConfig->PoolPadW, problemConfig->PoolStrideH, problemConfig->PoolStrideW, 
                                          Pooling_max);

    ConvActivePoolingParam<TensorDf4> param(conv_param, active_param, pooling_param);

    ConvActPooling<AMD, AK_FLOAT> conv;
    conv.compute_output_shape(inputs, outputs, param);
    //MUST USE "valid_shape()"
    outputs[0]->re_alloc(outputs[0]->valid_shape());

    print_tensor_shape("weights", weights);
    print_tensor_shape("in", *inputs[0]);
    print_tensor_shape("out", *outputs[0]);


    SABER_CHECK(conv.init(inputs, outputs, param, SPECIFY, impl, ctx1));

    for(int i =0 ; i < warm_iter; i++) {
        conv(inputs, outputs, param, ctx1);
    }
    clFinish(ctx1.get_compute_stream());

    SaberTimer<AMD> t1;
    Env<AMD>::start_record();
    for(int i =0 ; i < iter; i++) {
        t1.start(ctx1);
        conv(inputs, outputs, param, ctx1);
        t1.end(ctx1);
    }
    Env<AMD>::stop_record();

    score.ElapsedMilliSec = t1.get_average_ms();
    score.ElapsedMilliSecBest = t1.get_best_ms();

    clFlush(ctx1.get_compute_stream());
    clFinish(ctx1.get_compute_stream());

}

void ConvFwd3x3_Host(TensorHf4* out_host, TensorHf4* out_conv_host, TensorHf4& in_host, TensorHf4& weights_host, T_ProblemConfig* problemConfig)
{
    LOG(INFO) << "conv fwd 3x3 host.";
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
    int negSlope = problemConfig->NegSlope;

    int width_out = width_in + pad_w * 2 - width_wei + 1;
    int height_out = height_in + pad_h * 2 - height_wei + 1;

    int stride_n_in = width_in * height_in * channel_in;   // stride for differetn batch of input
    int stride_c_in = width_in * height_in;                // stride for differetn channel in the same batch of input
    int stride_k_wei = width_wei * height_wei * channel_in;// stride for differetn feature of weight
    int stride_c_wei = width_wei * height_wei;             // stride for differetn channel in the same feature of weight
    int stride_n_out = out_conv_host->width() * out_conv_host->height() * feature_out;// stride for differetn bathc of output
    int stride_k_out = out_conv_host->width() * out_conv_host->height();  // stride for differetn feature in the same batch of output

    int pool_width_pad = problemConfig->PoolPadW;		
    int pool_heigh_pad = problemConfig->PoolPadH;
    int pool_width_stride = problemConfig->PoolStrideW;	
    int pool_heigh_stride = problemConfig->PoolStrideH;
    int pool_width_kernel = problemConfig->PoolSizeW;
    int pool_heigh_kernel = problemConfig->PoolSizeH;
    int pool_width_in = width_out;
    int pool_heigh_in = height_out;
    int pool_width_out = pool_width_in / pool_width_kernel;
    int pool_heigh_out = pool_heigh_in / pool_heigh_kernel;
    int pool_stride_c_in = stride_k_out;  
    int pool_stride_n_in = stride_n_out;
    int pool_stride_c_out = stride_k_out / (pool_width_kernel * pool_heigh_kernel);
    int pool_stride_n_out = stride_n_out / (pool_width_kernel * pool_heigh_kernel);

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
                    out_conv_host->mutable_data()[o * stride_n_out + w * stride_k_out + i * width_out + j] = 
                      acc > 0 ? acc : acc * negSlope;
                }
            }
        }
    }
    int pad0 = pool_width_pad;
    int pad1 = pool_heigh_pad;
    int stride0 = pool_width_stride;
    int stride1 = pool_heigh_stride;
    int kernel_size0 = pool_width_kernel;
    int kernel_size1 = pool_heigh_kernel;
    
    // pooling in
    int bot_width = pool_width_in;
    int bot_height = pool_heigh_in;
    int bot_stride = pool_width_in;
    int bot_channel_stride = pool_stride_c_in;
    int bot_batch_stride = pool_stride_n_in;
    
    // pooling out 
    int top_width = pool_width_out;
    int top_height = pool_heigh_out;
    int top_stride = pool_width_out;
    int top_channel_stride = pool_stride_c_out;
    int top_batch_stride = pool_stride_n_out;
    
    float res = 0;
    for (int batch = 0; batch < batch_size; batch++)		// for batch size
    {
        for (int o = 0; o < feature_out; o++)
        { 
            for (int j = 0; j < top_height; j++)
            {
                for (int i = 0; i < top_width; i++)
                {
                    // c-emulator
                    if (problemConfig->PoolMethod == MLO_POOLING_OP_MAX)
                    {
                        res = -FLT_MAX;
                    }
                    else if (problemConfig->PoolMethod == MLO_POOLING_OP_AVE)
                    {
                        res = 0;
                    }
    
                    int hstart = j * stride1 - pad1;
                    int wstart = i * stride0 - pad0;
                    int hend = std::min(hstart + kernel_size1, bot_height + pad1);
                    int wend = std::min(wstart + kernel_size0, bot_width + pad0);
                    int pool_size = (hend - hstart) * (wend - wstart);
                    hstart = std::max(hstart, 0);
                    wstart = std::max(wstart, 0);
                    hend = std::min(hend, bot_height);
                    wend = std::min(wend, bot_width);
    
                    //printf("hstart:%d, wstart:%d, hend:%d, wend:%d, pool_size:%d\n", hstart, wstart, hend, wend, pool_size);
                    for (int h = hstart; h < hend; ++h)
                    {
                        for (int w = wstart; w < wend; ++w)
                        {
                            if (problemConfig->PoolMethod == MLO_POOLING_OP_MAX)
                            {
                                size_t bot_index = batch * bot_batch_stride + o * bot_channel_stride +	h * bot_stride + w;
                                if (out_conv_host->data()[bot_index] > res)
                                {
                                    res = out_conv_host->data()[bot_index];
                                }
                            }
                            else if (problemConfig->PoolMethod == MLO_POOLING_OP_AVE)
                            {
                                res += out_conv_host->data()[batch * bot_batch_stride + o * bot_channel_stride + h * bot_stride + w];
                            }
                        }
                    }
    
                    if (problemConfig->PoolMethod == MLO_POOLING_OP_AVE)
                    {
                        res /= pool_size;
                    }
    
                    out_host->mutable_data()[batch * top_batch_stride + o * top_channel_stride + j * top_stride + i] = res;
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
    TensorHf4 bias_host;
    TensorDf4 out;
    TensorHf4 out_host;         //to store the result from device
    TensorHf4 out_golden_host;  //to store the result from cpu calculation
    TensorHf4 out_conv_host;    //for conv_relu_pooling case

    for (int batch_size = 1; batch_size <= 32; batch_size *= 2) {
        // ======================================================================
        // problem config conv12:
        // ======================================================================
        problemConfig = new T_ProblemConfig();
        problemConfig->ConfigName = "Conv12+relu+pooling";
        problemConfig->N = batch_size;  //batch
        problemConfig->W = 224;         //width
        problemConfig->H = 224;         //height
        problemConfig->C = 64;          //channel
        problemConfig->K = 64;          //kernels
        problemConfig->X = 3;           //width of kernel
        problemConfig->Y = 3;           //height of kernel
        problemConfig->PadW = 1;        //width of pad
        problemConfig->PadH = 1;        //height of pad
        //Activation
        problemConfig->NegSlope = 2;
        //Pooling
        problemConfig->PoolMethod = MLO_POOLING_OP_MAX;
        problemConfig->PoolPadW = 0;
        problemConfig->PoolPadH = 0;
        problemConfig->PoolSizeW = 2;
        problemConfig->PoolSizeH = 2;
        problemConfig->PoolStrideW = 2;
        problemConfig->PoolStrideH = 2;

        problemConfigList.push_back(problemConfig);

        // ======================================================================
        // problem config conv22:
        // ======================================================================
        problemConfig = new T_ProblemConfig();
        problemConfig->ConfigName = "Conv22+relu+pooling";
        problemConfig->N = batch_size;  //batch
        problemConfig->W = 112;         //width
        problemConfig->H = 112;         //height
        problemConfig->C = 128;         //channel
        problemConfig->K = 128;         //kernels
        problemConfig->X = 3;           //width of kernel
        problemConfig->Y = 3;           //height of kernel
        problemConfig->PadW = 1;        //width of pad
        problemConfig->PadH = 1;        //height of pad
        //Activation
        problemConfig->NegSlope = 2;
        //Pooling
        problemConfig->PoolMethod = MLO_POOLING_OP_MAX;
        problemConfig->PoolPadW = 0;
        problemConfig->PoolPadH = 0;
        problemConfig->PoolSizeW = 2;
        problemConfig->PoolSizeH = 2;
        problemConfig->PoolStrideW = 2;
        problemConfig->PoolStrideH = 2;

        problemConfigList.push_back(problemConfig);

        // ======================================================================
        // problem config conv33:
        // ======================================================================
        problemConfig = new T_ProblemConfig();
        problemConfig->ConfigName = "Conv33+relu+pooling";
        problemConfig->N = batch_size;  //batch
        problemConfig->W = 56;          //width
        problemConfig->H = 56;          //height
        problemConfig->C = 256;         //channel
        problemConfig->K = 256;         //kernels
        problemConfig->X = 3;           //width of kernel
        problemConfig->Y = 3;           //height of kernel
        problemConfig->PadW = 1;        //width of pad
        problemConfig->PadH = 1;        //height of pad
        //Activation
        problemConfig->NegSlope = 2;
        //pooling
        problemConfig->PoolMethod = MLO_POOLING_OP_MAX;
        problemConfig->PoolPadW = 0;
        problemConfig->PoolPadH = 0;
        problemConfig->PoolSizeW = 2;
        problemConfig->PoolSizeH = 2;
        problemConfig->PoolStrideW = 2;
        problemConfig->PoolStrideH = 2;

        problemConfigList.push_back(problemConfig);

        // ======================================================================
        // problem config conv43:
        // ======================================================================
        problemConfig = new T_ProblemConfig();
        problemConfig->ConfigName = "Conv43+relu+pooling";
        problemConfig->N = batch_size;  //batch
        problemConfig->W = 28;          //width
        problemConfig->H = 28;          //height
        problemConfig->C = 512;         //channel
        problemConfig->K = 512;         //kernels
        problemConfig->X = 3;           //width of kernel
        problemConfig->Y = 3;           //height of kernel
        problemConfig->PadW = 1;        //width of pad
        problemConfig->PadH = 1;        //height of pad
        //Activation
        problemConfig->NegSlope = 2;
        //pooling
        problemConfig->PoolMethod = MLO_POOLING_OP_MAX;
        problemConfig->PoolPadW = 0;
        problemConfig->PoolPadH = 0;
        problemConfig->PoolSizeW = 2;
        problemConfig->PoolSizeH = 2;
        problemConfig->PoolStrideW = 2;
        problemConfig->PoolStrideH = 2;

        problemConfigList.push_back(problemConfig);

        // ======================================================================
        // problem config conv53:
        // ======================================================================
        problemConfig = new T_ProblemConfig();
        problemConfig->ConfigName = "Conv53+relu+pooling";
        problemConfig->N = batch_size;  //batch
        problemConfig->W = 14;          //width
        problemConfig->H = 14;          //height
        problemConfig->C = 512;         //channel
        problemConfig->K = 512;         //kernels
        problemConfig->X = 3;           //width of kernel
        problemConfig->Y = 3;           //height of kernel
        problemConfig->PadW = 1;        //width of pad
        problemConfig->PadH = 1;        //height of pad
        //Activation
        problemConfig->NegSlope = 2;
        //pooling
        problemConfig->PoolMethod = MLO_POOLING_OP_MAX;
        problemConfig->PoolPadW = 0;
        problemConfig->PoolPadH = 0;
        problemConfig->PoolSizeW = 2;
        problemConfig->PoolSizeH = 2;
        problemConfig->PoolStrideW = 2;
        problemConfig->PoolStrideH = 2;

        problemConfigList.push_back(problemConfig);
    }

    Context<AMD> ctx1(0, 1, 1);

    API::stream_t amd_cstream = ctx1.get_compute_stream();

    LOG(INFO) << "Total " << problemConfigList.size() << " problems...";

    //Check the Device's max frequency
    Device<AMD> dev = Env<AMD>::cur_env()[0];
    cl_device_id id = dev.get_device();
    //static void get_param(cl_device_id dev, cl_device_info param_name, T **param_value){
    size_t valueSize;
    cl_uint *num;
    clGetDeviceInfo(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, 0, NULL, &valueSize);
    cl_uint *value = (cl_uint *)malloc(valueSize);
    clGetDeviceInfo(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, valueSize, value, NULL);
    double GPU_CORE_FREQ_HZ = ((double)*value)*1e6; //1500

    //Begin loop
    for (auto p : problemConfigList)
    {
        //TODO: get the problem and solve it...
        LOG(INFO) << "Problem: " << p->ConfigName;
        Env<AMD>::set_tag(p->ConfigName.c_str());
        statics = new T_Statics();

        //allocate weights buffer
        weights.re_alloc({p->K, p->C, p->Y, p->X});
        weights_host.re_alloc({p->K, p->C, p->Y, p->X});

        //allocate input buffer
        in.re_alloc({p->N, p->C, p->H, p->W});
        in_host.re_alloc({p->N, p->C, p->H, p->W});

        bias.re_alloc({1, p->C, 1, 1});
        bias_host.re_alloc({1, p->C, 1, 1});
        //assign default value to weights and input buffer
        //fill_tensor_device_const(weights, 2.f);
        //fill_tensor_device_const(in, 1.f);
        fill_tensor_device_const(bias, 0.f);
        fill_tensor_device_rand(weights, -1.f, 1.f);
        fill_tensor_device_rand(in, -1.f, 1.f);

        std::vector<TensorDf4*> input_v, output_conv_v, output_v;

        input_v.push_back(&in);
        output_v.push_back(&out);

        //wait for device ready
        clFlush(amd_cstream);
        clFinish(amd_cstream);

        //Begin to use device...
        test_conv_active_pooling(p, input_v, output_v,
                 weights, 1, 1, problemConfig->NegSlope,
                 bias, SABER_IMPL, warm_iter, iter, ctx1, statics->score);

        //print_tensor_shape("weights", weights);
        //print_tensor_shape("in", in);
        //print_tensor_shape("out", out);
        //wait for device ready
        clFlush(amd_cstream);
        clFinish(amd_cstream);

        out_host.re_alloc(out.shape());
        out_host.copy_from(out);


        //test result in host side.
	    //NOTE: The width and height of output are parameters for convolution in conv_act_pooling
        TensorDf4 out_conv;
        TensorHf4 out_conv_host;
        output_conv_v.push_back(&out_conv);

        Conv<AMD, AK_FLOAT> conv;
        ConvParam<TensorDf4> conv_param(1, 1, 1,
                               1, 1,
                               1, 1,
                               &weights, &bias);
        conv.compute_output_shape(input_v, output_conv_v, conv_param);

        out_conv_host.re_alloc(out_conv.shape());
        out_golden_host.re_alloc(out.shape());
        in_host.copy_from(in);
        weights_host.copy_from(weights);
        bias_host.copy_from(bias);
        ConvFwd3x3_Host(&out_golden_host, &out_conv_host, in_host, weights_host, p);

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
        LOG(INFO) << "ConfigName = " << p->ConfigName << "cmp result: max_r = " << statics->max_r << " max_d = " << statics->max_d;
        //CHECK_LE(fabs(statics->max_r), 1.0e-6) << "error result";

        statics->score.Calculation = p->W * p->H * p->C * p->N * p->K * p->X * p->Y;

        statics->score.TheoryElapsedMilliSec = statics->score.Calculation / 64 / 64 / GPU_CORE_FREQ_HZ * 1000;
        statics->score.TFlops = statics->score.Calculation / statics->score.ElapsedMilliSec / 1e9;
        statics->score.perf =  statics->score.TheoryElapsedMilliSec / statics->score.ElapsedMilliSec * 100;

        staticsList.push_back(statics);

        free(p);
    }

    LOG(INFO) << "GPU_CORE_FREQ_HZ: " << GPU_CORE_FREQ_HZ << " =================================";
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
    Env<AMD>::pop();
}
int main(int argc, const char** argv){
    anakin::saber::Env<AMD>::env_init();

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

