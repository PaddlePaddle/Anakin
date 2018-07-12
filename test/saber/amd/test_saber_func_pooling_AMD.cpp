#include "saber/core/context.h"
#include "saber/funcs/pooling.h"
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

typedef struct ProblemConfigType
{
#define MLO_POOLING_OP_MAX 0
#define MLO_POOLING_OP_AVE 1
#define FLT_MAX 3.402823466e+38f

    std::string ConfigName;
    int N;
    int W, H;
    int C, K;
    // pooling config
    int PoolMethod;
    int PoolPadW, PoolPadH;
    int PoolStrideW, PoolStrideH;
    int PoolSizeW, PoolSizeH;

}T_ProblemConfig;

void test_pooling_device(T_ProblemConfig* problemConfig, 
                std::vector<TensorDf4*> &inputs, 
                std::vector<TensorDf4*> &outputs,
                int stride, int pad, TensorDf4 &bias,
                anakin::saber::ImplEnum impl,
                int warm_iter,
                int iter,
                Context<AMD> &ctx1,
                SaberTimer<AMD> &timer) {

    PoolingParam<TensorDf4> param(problemConfig->PoolSizeH, problemConfig->PoolSizeW, problemConfig->PoolPadH, 
                                          problemConfig->PoolPadW, problemConfig->PoolStrideH, problemConfig->PoolStrideW, 
                                          Pooling_max);

    Pooling<AMD, AK_FLOAT> pooling;
    pooling.compute_output_shape(inputs, outputs, param);
    //MUST USE "valid_shape()"
    outputs[0]->re_alloc(outputs[0]->valid_shape());

    print_tensor_shape("in", *inputs[0]);
    print_tensor_shape("out", *outputs[0]);

    SABER_CHECK(pooling.init(inputs, outputs, param, SPECIFY, impl, ctx1));
    
    for(int i =0 ; i < iter; i++)
        pooling(inputs, outputs, param, ctx1);
    clFinish(ctx1.get_compute_stream());

    for(int i =0 ; i < iter; i++) {
        timer.start(ctx1);

        pooling(inputs, outputs, param, ctx1);

        timer.end(ctx1);
    }

    clFlush(ctx1.get_compute_stream());
    clFinish(ctx1.get_compute_stream());

}

void Pooling_Host(TensorHf4* out_host, TensorHf4* in_host, T_ProblemConfig* problemConfig)
{
    LOG(INFO) << "Pooling host";

    int batch_size = problemConfig->N;
    int channel_in = problemConfig->C;
    int feature_out = problemConfig->K;
    int width_in = problemConfig->W;
    int height_in = problemConfig->H;

    int width_out = (width_in + problemConfig->PoolPadW * 2) / problemConfig->PoolSizeW;
    int height_out = (height_in + problemConfig->PoolPadH * 2) / problemConfig->PoolSizeH;

    int stride_n_in = width_in * height_in * channel_in;   // stride for differetn batch of input
    int stride_c_in = width_in * height_in;                // stride for differetn channel in the same batch of input
    int stride_n_out = width_in * height_in * feature_out;// stride for differetn bathc of output
    int stride_k_out = width_in * height_in;  // stride for differetn feature in the same batch of output

    int pool_width_pad = problemConfig->PoolPadW;		
    int pool_heigh_pad = problemConfig->PoolPadH;
    int pool_width_stride = problemConfig->PoolStrideW;	
    int pool_heigh_stride = problemConfig->PoolStrideH;
    int pool_width_kernel = problemConfig->PoolSizeW;
    int pool_heigh_kernel = problemConfig->PoolSizeH;
    int pool_width_in = width_in;
    int pool_heigh_in = height_in;
    int pool_width_out = pool_width_in / pool_width_kernel;
    int pool_heigh_out = pool_heigh_in / pool_heigh_kernel;
    int pool_stride_c_in = stride_k_out;  
    int pool_stride_n_in = stride_n_out;
    int pool_stride_c_out = stride_k_out / (pool_width_kernel * pool_heigh_kernel);
    int pool_stride_n_out = stride_n_out / (pool_width_kernel * pool_heigh_kernel);

    //TensorHf4 in_host;
    //in_host.re_alloc({batch_size, channel_in, width_in, height_in});

    //Fill the data into memory
    //fill_tensor_host_const(in_host, 1.f);

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
    /*
    LOG(INFO)<<"OP_ID("<<problemConfig->PoolMethod<<"), KERNEL_SZ0("<<kernel_size0<<"), KERNEL_SZ1("<<kernel_size1i<<")";
    LOG(INFO)<<"PAD0("<<pad0<<"),PAD1("<<pad1<<"),STRIDE0("<<stride0<<"),stride1("<<stride1<<")";
    LOG(INFO)<<"BOT_WIDTH("<<bot_width<<"),BOT_HEIGHT("<<bot_height<<"),BOT_STRIDE("<<bot_stride<<"),BOT_CHANNEL_STRIDE("<<bot_channel_stride<<"),BOT_BATCH_STRIDE("<<bot_batch_stride<<")";
    */
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
                                if (in_host->data()[bot_index] > res)
                                {
                                    res = in_host->data()[bot_index];
                                }
                            }
                            else if (problemConfig->PoolMethod == MLO_POOLING_OP_AVE)
                            {
                                res += in_host->data()[batch * bot_batch_stride + o * bot_channel_stride + h * bot_stride + w];
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

TEST(TestSaberFuncAMD, test_pooling) {

    std::list<T_ProblemConfig*> problemConfigList;
    T_ProblemConfig * problemConfig;

    TensorDf4 in;
    TensorHf4 in_host;
    TensorDf4 bias;
    TensorHf4 bias_host;
    TensorDf4 out;
    TensorHf4 out_host;        //to store the result from device
    TensorHf4 out_golden_host; //to store the result from cpu calculation

    // ======================================================================
    // problem config Pooling12:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Pooling12";
    problemConfig->N = 1;    //batch
    problemConfig->W = 224;  //width
    problemConfig->H = 224;  //height
    problemConfig->C = 64;    //channel
    problemConfig->K = 64;   //kernels
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
    // problem config Pooling22:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Pooling22";
    problemConfig->N = 1;    //batch
    problemConfig->W = 112;  //width
    problemConfig->H = 112;  //height
    problemConfig->C = 128;    //channel
    problemConfig->K = 128;   //kernels
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
    // problem config Pooling33:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Pooling33";
    problemConfig->N = 1;    //batch
    problemConfig->W = 56;  //width
    problemConfig->H = 56;  //height
    problemConfig->C = 256;    //channel
    problemConfig->K = 256;   //kernels
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
    // problem config Pooling43:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Pooling43";
    problemConfig->N = 1;    //batch
    problemConfig->W = 28;  //width
    problemConfig->H = 28;  //height
    problemConfig->C = 512;    //channel
    problemConfig->K = 512;   //kernels
    //pooling
    problemConfig->PoolMethod = MLO_POOLING_OP_MAX;
    problemConfig->PoolPadW = 0;
    problemConfig->PoolPadH = 0;
    problemConfig->PoolSizeW = 2;
    problemConfig->PoolSizeH = 2;
    problemConfig->PoolStrideW = 2;
    problemConfig->PoolStrideH = 2;

    problemConfigList.push_back(problemConfig);
/*
    // ======================================================================
    // problem config conv53:
    // ======================================================================
    problemConfig = new T_ProblemConfig();
    problemConfig->ConfigName = "Conv53+relu+pooling";
    problemConfig->N = 1;    //batch
    problemConfig->W = 14;  //width
    problemConfig->H = 14;  //height
    problemConfig->C = 512;    //channel
    problemConfig->K = 512;   //kernels
    problemConfig->X = 3;    //width of kernel
    problemConfig->Y = 3;    //height of kernel
    problemConfig->PadW = 1; //width of pad
    problemConfig->PadH = 1; //height of pad
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
*/
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
        //fill_tensor_device_const(in, 1.f);
        fill_tensor_device_rand(in, 1, 10);
        in_host.copy_from(in);

        std::vector<TensorDf4*> input_v, output_v;

        input_v.push_back(&in);
        output_v.push_back(&out);

        //wait for device ready
        clFlush(amd_cstream);
        clFinish(amd_cstream);

            //Begin to use device...
        test_pooling_device(p, input_v, output_v,
                 1, 1,
                 bias, SABER_IMPL, warm_iter, iter, ctx1, t_device);

        //print_tensor_shape("in", in);
        //print_tensor_shape("out", out);
        //wait for device ready
        clFlush(amd_cstream);
        clFinish(amd_cstream);

        out_host.re_alloc(out.shape());
        out_host.copy_from(out);


        //test result in host side.
        out_golden_host.re_alloc(out.shape());

        t_host.start(ctx1);
        Pooling_Host(&out_golden_host, &in_host, p);
        t_host.end(ctx1);

        //LOG(INFO) << "PRINT DEVICE TENSOR: img";
        //print_tensor_device(img);
        //sleep(2);

        //LOG(INFO) << "PRINT DEVICE TENSOR: out";
        //print_tensor_device(out);
        //sleep(2);

        //LOG(INFO) << "PRINT HOST TENSOR: out";
        //print_tensor_host(out_golden_host);
        //sleep(2);

        double max_r, max_d;
        tensor_cmp_host(out_host.data(), out_golden_host.data(), out_host.size(), max_r, max_d);
        LOG(INFO) << "ConfigName = " << p->ConfigName << "cmp result: max_r = " << max_r << " max_d = " << max_d;
        LOG(INFO) << "cmp elapse time: device( best : " <<t_device.get_best_ms() <<" ms , average : " << t_device.get_average_ms() << " ms) : host(" << t_host.get_average_ms() << " ms)";
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

