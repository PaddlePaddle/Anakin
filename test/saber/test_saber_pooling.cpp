#include <vector>

#include "saber/core/context.h"
#include "test/saber/test_saber_base.h"
#include "test_saber_func.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/funcs/pooling.h"

using namespace anakin::saber;

template<typename dtype,typename TargetType_D,typename TargetType_H>
void pooling_cpu_func(const std::vector<Tensor<TargetType_H>*>& input,std::vector<Tensor<TargetType_H>*>& output,PoolingParam<TargetType_D>& param)
{
    const dtype* src_ptr=static_cast<dtype*>(input[0]->data());
    dtype* dst_ptr=static_cast<dtype*>(output[0]->mutable_data());
    
    int in_n=input[0]->num();
    int in_c=input[0]->channel();
    int in_h=input[0]->height();
    int in_w=input[0]->width();
    int size_in_n=in_c*in_h*in_w;
    int size_in_c=in_h*in_w;
    
    int out_h=output[0]->height();
    int out_w=output[0]->width();
    int size_out_n=in_c*out_h*out_w;
    int size_out_c=out_h*out_w;
    
    for(int ind_n=0;ind_n<in_n;++ind_n){
        for(int ind_c=0;ind_c<in_c;++ind_c){
            for(int ind_h=0;ind_h<out_h;++ind_h){
                int sh=ind_h*param.stride_h;
                int eh=sh+param.window_h;
                if(param.pad_h>0)
                {
                    sh=(sh-param.pad_h)<0?0:sh-param.pad_h;
                    eh=(eh-param.pad_h)>in_h?in_h:eh-param.pad_h;
                }
                for(int ind_w=0;ind_w<out_w;++ind_w){
                    int sw=ind_w*param.stride_w;
                    int ew=sw+param.window_w;
                    if(param.pad_w>0){
                        sw=(sw-param.pad_w)<0?0:sw-param.pad_w;
                        ew=(ew-param.pad_w)>in_w?in_w:ew-param.pad_w;
                    }
                    
                    dtype result=0;
                    
                    int dst_ind=ind_n*size_out_n+ind_c*size_out_c+ind_h*out_w+ind_w;
                    for(int kh=sh;kh<eh;++kh){
                        for(int kw=sw;kw<ew;++kw){
                            int src_ind=ind_n*size_in_n+ind_c*size_in_c+kh*in_w+kw;
                            
                            if(param.pooling_type==Pooling_max){
                                result=result>=src_ptr[src_ind]?result:src_ptr[src_ind];
                            }
                            if(param.pooling_type==Pooling_average_include_padding){
                                result+=src_ptr[src_ind];
                            }
                            if(param.pooling_type==Pooling_average_exclude_padding){
                                
                            }
                            
                        }
                    }
                    if(param.pooling_type==Pooling_average_include_padding){
                        result/=param.window_h*param.window_w;
                    }
                    if(param.pooling_type==Pooling_average_exclude_padding){
                        result/=(ew-sw)*(eh-sh);
                    }
                    dst_ptr[dst_ind]=result;
                        
                }
            }
        }
            
    }
}

TEST(TestSaberFunc, test_func_pool)
{
#ifdef USE_CUDA
    TestSaberBase<NV,NVHX86,AK_FLOAT,Pooling,PoolingParam> testbase;
    
    for(int window_h:{2,4}){
        for(int window_w:{2,4}){
            for(int pad_h:{0}){
                for(int pad_w:{0}){
                    for(int pooling_type:{Pooling_max}){
                        for(int stride_h:{2}){
                            for(int stride_w:{2}){
                                PoolingParam<NV> param(window_h, window_w, pad_h, pad_w, stride_h, stride_w, pooling_type);
                                
                                for(int in_n:{1,2}){
                                    for(int in_c:{3,8}){
                                        for(int in_w:{8,64}){
                                            for(int in_h:{8,64}){
                                        testbase.set_param(param);//set param
                                        testbase.set_input_shape(Shape({in_n,in_c,in_h,in_w}));//add some input shape
                                        testbase.run_test(pooling_cpu_func<float,NV,NVHX86>);//run test
                                                
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

/*

using pool_test_params_float = pool_test_params;

template <typename data_t,typename TargetType_D,typename TargetType_H>
bool pooling_test(pool_test_params_float &p) {
    
    test_pool_desc_t pd = p.test_pd;
    
    std::vector<int> src_dims = {pd.mb, pd.c, pd.ih, pd.iw};
    std::vector<int> dst_dims = {pd.mb, pd.c, pd.oh, pd.ow};
    
    Shape shape_input({pd.mb, pd.c / 16, pd.ih, pd.iw/ 16});
    Tensor<X86> input(shape_input);
    fill_tensor_rand(input);
    
    Shape shape_output({pd.mb, pd.c / 16, pd.oh, pd.ow/16});
    Tensor<X86> output(shape_output);
    fill_tensor_rand(output);
    
    Context<TargetType_H> ctx_host;
    std::vector<Tensor<X86>*> inputs(1, &input);
    std::vector<Tensor<X86>*> outputs(1, &output);
    
    PoolingParam<TargetType_H> pool_param(pd.kh, pd.kw, pd.padt, pd.padl, pd.strh, pd.strw, p.aalgorithm);
    Pooling<TargetType_H, AK_FLOAT> pool;
    if (pool.init(inputs, outputs, pool_param, SPECIFY, SABER_IMPL, ctx_host) != SaberSuccess) {
        LOG(ERROR) << "init failed";
        return false;
    }
    
    pool(inputs, outputs, pool_param, ctx_host);
    
    return check_pool_fwd<data_t>(p, inputs, outputs);
}

#define EXPAND_SIZES_2D(mb,ic,ih,iw,oh,ow,kh,kw,padt,padl,strh,strw) \
4, {mb, ic, 1, ih, iw, 1, oh, ow, 1, kh, kw, 0, padt, padl, 1, strh, strw}

TEST(TestSaberFunc, test_func_pool) {
    Env<X86>::env_init();
    
    pool_test_params_float test_param [] = {
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D(2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1)},
        pool_test_params_float{Pooling_average_include_padding,  EXPAND_SIZES_2D( 2, 2048, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 64, 112, 112, 56, 56, 3, 3, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 64, 224, 224, 112, 112, 2, 2, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 128, 112, 112, 56, 56, 2, 2, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 256, 56, 56, 28, 28, 2, 2, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 512, 28, 28, 14, 14, 2, 2, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 6, 512, 14, 14, 7, 7, 2, 2, 0, 0, 2, 2)}
    };
    
    for (size_t i = 0; i < ARRAY_SIZE(test_param); i++) {
        bool ret = pooling_test<float,X86,X86>(test_param[i]);
        if (ret) {
            LOG(INFO) << "Test Passed";
        }
        else {
            LOG(ERROR) << "Test Failed";
        }
    }
}

/*
template <typename data_t,typename TargetType,typename TargetType_H>
bool pooling_test(pool_test_params_float &p) {

    test_pool_desc_t pd = p.test_pd;

    std::vector<int> src_dims = {pd.mb, pd.c, pd.ih, pd.iw};
    std::vector<int> dst_dims = {pd.mb, pd.c, pd.oh, pd.ow};

    Shape shape_input(pd.mb, pd.c / 16, pd.ih, pd.iw, 16);
    Tensor5f_C16 input(shape_input);
    fill_tensor_host_rand(input);

    Shape shape_output(pd.mb, pd.c / 16, pd.oh, pd.ow, 16);
    Tensor5f_C16 output(shape_output);
    fill_tensor_host_rand(output);

    Context<TargetType_H> ctx_host;
    std::vector<Tensor5f_C16*> inputs(1, &input);
    std::vector<Tensor5f_C16*> outputs(1, &output);

    PoolingParam<TargetType_H> pool_param(pd.kh, pd.kw, pd.padt, pd.padl, pd.strh, pd.strw, p.aalgorithm);
    Pooling<TargetType_H, AK_FLOAT> pool;
    if (pool.init(inputs, outputs, pool_param, SPECIFY, SABER_IMPL, ctx_host) != SaberSuccess) {
        LOG(ERROR) << "init failed";
        return false;
    }

    pool(inputs, outputs, pool_param, ctx_host);

    return check_pool_fwd<data_t>(p, inputs, outputs);
}

#define EXPAND_SIZES_2D(mb,ic,ih,iw,oh,ow,kh,kw,padt,padl,strh,strw) \
    4, {mb, ic, 1, ih, iw, 1, oh, ow, 1, kh, kw, 0, padt, padl, 1, strh, strw}

TEST(TestSaberFunc, test_func_pool) {
    Env<X86>::env_init();

    pool_test_params_float test_param [] = {
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D(2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1)},
        pool_test_params_float{Pooling_average_include_padding,  EXPAND_SIZES_2D( 2, 2048, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 64, 112, 112, 56, 56, 3, 3, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 64, 224, 224, 112, 112, 2, 2, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 128, 112, 112, 56, 56, 2, 2, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 256, 56, 56, 28, 28, 2, 2, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 512, 28, 28, 14, 14, 2, 2, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 6, 512, 14, 14, 7, 7, 2, 2, 0, 0, 2, 2)}
    };

    for (size_t i = 0; i < ARRAY_SIZE(test_param); i++) {
        bool ret = pooling_test<float>(test_param[i]);
        if (ret) {
            LOG(INFO) << "Test Passed";
        }
        else {
            LOG(ERROR) << "Test Failed";
        }
    }
}
*/

