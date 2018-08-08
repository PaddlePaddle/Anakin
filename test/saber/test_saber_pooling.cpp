#include <vector>
#include <limits>

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
                    
                    dtype result;
                    
                    int dst_ind=ind_n*size_out_n+ind_c*size_out_c+ind_h*out_w+ind_w;
                    for(int kh=sh;kh<eh;++kh){
                        for(int kw=sw;kw<ew;++kw){
                            int src_ind=ind_n*size_in_n+ind_c*size_in_c+kh*in_w+kw;
                            if (kh == sh && kw == sw){
                                result = src_ptr[src_ind];
                            } else {
                                if (param.pooling_type==Pooling_max){
                                    result=result>=src_ptr[src_ind]?result:src_ptr[src_ind];
                                }
                                if (param.pooling_type==Pooling_average_include_padding){
                                    result+=src_ptr[src_ind];
                                }
                                if (param.pooling_type==Pooling_average_exclude_padding){
                                    result+=src_ptr[src_ind];
                                }
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
TestSaberBase<NV,NVHX86,AK_FLOAT,Pooling,PoolingParam> testbase_cu;
    
for(int window_h:{2, 4}){
    for(int window_w:{2, 4}){
        for(int pad_h:{0, 1}){
            for(int pad_w:{0, 1}){
                for(int pooling_type:{Pooling_max, Pooling_average_include_padding, Pooling_average_exclude_padding}){
                    for(int stride_h:{1, 2}){
                        for(int stride_w:{1, 2}){
                            PoolingParam<NV> param(window_h, window_w, pad_h, pad_w, stride_h, stride_w, pooling_type);
                            LOG(INFO)<<"win_h:"<<window_h<<"win_w:"<<window_w \
                            <<"pad_h:"<<pad_h<<"pad_w:"<<pad_w \
                            <<"stride_h:"<<stride_h<<"stride_w:"<<stride_w \
                            <<"pooling_type:"<<pooling_type;
                            
                            for(int in_n:{1, 2}){
                                for(int in_c:{1, 3, 8}){
                                    for(int in_h:{32, 64}){
                                        for(int in_w:{32, 64}){
                                            LOG(INFO)<<"n:"<<in_n<<",in_c:"<<in_c<<",in_h:"<<in_h<<",in_w:"<<in_w;
                                            testbase_cu.set_param(param);//set param
                                            testbase_cu.set_input_shape(Shape({in_n,in_c,in_h,in_w}));//add some input shape
                                            testbase_cu.run_test(pooling_cpu_func<float,NV,NVHX86>);//run test
                                            
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
#ifdef USE_X86_PLACE
TestSaberBase<X86,X86,AK_FLOAT,Pooling,PoolingParam> testbase_cpu;
    
for(int window_h:{2, 4}){
    for(int window_w:{2, 4}){
        for(int pad_h:{0, 1}){
            for(int pad_w:{0, 1}){
                for(int pooling_type:{Pooling_max, Pooling_average_include_padding, Pooling_average_exclude_padding}){
                    for(int stride_h:{1, 2}){
                        for(int stride_w:{1, 2}){
                            PoolingParam<X86> param(window_h, window_w, pad_h, pad_w, stride_h, stride_w, pooling_type);
                            LOG(INFO)<<"win_h:"<<window_h<<"win_w:"<<window_w \
                            <<"pad_h:"<<pad_h<<"pad_w:"<<pad_w \
                            <<"stride_h:"<<stride_h<<"stride_w:"<<stride_w \
                            <<"pooling_type:"<<pooling_type;
                            
                            for(int in_n:{1, 2}){
                                for(int in_c:{1, 3, 8}){
                                    for(int in_h:{32, 64}){
                                        for(int in_w:{32, 64}){
                                            LOG(INFO)<<"n:"<<in_n<<",in_c:"<<in_c<<",in_h:"<<in_h<<",in_w:"<<in_w;
                                            testbase_cpu.set_param(param);//set param
                                            testbase_cpu.set_input_shape(Shape({in_n,in_c,in_h,in_w}));//add some input shape
                                            testbase_cpu.run_test(pooling_cpu_func<float,X86,X86>);//run test
                                            
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
