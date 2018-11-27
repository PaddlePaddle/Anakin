#include <vector>
#include <cmath>

#include "saber/core/context.h"
#include "saber/funcs/pooling_with_index.h"
#include "test/saber/test_saber_func.h"
#include "test/saber/test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/core/data_traits.h"

using namespace anakin::saber;

template <typename dtype,typename TargetType_D,typename TargetType_H>
void pooling_with_index_cpu_func(const std::vector<Tensor<TargetType_H>*>& input,
                                 std::vector<Tensor<TargetType_H>*>& output, PoolingParam<TargetType_D>& param) {
    const dtype* src_ptr = static_cast<const dtype*>(input[0] -> data());
    dtype* out_data_ptr = static_cast<dtype*>(output[0] -> mutable_data());
    dtype* out_index_ptr = static_cast<dtype*>(output[1] -> mutable_data());
    
    int in_n = input[0] -> num();
    int in_c = input[0] -> channel();
    int in_h=input[0] -> height();
    int in_w=input[0] -> width();
    int size_in_n = in_c*in_h*in_w;
    int size_in_c = in_h*in_w;
    
    int out_h = output[0] -> height();
    int out_w = output[0] -> width();
    int size_out_n = in_c*out_h*out_w;
    int size_out_c = out_h*out_w;
    
    for(int ind_n = 0; ind_n < in_n; ++ind_n){
        for(int ind_c=0; ind_c < in_c; ++ind_c){
            for(int ind_h=0; ind_h<out_h; ++ind_h){
                int sh=ind_h*param.stride_h;
                int eh=sh+param.window_h;
                if(param.pad_h > 0)
                {
                    sh=(sh - param.pad_h) < 0? 0 : sh-param.pad_h;
                    eh=(eh - param.pad_h)>in_h? in_h : eh-param.pad_h;
                }
                for(int ind_w=0; ind_w < out_w; ++ind_w){
                    int sw = ind_w*param.stride_w;
                    int ew = sw + param.window_w;
                    if(param.pad_w > 0){
                        sw = (sw - param.pad_w) < 0? 0 : sw-param.pad_w;
                        ew = (ew - param.pad_w) > in_w?in_w : ew-param.pad_w;
                    }
                    
                    dtype result;
                    dtype index;
                    
                    int dst_ind = ind_n*size_out_n + ind_c*size_out_c + ind_h*out_w + ind_w;
                    for(int kh = sh; kh<eh; ++kh){
                        for(int kw = sw; kw < ew; ++kw){
                            int src_ind = ind_n*size_in_n + ind_c*size_in_c + kh*in_w + kw;
                            if (kh == sh && kw == sw){
                                result = src_ptr[src_ind];
                                index = kh*in_w + kw;
                            } else {
                                    index = result >= src_ptr[src_ind]? index : kh*in_w + kw;
                                    result = result >= src_ptr[src_ind]? result : src_ptr[src_ind];
                            }
                            
                        }
                    }
                    out_data_ptr[dst_ind] = result;
                    out_index_ptr[dst_ind] = index;
                }
            }
        }
        
    }
}

template <typename TargetType_D, typename TargetType_H, DataType OpDtype>
void test_pooling_with_index(){
    
    typedef typename DataTrait<TargetType_D, OpDtype> :: Dtype dtype;
    //Init the test_base
    TestSaberBase<TargetType_D, TargetType_H, OpDtype, PoolingWithIndex, PoolingParam> testbase(1, 2);
    for(int window_h:{2, 4}){
        for(int window_w:{2, 4}){
            for(int pad_h:{0, 1}){
                for(int pad_w:{0, 1}){
                        for(int stride_h:{1, 2}){
                            for(int stride_w:{1, 2}){
                                PoolingParam<TargetType_D> param(window_h, window_w, pad_h, pad_w, stride_h, stride_w, Pooling_max);
                                LOG(INFO)<<"win_h:"<<window_h<<"win_w:"<<window_w \
                                <<"pad_h:"<<pad_h<<"pad_w:"<<pad_w \
                                <<"stride_h:"<<stride_h<<"stride_w:"<<stride_w;
                                
                                for(int in_n:{1, 2}){
                                    for(int in_c:{1, 3, 8}){
                                        for(int in_h:{8, 64}){
                                            for(int in_w:{8, 64}){
                                                LOG(INFO)<<"n:"<<in_n<<",in_c:"<<in_c<<",in_h:"<<in_h<<",in_w:"<<in_w;
                                                testbase.set_param(param);//set param
                                                testbase.set_input_shape(Shape({in_n,in_c,in_h,in_w}));//add some input shape
                                                testbase.run_test(pooling_with_index_cpu_func<dtype, TargetType_D,
                                                                  TargetType_H>);//run test
                                                
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

TEST(TestSaberFunc, test_func_power) {
#ifdef USE_CUDA
    test_pooling_with_index<NV, NVHX86, AK_FLOAT>();
#endif
#ifdef USE_X86_PLACE
    test_pooling_with_index<X86, X86, AK_FLOAT>();
#endif

}



int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    
    return 0;
}
