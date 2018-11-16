#include "saber/funcs/impl/x86/saber_pooling_with_index.h"

namespace anakin{
namespace saber{
    
template class SaberPoolingWithIndex<X86, AK_FLOAT>;
    
template <>
SaberStatus SaberPoolingWithIndex<X86, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<X86>*>& inputs,
                    std::vector<Tensor<X86>*>& outputs,
                    PoolingParam<X86> &param){
        if (!outputs[0] -> is_continue_mem() || !inputs[0] -> is_continue_mem()){
            LOG(ERROR) <<"pooling_with_index only support continue memory";
            return SaberUnImplError;
        }
        const float* src_ptr = static_cast<const float*>(inputs[0] -> data());
        float* out_data_ptr = static_cast<float*>(outputs[0] -> mutable_data());
        float* out_index_ptr = static_cast<float*>(outputs[1] -> mutable_data());
        
        int in_n = inputs[0] -> num();
        int in_c = inputs[0] -> channel();
        int in_h=inputs[0] -> height();
        int in_w=inputs[0] -> width();
        int size_in_n = in_c*in_h*in_w;
        int size_in_c = in_h*in_w;
        
        int out_h = outputs[0] -> height();
        int out_w = outputs[0] -> width();
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
                        
                        float result = 0.f;
                        float index = 0.f;
                        
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
        return SaberSuccess;
            
}

    
}
}
