#include "saber/funcs/impl/x86/saber_permute_power.h"

namespace anakin{
namespace saber{

template class SaberPermutePower<X86, AK_FLOAT>;
    
template <>
SaberStatus SaberPermutePower<X86, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<X86>*>& inputs,
             std::vector<Tensor<X86>*>& outputs,
             PermutePowerParam<X86> &param){
        const float* src_ptr = static_cast<const float*>(inputs[0] -> data());
        float* dst_ptr = static_cast<float*>(outputs[0] -> mutable_data());
        
        float p = param.power_param.power;
        float scale = param.power_param.scale;
        float shift = param.power_param.shift;
        
        if (!_need_permute){
            outputs[0] -> copy_from(*inputs[0]);
        } else {
            std::vector<int> orders = param.permute_param.order;
            int out_size = outputs[0] -> valid_size();
            int num_axes = outputs[0] -> valid_shape().size();
            std::vector<int> new_steps = outputs[0] -> get_stride();
            std::vector<int> old_steps = inputs[0] -> get_stride();
            std::vector<int> new_valid_shape = outputs[0] -> valid_shape();
            if (outputs[0] -> is_continue_mem() && inputs[0] -> is_continue_mem()){
                for (int j=0; j<out_size; ++j){
                    int in_idx = 0;
                    int id = j;
                    for (int i = 0; i < num_axes; ++i) {
                        int order = orders[i];
                        int new_step = new_steps[i];
                        int old_step = old_steps[order];
                        int offset = (id / new_step) * old_step;
                        in_idx += offset;
                        id %= new_step;
                    }
                    if (p == 1){
                        dst_ptr[j] = src_ptr[in_idx]*scale + shift;
                    } else {
                        dst_ptr[j] = pow(src_ptr[in_idx]*scale + shift, p);
                    }
                }
            } else {
                for (int j=0; j<out_size; ++j){
                    int in_idx = 0;
                    int out_idx  = 0;
                    int new_valid_stride = 1;
                    for (int i = num_axes - 1; i >= 0; --i) {
                        int order = orders[i];
                        int new_step = new_steps[i];
                        int old_step = old_steps[order];
                        int id = (j / new_valid_stride) % new_valid_shape[i];
                        in_idx += id * old_step;
                        out_idx += id * new_step;
                        new_valid_stride *= new_valid_shape[i];
                    }
                    if (p == 1){
                        dst_ptr[out_idx] = src_ptr[in_idx]*scale + shift;
                    } else {
                        dst_ptr[out_idx] = pow(src_ptr[in_idx]*scale + shift, p);
                    }
                }
            }
        }//if !need_permute
        
        //if _need_permute is false, do power individually
        if (!_need_permute){
            int out_size = outputs[0] -> valid_size();
            
            if (outputs[0] -> is_continue_mem() && inputs[0] -> is_continue_mem()){
                if (p ==1){
                    for (int i=0; i < out_size; ++i){
                        dst_ptr[i] = dst_ptr[i] * scale + shift;
                    }
                } else {
                    for (int i=0; i < out_size; ++i){
                        dst_ptr[i] = pow(dst_ptr[i] * scale + shift, p);
                    }
                }
            } else {
                int num_axes = outputs[0] -> valid_shape().size();
                std::vector<int> new_steps = outputs[0] -> get_stride();
                std::vector<int> old_steps = inputs[0] -> get_stride();
                std::vector<int> new_valid_shape = outputs[0] -> valid_shape();
                
                if (p ==1){
                    for (int i=0; i<out_size; ++i){
                        int in_idx = 0;
                        int out_idx = 0;
                        int new_valid_stride = 1;
                        for (int axis_id = num_axes; axis_id >=0; --axis_id){
                            int id = (i / new_valid_stride) % new_valid_shape[axis_id];
                            in_idx += id*old_steps[axis_id];
                            out_idx += id*new_steps[axis_id];
                            new_valid_stride *= new_valid_shape[axis_id];
                        }
                        dst_ptr[out_idx] = dst_ptr[in_idx] *scale + shift;
                    }
                } else {
                    for (int i=0; i<out_size; ++i){
                        int in_idx = 0;
                        int out_idx = 0;
                        int new_valid_stride = 1;
                        for (int axis_id = num_axes; axis_id >=0; --axis_id){
                            int id = (i / new_valid_stride) % new_valid_shape[axis_id];
                            in_idx += id*old_steps[axis_id];
                            out_idx += id*new_steps[axis_id];
                            new_valid_stride *= new_valid_shape[axis_id];
                        }
                        dst_ptr[out_idx] = pow(dst_ptr[in_idx] *scale + shift, p);
                    }
                }//if p=1
            }//if is_continue_mem
        }
        return SaberSuccess;
}

    
}
}
