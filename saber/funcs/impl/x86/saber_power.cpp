#include "saber/funcs/impl/x86/saber_power.h"
#include <cmath>
namespace anakin{
namespace saber {

template <>
SaberStatus SaberPower<X86, AK_FLOAT>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        PowerParam<X86>& param) {
    const float p = param.power;
    const float scale = param.scale;
    const float shift = param.shift;
    
    const float* src_ptr = static_cast<const float*>(inputs[0] -> data());
    float* dst_ptr = static_cast<float*>(outputs[0] -> mutable_data());
    int count = outputs[0] -> valid_size();
    if (inputs[0] -> is_continue_mem() && outputs[0] -> is_continue_mem()){
        if (p == 1){
            for (int i=0; i < count; ++i){
                dst_ptr[i] = src_ptr[i]* scale + shift;
            }
        } else {
            for (int i=0; i < count; ++i){
                dst_ptr[i] = pow(src_ptr[i]*scale + shift, p);
            }
        }
    } else {
        int num_axis = outputs[0] -> dims();
        int in_offset = 0;
        int out_offset = 0;
        int valid_stride = 1;
        const int*  in_strides = static_cast<const int*>(_in_steps.data());
        const int* out_strides = static_cast<const int*>(_out_steps.data());
        const int* valid_shape = static_cast<const int*>(_out_valid_shape.data());
        if (p ==1){
            for (int i=0; i < count; ++i){
                for (int axis_id = num_axis; axis_id >= 0; --axis_id){
                    int id = (i / valid_stride) % valid_shape[axis_id];
                    out_offset += id*out_strides[axis_id];
                    in_offset += id*in_strides[axis_id];
                    valid_stride *= valid_shape[axis_id];
                }
                dst_ptr[out_offset] = src_ptr[in_offset]*scale + shift;
            }
        } else {
            for (int i=0; i < count; ++i){
                for (int axis_id = num_axis; axis_id >= 0; --axis_id){
                    int id = (i / valid_stride) % valid_shape[i];
                    out_offset += id*out_strides[i];
                    in_offset += id*in_strides[i];
                    valid_stride *= valid_shape[i];
                }
                dst_ptr[out_offset] = pow(src_ptr[in_offset]*scale + shift, p);
            }
        }
    }
   return SaberSuccess;
}

template class SaberPower<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberPower, PowerParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberPower, PowerParam, X86, AK_INT8);
}
} // namespace anakin
