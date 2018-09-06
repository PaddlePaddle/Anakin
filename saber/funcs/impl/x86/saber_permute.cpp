#include "saber/funcs/impl/x86/saber_permute.h"

namespace anakin{
namespace saber{
template class SaberPermute<X86, AK_FLOAT>;

template <>
SaberStatus SaberPermute<X86, AK_FLOAT>::\
    create(const std::vector<Tensor<X86>*>& inputs,
           std::vector<Tensor<X86>*>& outputs,
           PermuteParam<X86> &param, Context<X86> &ctx) {
        
        Shape order_shape({_num_axes, 1, 1, 1});
        _in_steps.reshape(order_shape);
        _out_steps.reshape(order_shape);
        _out_valid_shape.reshape(order_shape);
        
        Shape in_stride = inputs[0]->get_stride();
        Shape out_stride = outputs[0]->get_stride();
        
        memcpy(_in_steps.mutable_data(), &in_stride[0], sizeof(int) * _in_steps.size());
        memcpy(_out_steps.mutable_data(), &out_stride[0], sizeof(int) * _out_steps.size());
        memcpy(_out_valid_shape.mutable_data(), &((outputs[0]->valid_shape())[0]), sizeof(int) * _out_valid_shape.size());
        return SaberSuccess;
}

template <>
SaberStatus SaberPermute<X86, AK_FLOAT>::\
    init(const std::vector<Tensor<X86>*>& inputs,
         std::vector<Tensor<X86>*>& outputs,
         PermuteParam<X86> &param, Context<X86> &ctx) {
        this->_ctx = &ctx;
        _num_axes = inputs[0]->valid_shape().size();
        for (int i = 0; i < _num_axes; i++) {
            if (std::find(_order_dims.begin(), _order_dims.end(),
                          param.order[i]) == _order_dims.end()) {
                _order_dims.push_back(param.order[i]);
            }
        }

        CHECK_EQ(_num_axes, _order_dims.size());

        // set _need_permute
        _need_permute = false;
        for (int i = 0; i < _num_axes; ++i) {
            if (param.order[i] != i) {
                _need_permute = true;
                break;
            }
        }
        Shape order_shape({_num_axes, 1, 1, 1});
        _permute_order.reshape(order_shape);
        memcpy(_permute_order.mutable_data(), &(param.order[0]), sizeof(int) * _permute_order.size());
        return create(inputs, outputs, param, ctx);
}
    
template <>
SaberStatus SaberPermute<X86, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<X86>*>& inputs,
                std::vector<Tensor<X86>*>& outputs,
                PermuteParam<X86> &param){
        if (!_need_permute){
            outputs[0] -> copy_from(*inputs[0]);
            return SaberSuccess;
        }
        const float* src_ptr = static_cast<const float*>(inputs[0] -> data());
        float* dst_ptr = static_cast<float*>(outputs[0] -> mutable_data());
        std::vector<int> orders = param.order;
        int out_size = outputs[0] -> valid_size();
        int num_axes = inputs[0] -> valid_shape().size();
        std::vector<int> new_steps = outputs[0] -> get_stride();
        std::vector<int> old_steps = inputs[0] -> get_stride();
        std::vector<int> new_valid_shape = outputs[0] -> valid_shape();
        if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()){
            for (int j=0; j<out_size; ++j){
                int in_idx = 0;
                int id = j;
                for (int i = 0; i < num_axes; ++i) {
                    int order = orders[i];
                    int new_step = new_steps[i];
                    int old_step = old_steps[order];
                    int offset = (id / new_step) *old_step;
                    in_idx += offset;
                    id %= new_step;
                }
                dst_ptr[j] = src_ptr[in_idx];
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
                dst_ptr[out_idx] = src_ptr[in_idx];
            }
        }
        return SaberSuccess;
        
}

    
} //namespace saber

} //namespace anakin
