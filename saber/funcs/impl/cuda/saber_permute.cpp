#include "saber/funcs/impl/cuda/saber_permute.h"

#include "saber/funcs/impl/impl_permute.h"

namespace anakin {
namespace saber {
template class SaberPermute<NV, AK_FLOAT>;

template <>
SaberStatus SaberPermute<NV, AK_FLOAT>::\
create(const std::vector<Tensor<NV>*>& inputs,
       std::vector<Tensor<NV>*>& outputs,
       PermuteParam<NV>& param, Context<NV>& ctx) {

    Shape order_shape({_num_axes, 1, 1, 1});
    _in_steps.reshape(order_shape);
    _out_steps.reshape(order_shape);
    _out_valid_shape.reshape(order_shape);

    Shape in_stride = inputs[0]->get_stride();
    Shape out_stride = outputs[0]->get_stride();

    cudaMemcpy(_in_steps.mutable_data(), &in_stride[0],
               sizeof(int) * _in_steps.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(_out_steps.mutable_data(), &out_stride[0],
               sizeof(int) * _out_steps.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(_out_valid_shape.mutable_data(), &((outputs[0]->valid_shape())[0]),
               sizeof(int) * _out_valid_shape.size(), cudaMemcpyHostToDevice);
    return SaberSuccess;
}

template <>
SaberStatus SaberPermute<NV, AK_FLOAT>::\
init(const std::vector<Tensor<NV>*>& inputs,
     std::vector<Tensor<NV>*>& outputs,
     PermuteParam<NV>& param, Context<NV>& ctx) {
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
    cudaMemcpy(_permute_order.mutable_data(), &(param.order[0]),
               sizeof(int) * _permute_order.size(), cudaMemcpyHostToDevice);
    return create(inputs, outputs, param, ctx);
}

} //namespace saber

} //namespace anakin
