
#include "saber/funcs/impl/x86/saber_cumsum.h"
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberCumsum<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        CumsumParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberCumsum<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        CumsumParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberCumsum<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        CumsumParam<X86> &param) {
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;
    CHECK_EQ(inputs.size(), 1) << "cumsum input tensor number must be one";
    CHECK_EQ(outputs.size(), 1) << "cumsum output tensor number must be one";
    OpDataType *input_data = (OpDataType*)inputs[0]->mutable_data();
    OpDataType *output_data = (OpDataType*)outputs[0]->mutable_data();
    auto valid_shape = inputs[0]->valid_shape();
    int axis = param.axis < 0 ? param.axis + inputs[0]->dims() : param.axis;
    int dims = valid_shape.size();
    int pre = inputs[0]->count_valid(0, axis);
    int post = inputs[0]->count_valid(axis+1, dims);
    int idx = 0;
    if (param.reverse == false) {
        if (param.exclusive == true) {
            for (size_t i = 0; i < pre; i++) {
                idx = i * valid_shape[axis] * post;
                memset(output_data + idx, 0, sizeof(OpDataType) * post);
                idx += post;
                for (size_t j = 1; j < valid_shape[axis]; j++) {
                    for (size_t k = 0; k < post; k++) {
                        output_data[idx] = output_data[idx - post] + input_data[idx - post];
                        idx++;
                    }
                }
            }
        } else {
            for (size_t i = 0; i < pre; i++) {
                idx = i * valid_shape[axis] * post;
                memcpy(output_data + idx, input_data + idx, sizeof(OpDataType) * post);
                idx += post;
                for (size_t j = 1; j < valid_shape[axis]; j++) {
                    for (size_t k = 0; k < post; k++) {
                        output_data[idx] = output_data[idx - post] + input_data[idx];
                        idx++;
                    }
                }
            }
        }
    } else {
        if (param.exclusive == true) {
            for (size_t i = 0; i < pre; i++) {
                idx = i * valid_shape[axis] * post;
                memset(output_data + idx, 0, sizeof(OpDataType) * post);
                auto out_tmp = output_data + idx;
                for (size_t j = 1; j < valid_shape[axis]; j++) {
                    for (size_t k = 0; k < post; k++) {
                        out_tmp[j * post + k] = out_tmp[(j - 1) * post + k] 
                            + input_data[idx + (valid_shape[param.axis]  - j) * post + k];
                    }
                }
            } 
        } else {
            for (size_t i = 0; i < pre; i++) {
                idx = i * valid_shape[axis] * post;
                memcpy(output_data + idx, input_data + idx + (valid_shape[axis] - 1) * post, 
                        sizeof(OpDataType) * post);
                auto out_tmp = output_data + idx;
                for (size_t j = 1; j < valid_shape[axis]; j++) {
                    for (size_t k = 0; k < post; k++) {
                        out_tmp[j * post + k] = out_tmp[(j - 1) * post + k] 
                            + input_data[idx + (valid_shape[axis] - 1 - j) * post + k];
                    }
                }
            } 
        }      
    }

    for (size_t i = 0; i < inputs.size(); i++) {
        outputs[i]->set_seq_offset(inputs[i]->get_seq_offset());
    }
    return SaberSuccess;
}

template class SaberCumsum<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberCumsum, CumsumParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberCumsum, CumsumParam, X86, AK_INT8);
}
} // namespace anakin
