#include "saber/funcs/impl/x86/saber_eltwise.h"
#include "saber/funcs/impl/x86/x86_utils.h"


namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberEltwise<X86, OpDtype>::init(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    EltwiseParam<X86>& param,
    Context<X86>& ctx) {
    // get context
    this->_ctx = &ctx;
    _with_relu = param.has_eltwise && param.activation_param.active == Active_relu;
    _other_activation = param.has_eltwise && param.activation_param.active != Active_relu
                        && param.activation_param.active != Active_unknow;

    if (_other_activation) {
        LOG(FATAL) << "not support other_activation";
    }

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberEltwise<X86, OpDtype>::create(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    EltwiseParam<X86>& param,
    Context<X86>& ctx) {
    this->_param = &param;
    this->_ctx = &ctx;

    return SaberSuccess;
}
template <DataType OpDtype>
template <bool with_relu>
void SaberEltwise<X86, OpDtype>::simple_sum(const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseParam<X86>& param) {
    const int input_num = inputs.size();
    const size_t inner_size = inputs[0]->valid_size();
    OpDataType* target = (OpDataType*) outputs[0]->mutable_data();
    std::vector<const OpDataType*> in_ptrs(input_num);

    for (int i = 0; i < input_num; ++i) {
        in_ptrs[i] = (OpDataType*) inputs[i]->data();
    }

    const OpDataType* coeff = static_cast<const OpDataType*>(param.coeff.data());

    //TODO:can be SIMD to improve cache efficient
    for (int inner_id = 0; inner_id < inner_size; ++inner_id) {
        OpDataType tmp = coeff[0] * in_ptrs[0][inner_id];

        for (int input_id = 1; input_id < input_num; ++input_id) {
            tmp += coeff[input_id] * in_ptrs[input_id][inner_id];
        }

        if (with_relu) {
            target[inner_id] = tmp > 0 ? tmp : 0;
        } else {
            target[inner_id] = tmp;
        }

    }
}
template <DataType OpDtype>
template <bool with_relu>
void SaberEltwise<X86, OpDtype>::simple_prod(const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseParam<X86>& param) {
    const int input_num = inputs.size();
    const size_t inner_size = inputs[0]->valid_size();
    OpDataType* target = (OpDataType*) outputs[0]->mutable_data();
    std::vector<const OpDataType*> in_ptrs(input_num);

    for (int i = 0; i < input_num; ++i) {
        in_ptrs[i] = (OpDataType*) inputs[i]->data();
    }

    for (int inner_id = 0; inner_id < inner_size; ++inner_id) {
        OpDataType tmp = in_ptrs[0][inner_id];

        for (int input_id = 1; input_id < input_num; ++input_id) {
            tmp *= in_ptrs[input_id][inner_id];
        }

        if (with_relu) {
            target[inner_id] = tmp > 0 ? tmp : 0;
        } else {
            target[inner_id] = tmp;
        }
    }
}

template <DataType OpDtype>
template <bool with_relu>
void SaberEltwise<X86, OpDtype>::simple_max(const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseParam<X86>& param) {
    const int input_num = inputs.size();
    volatile const size_t inner_size = inputs[0]->valid_size();
    OpDataType* target = (OpDataType*) outputs[0]->mutable_data();
    std::vector<const OpDataType*> in_ptrs(input_num);

    for (int i = 0; i < input_num; ++i) {
        in_ptrs[i] = (OpDataType*) inputs[i]->data();
    }

    for (int inner_id = 0; inner_id < inner_size; ++inner_id) {
        OpDataType tmp = in_ptrs[0][inner_id];

        for (int input_id = 1; input_id < input_num; ++input_id) {
            tmp = tmp >= in_ptrs[input_id][inner_id] ? tmp : in_ptrs[input_id][inner_id];
        }

        if (with_relu) {
            target[inner_id] = tmp > 0 ? tmp : 0;
        } else {
            target[inner_id] = tmp;
        }
    }
}


template <DataType OpDtype>
SaberStatus SaberEltwise<X86, OpDtype>::dispatch(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    EltwiseParam<X86>& param) {
    CHECK_EQ(outputs.size(), (size_t)1);

    switch (param.operation) {
    case Eltwise_sum:
        if (_with_relu) {
            simple_sum<true>(inputs, outputs, param);
        } else {
            simple_sum<false>(inputs, outputs, param);
        }

        break;

    case Eltwise_prod:
        if (_with_relu) {
            simple_prod<true>(inputs, outputs, param);
        } else {
            simple_prod<false>(inputs, outputs, param);
        }

        break;

    case Eltwise_max:
        if (_with_relu) {
            simple_max<true>(inputs, outputs, param);
        } else {
            simple_max<false>(inputs, outputs, param);
        }

        break;

    default:
        LOG(FATAL) << "unknown elementwise operation. ";
    }

    return SaberSuccess;

}

template class SaberEltwise<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberEltwise, EltwiseParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberEltwise, EltwiseParam, X86, AK_INT8);
}
} // namespace anakin
