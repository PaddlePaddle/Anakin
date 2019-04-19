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
void SaberEltwise<X86, OpDtype>::simple_sum(const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseParam<X86>& param, bool with_relu) {
    const int input_num = inputs.size();
    const size_t inner_size = inputs[0]->valid_size();
    OpDataType* target = (OpDataType*) outputs[0]->mutable_data();
    std::vector<const OpDataType*> in_ptrs(input_num);

    for (int i = 0; i < input_num; ++i) {
        in_ptrs[i] = (OpDataType*) inputs[i]->data();
    }

    const OpDataType* coeff = static_cast<const OpDataType*>(param.coeff.data());
    //TODO:can be SIMD to improve cache efficient
#pragma omp parallel for schedule(static)
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
void SaberEltwise<X86, OpDtype>::simple_prod(const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseParam<X86>& param, bool with_relu) {
    const int input_num = inputs.size();
    const size_t inner_size = inputs[0]->valid_size();
    OpDataType* target = (OpDataType*) outputs[0]->mutable_data();
    std::vector<const OpDataType*> in_ptrs(input_num);

    for (int i = 0; i < input_num; ++i) {
        in_ptrs[i] = (OpDataType*) inputs[i]->data();
    }
#pragma omp parallel for schedule(static)
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
void SaberEltwise<X86, OpDtype>::simple_max(const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseParam<X86>& param, bool with_relu) {
    const int input_num = inputs.size();
    volatile const size_t inner_size = inputs[0]->valid_size();
    OpDataType* target = (OpDataType*) outputs[0]->mutable_data();
    std::vector<const OpDataType*> in_ptrs(input_num);

    for (int i = 0; i < input_num; ++i) {
        in_ptrs[i] = (OpDataType*) inputs[i]->data();
    }
#pragma omp parallel for schedule(static)
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
void SaberEltwise<X86, OpDtype>::simple_div(const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EltwiseParam<X86>& param, bool with_relu) {
    const int input_num = inputs.size();
    volatile const size_t inner_size = inputs[0]->valid_size();
    OpDataType* target = (OpDataType*) outputs[0]->mutable_data();
    std::vector<const OpDataType*> in_ptrs(input_num);

    for (int i = 0; i < input_num; ++i) {
        in_ptrs[i] = (OpDataType*) inputs[i]->data();
    }
    if (inputs[1]->valid_size() == inputs[0]->valid_size()) {
#pragma omp parallel for schedule(static)
        for (int inner_id = 0; inner_id < inner_size; ++inner_id) {
            OpDataType tmp = in_ptrs[0][inner_id];

            for (int input_id = 1; input_id < input_num; ++input_id) {
                tmp /= in_ptrs[input_id][inner_id];
            }

            if (with_relu) {
                target[inner_id] = tmp > 0 ? tmp : 0;
            } else {
                target[inner_id] = tmp;
            }
        }
    } else {
        CHECK_EQ(inputs.size(), 2) << "elt with axis not support fusion";
        int outer_num = inputs[0]->count(0, param.axis);
        int mid_num = outputs[0]->valid_size();
        int inner_num = inputs[0]->count(param.axis, inputs[0]->dims()) / mid_num;
        for (int outer_id = 0; outer_id < outer_num; ++outer_id) {
#pragma omp parallel for schedule(static)
            for (int mid_id = 0; mid_id < mid_num; mid_id++) {
                OpDataType div_data = in_ptrs[1][mid_id];
                for (int inner_id = 0; inner_id < inner_num; inner_id++) {
                    int index = (outer_id * mid_num + mid_id) * inner_num + inner_id;
                    OpDataType tmp = in_ptrs[0][index] / div_data;
                    if (with_relu) {
                        target[index] = tmp > 0 ? tmp : 0;
                    } else {
                        target[index] = tmp;
                    }
                }
            }

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
        simple_sum(inputs, outputs, param, _with_relu);
        break;

    case Eltwise_prod:
        simple_prod(inputs, outputs, param, _with_relu);
        break;

    case Eltwise_max:
        simple_max(inputs, outputs, param, _with_relu);
        break;

    case Eltwise_div:
        simple_div(inputs, outputs, param, _with_relu);
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
