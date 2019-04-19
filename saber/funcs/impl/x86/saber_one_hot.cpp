
#include "saber/funcs/impl/x86/saber_one_hot.h"

namespace anakin {

namespace saber {

template <>
SaberStatus SaberOneHot<X86, AK_FLOAT>::create(
        const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        OneHotParam<X86>& param, Context<X86>& ctx) {

    return SaberSuccess;
}

template <>
SaberStatus SaberOneHot<X86, AK_FLOAT>::init(
        const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        OneHotParam<X86>& param, Context<X86>& ctx) {

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberOneHot<X86, AK_FLOAT>::dispatch(
        const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        OneHotParam<X86>& param) {
    memset(outputs[0]->mutable_data(), 0, outputs[0]->valid_size() * outputs[0]->get_dtype_size());

    int depth = param.depth;
    const float* in_ptr = (const float*)inputs[0]->data();
    float* out_ptr = (float*)outputs[0]->mutable_data();
    int dims = inputs[0]->valid_size();
    for (int i = 0; i < dims; ++i) {
        out_ptr[i * depth + (int)in_ptr[i]] = 1.0;
    }
    return SaberSuccess;
}

template class SaberOneHot<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberOneHot, OneHotParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberOneHot, OneHotParam, X86, AK_INT8);

}
}