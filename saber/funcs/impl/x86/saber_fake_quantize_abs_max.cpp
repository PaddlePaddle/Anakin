#include "saber/funcs/impl/x86/saber_fake_quantize_abs_max.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include <cmath>
namespace anakin {
namespace saber {

/**
 *  @brief  formula: 
 *                   scale = max(abs(X))
 *                   range = 2^{bit_length - 1} - 1
 *                   Out = round(X/scale * range)
 * 
 * 
 */
template <DataType OpDtype>
SaberStatus SaberFakeQuantizeAbsMax<X86, OpDtype>::dispatch(\
    const std::vector<Tensor<X86> *>& inputs, \
    std::vector<Tensor<X86> *>& outputs, \
    FakeQuantizeAbsMaxParam<X86>& param) {
    const OpDataType* src = (const OpDataType*)inputs[0]->data();
    auto dst = outputs[0]->mutable_data();
    int valid_size = inputs[0]->valid_size();
    auto max_data = 0.f;
    for (int i = 0; i < valid_size; i++) {
        auto abs_data = src[i] > 0.f ? src[i] : -src[i];
        max_data = abs_data > max_data ? abs_data : max_data; 
    }
    auto range = (1 << (param.bit_length - 1)) - 1;
    auto scale = 1.f / max_data * range;
    if (param.bit_length == 8) {
        char* dst_tmp = (char*)dst;
        for (int i = 0; i < valid_size; i++) {
            dst_tmp[i] = round(src[i] * scale);
            //LOG(INFO) << i << " " << int(dst_tmp[i]);
        }
    } else if (param.bit_length == 16) {
        int16_t* dst_tmp = (int16_t*)dst;
        for (int i = 0; i < valid_size; i++) {
            dst_tmp[i] = round(src[i] * scale);
        }
    } else {
        LOG(FATAL) <<"other bit length has not been supported";
    }  

    return SaberSuccess;
}

template class SaberFakeQuantizeAbsMax<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberFakeQuantizeAbsMax, FakeQuantizeAbsMaxParam, X86, AK_INT16);
DEFINE_OP_TEMPLATE(SaberFakeQuantizeAbsMax, FakeQuantizeAbsMaxParam, X86, AK_INT8);
}
}
