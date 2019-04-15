#include "saber/funcs/impl/x86/winograd.h"
#include "saber/funcs/impl/x86/winograd_float.h"
#include "saber/funcs/impl/x86/winograd_avx2.h"
//#include "saber/funcs/impl/x86/winograd_avx.h"
//#include "saber/funcs/impl/x86/winograd_avx2_nchwc8.h"
namespace anakin {
namespace saber {

template <>
SaberStatus SaberConvWinograd<AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param, Context<X86>& ctx) {

    return _impl->create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConvWinograd<AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param, Context<X86>& ctx) {
    LayoutType input_layout = inputs[0]->get_layout();
    LayoutType out_layout = outputs[0]->get_layout();

    //    if(input_layout==Layout_NCHW_C8R&&out_layout==Layout_NCHW_C8R){
    //        this->_impl = new SaberConvWinogradAvx2Nchwc8<AK_FLOAT>;
    //    }else
    if (input_layout == Layout_NCHW && out_layout == Layout_NCHW) {
#if defined(__AVX2__) and defined(__FMA__)
        this->_impl = new SaberConvWinogradAvx2<AK_FLOAT>;
#else
        this->_impl = new SaberConvWinogradFloat<AK_FLOAT>;
#endif
    } else {
        LOG(FATAL) << "winograd conv not support this layout";
    }

    return _impl->init(inputs, outputs, param, ctx);

}

template <>
SaberStatus SaberConvWinograd<AK_FLOAT>::dispatch(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param) {
    return _impl->dispatch(inputs, outputs, param);

}

}
}