
#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_WINOGRAD_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_WINOGRAD_H
#include "saber/funcs/impl/impl_conv.h"
#include "saber/core/tensor.h"

namespace anakin {
namespace saber {
template<DataType OpDtype>
class SaberConvWinograd : public ImplBase <
    X86, OpDtype, ConvEltwiseParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;
    typedef ImplBase<X86, OpDtype, ConvEltwiseParam<X86> > Impl_t;

    SaberConvWinograd() {}

    ~SaberConvWinograd() {
        if (_impl!= nullptr){
            delete _impl;
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                             std::vector<Tensor<X86> *>& outputs,
                             ConvEltwiseParam<X86>& param, Context<X86>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                               std::vector<Tensor<X86> *>& outputs,
                               ConvEltwiseParam<X86>& param, Context<X86>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<X86> *>& inputs,
                                 std::vector<Tensor<X86> *>& outputs,
                                 ConvEltwiseParam<X86>& param);

private:
    Impl_t *_impl;

};
}
}
#endif //ANAKIN_WINOGRAD_H
