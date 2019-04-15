
#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_WINOGRAD_AVX2_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_WINOGRAD_AVX2_H
#include "saber/funcs/impl/impl_conv.h"
#include "saber/core/tensor.h"

namespace anakin {
namespace saber {
template<DataType OpDtype>
class SaberConvWinogradAvx2 : public ImplBase <
    X86, OpDtype, ConvEltwiseParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberConvWinogradAvx2() {}

    ~SaberConvWinogradAvx2() {
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
    Tensor<X86> _winor_weights;
    Tensor<X86> _winor_temp;

};
}
}
#endif //ANAKIN_WINOGRAD_H
