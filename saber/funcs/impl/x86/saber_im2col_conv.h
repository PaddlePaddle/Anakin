
#include "saber/core/tensor.h"
#include "saber_funcs_param.h"
#include "saber_types.h"
#include "saber/funcs/impl/impl_base.h"

namespace anakin {
namespace saber {

template<DataType OpDtype = AK_FLOAT>
class SaberIm2colConv : public ImplBase<
        X86, OpDtype, ConvParam <X86>> {
public:

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
            std::vector<Tensor<X86>*>& outputs,
            ConvParam<X86> &param, Context<X86>&ctx) override;
    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
            std::vector<Tensor<X86>*>& outputs,
            ConvParam<X86> &param, Context<X86>&ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86> *>& inputs,
            std::vector<Tensor < X86>*>& outputs,
            ConvParam <X86> &param) override;

private:
    Tensor<X86> _im2col_tensor;
};

}
}