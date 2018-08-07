#ifndef ANAKIN_SABER_FUNCS_IMPL_BMDNN_CONV2D_H
#define ANAKIN_SABER_FUNCS_IMPL_BMDNN_CONV2D_H

#include "saber/funcs/impl/impl_conv.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class VenderConv2D<BM, OpDtype> : public ImplBase<
        BM, OpDtype, ConvParam<BM> > {
            
public:
    VenderConv2D(): _handle(NULL) {}
    ~VenderConv2D() {}

    virtual SaberStatus init(const std::vector<Tensor<BM> *>& inputs,
                             std::vector<Tensor<BM> *>& outputs,
                             ConvParam<BM>& param, Context<BM>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<BM> *>& inputs,
                               std::vector<Tensor<BM> *>& outputs,
                               ConvParam<BM>& param, Context<BM>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<BM>*>& inputs,
                                 std::vector<Tensor<BM>*>& outputs,
                                 ConvParam<BM>& param);

private:
    bm_handle_t _handle;
};

}
}
#endif //ANAKIN_SABER_FUNCS_BMDNN_CONV2D_H
