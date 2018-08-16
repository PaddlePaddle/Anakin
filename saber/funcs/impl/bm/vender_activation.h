#ifndef ANAKIN_SABER_FUNCS_BMDNN_ACT_H
#define ANAKIN_SABER_FUNCS_BMDNN_ACT_H
#include "saber/funcs/impl/impl_activation.h"
namespace anakin {

namespace saber {

template <DataType OpDtype>
class VenderActivation<BM, OpDtype> : \
    public ImplBase <
    BM,
    OpDtype,
    ActivationParam<BM > > {
public:
    typedef Tensor<BM> OpTensor;
    typedef typename DataTraitBase<BM>::PtrDtype DataPtr;

    VenderActivation(): _handle(NULL), _active_type(Active_relu) {}

    ~VenderActivation() {}

    virtual SaberStatus init(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             ActivationParam<BM>& param, Context<BM>& ctx) {
        // not sure
        _handle = get_bm_handle();
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<OpTensor*>& inputs,
                               std::vector<OpTensor*>& outputs,
                               ActivationParam<BM>& param, Context<BM>& ctx) {
        // not sure
        return SaberSuccess;
    }

    //call bmdnn activation funcs here
    virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                                 std::vector<OpTensor*>& outputs,
                                 ActivationParam<BM>& param) {
        const DataPtr in_data = (inputs[0]->data());
        DataPtr out_data = (outputs[0]->mutable_data());
        int input_dim = inputs[0]->channel() * inputs[0]->height() * inputs[0]->width();
        int input_n = inputs[0]->num();

        _active_type = param.active;

        switch (_active_type) {
        case Active_relu:
            BMDNN_CHECK(bmdnn_relu_forward(_handle, in_data, 0.0, input_n, input_dim, out_data));
            break;

        case Active_sigmoid:
            BMDNN_CHECK(bmdnn_sigmoid_forward(_handle, in_data, input_n, input_dim, out_data));
            break;

        case Active_tanh:
            BMDNN_CHECK(bmdnn_tanh_forward(_handle, in_data, input_n, input_dim, out_data));
            break;
        default:LOG(INFO)<<"type not support now";
                return SaberUnImplError;
        }

        return SaberSuccess;
    }

private:
    bm_handle_t _handle;
    ActiveType _active_type;
};

template class VenderActivation<BM, AK_FLOAT>;
} // namespace saber

} // namespace anakin
#endif //ANAKIN_SABER_FUNCS_BMDNN_ACT_H
