#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/x86/saber_conv_act.h"
#include "saber/funcs/impl/x86/jit_call_conf.h"
#include "saber/funcs/impl/x86/jit_uni_dw_convolution.h"
#include "saber/funcs/impl/x86/jit_avx512_conv1x1_act.h"
#include "saber/funcs/impl/x86/jit_avx512_conv_act.h"
#include "saber/funcs/impl/x86/jit_avx2_conv_act.h"

namespace anakin {
namespace saber {

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberConv2DAct<X86, OpDtype, inDtype, outDtype,
    LayOutType_op, LayOutType_in, LayOutType_out>::init(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    ConvActiveParam<OpTensor> &param,
    Context<X86> &ctx)
{
    SaberStatus ret = SaberUnImplError;

    ConvParam<OpTensor> *conv_param = &(param.conv_param);
    const OpTensor *weight = conv_param->weight();
    Shape weight_shape(weight->shape());

    // go to different engines per different input parameters
    if (conv_param->group == weight_shape[0] && conv_param->group == weight_shape[1]) {
        // depth-wise convolution
        if (this->impl) {
            delete this->impl;
        }
        this->impl = new JitUniDWConvolution<OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>;
        ret = this->impl->init(inputs, outputs, param, ctx);
        if (ret == SaberSuccess) {
//            LOG(INFO) << "++++++++++++JitUniDWConvolution";
            return ret;
        }
    } else if (weight_shape[2] == 1 && weight_shape[3] == 1) {
        // 1x1 convolution+act
        if (this->impl) {
            delete this->impl;
        }
        this->impl = new JitAvx512Conv1x1Act<OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>;
        ret = this->impl->init(inputs, outputs, param, ctx);
        if (ret == SaberSuccess) {
//            LOG(INFO) << "++++++++++++JitAvx512Conv1x1Act";
            return ret;
        }
    } else if (std::is_same<LayOutType_out, NCHW_C16>::value) {
        if (this->impl) {
            delete this->impl;
        }
        this->impl = new JitAvx512ConvAct<OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>;
        ret = this->impl->init(inputs, outputs, param, ctx);
        if (ret == SaberSuccess) {
//            LOG(INFO) << "++++++++++++JitAvx512ConvAct";
            return ret;
        }
    } else if (std::is_same<LayOutType_out, NCHW_C8>::value) {
        if (this->impl) {
            delete this->impl;
        }
        this->impl = new JitAvx2ConvAct<OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>;
        ret = this->impl->init(inputs, outputs, param, ctx);
        if (ret == SaberSuccess) {
//            LOG(INFO) << "++++++++++++JitAvx2ConvAct";
            return ret;
        }
    }
    return SaberUnImplError;
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberConv2DAct<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ConvActiveParam<OpTensor> &param,
        Context<X86> &ctx)
{
    SaberStatus ret = SaberSuccess;
    if (!this->impl) {
        LOG(ERROR) << "impl is NULL";
        return SaberNotInitialized;
    }
    ret = this->impl->create(inputs, outputs, param, ctx);
    return ret;
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberConv2DAct<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ConvActiveParam<OpTensor> &param)
{
    SaberStatus ret = SaberSuccess;
    if (!this->impl) {
        LOG(ERROR) << "impl is NULL";
        return SaberNotInitialized;
    }
    ret = this->impl->dispatch(inputs, outputs, param);
    return ret;
}
template class SaberConv2DAct<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW_C16, NCHW_C16>;
template class SaberConv2DAct<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW_C16>;
template class SaberConv2DAct<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW_C8>;
//template class SaberConvAct<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}
} // namespace anakin
