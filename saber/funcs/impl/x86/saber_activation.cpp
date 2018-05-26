
#include "saber/funcs/impl/x86/saber_activation.h"
namespace anakin{
namespace saber {

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberActivation<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ActivationParam<OpTensor> &param,
        Context<X86> &ctx)
{

    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    this->_ctx = ctx;

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberActivation<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ActivationParam<OpTensor> &param,
        Context<X86> &ctx)
{
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    this->_ctx = ctx;

    return SaberSuccess;
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberActivation<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ActivationParam<OpTensor> &param)
{
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    // TODO !! need add other types of activation
    if (param.active == Active_relu) {
        for (size_t vc = 0; vc < inputs.size(); vc++) {
            size_t len = inputs[vc]->size();
            float *input_data = inputs[vc]->mutable_data();
            float *output_data = outputs[vc]->mutable_data();

            for (size_t i = 0; i < len; i++) {
                if (*input_data > 0) {
                    *output_data = *input_data;
                } else {
                    *output_data = 0;
                }

                input_data++;
                output_data++;
            }
        }
    }
    return SaberSuccess;
}

template class SaberActivation<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

}
} // namespace anakin
