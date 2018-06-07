
#include "saber/funcs/impl/x86/saber_scale.h"
namespace anakin{
namespace saber {

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberScale<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ScaleParam<OpTensor> &param,
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
SaberStatus SaberScale<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ScaleParam<OpTensor> &param,
        Context<X86> &ctx)
{
    return SaberSuccess;
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberScale<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ScaleParam<OpTensor> &param)
{
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    auto in_data = inputs[0]->data();
    auto out_data = outputs[0]->mutable_data();
    DataType_op* scale_data = (inputs.size() > 1) ? inputs[1]->data() : &(param.scale_w[0]);
    DataType_op* bias_data = param.bias_term ? &(param.scale_b[0]) : NULL;

    const int count = inputs[0]->valid_size();
    int axis = (param.num_axes == 0) ? 0 : param.axis;
    int num_axes = param.num_axes >=0 ? param.num_axes : inputs[0]->shape().dims() - axis;
    CHECK_LE(axis + num_axes, inputs[0]->shape().dims());
    int outer_dim = inputs[0]->count(0, axis);
    int inner_dim = inputs[0]->count(axis + num_axes, inputs[0]->shape().dims());
    int scale_dim = inputs[0]->count(axis, axis + num_axes);
    if (inputs.size() > 1) {
        CHECK_EQ(scale_dim, inputs[1]->valid_size()) << "scale dim not valid";
    } else {
        CHECK_EQ(scale_dim, param.scale_w.size()) << "scale dim not valid";
    }

    // TODO !! need add other types of scale
    for (int outer_id = 0; outer_id < outer_dim; outer_id++) {
        for (int scale_id = 0; scale_id < scale_dim; scale_id++) {
            auto scale = scale_data[scale_id];
            auto bias = param.bias_term ? bias_data[scale_id] : 0;
            for (int inner_id = 0; inner_id < inner_dim; inner_id++) {
                *out_data = (*in_data) * scale + bias;
                in_data++;
                out_data++;
            }
        }
    }
    return SaberSuccess;
}

template class SaberScale<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

}
} // namespace anakin
