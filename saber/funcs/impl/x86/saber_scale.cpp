
#include "saber/funcs/impl/x86/saber_scale.h"
namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberScale<X86, OpDtype>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ScaleParam<X86> &param,
        Context<X86> &ctx)
{

    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_in;
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_out;
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_op;
    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberScale<X86, OpDtype>::create(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ScaleParam<X86> &param,
        Context<X86> &ctx)
{
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberScale<X86, OpDtype>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ScaleParam<X86> &param)
{
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_in;
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_out;
    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_op;

    const DataType_op* in_data = (const DataType_op*)inputs[0]->data();
    DataType_op* out_data = (DataType_op*)outputs[0]->mutable_data();
    const DataType_op* scale_data = (inputs.size() > 1) ? (const DataType_op*)inputs[1]->data() : &(param.scale_w[0]);
    const DataType_op* bias_data = param.bias_term ? &(param.scale_b[0]) : NULL;

    const int count = inputs[0]->valid_size();
    int axis = (param.num_axes == 0) ? 0 : param.axis;
    int num_axes = param.num_axes >=0 ? param.num_axes : inputs[0]->shape().dims() - axis;
    CHECK_LE(axis + num_axes, inputs[0]->shape().dims());
    int outer_dim = inputs[0]->count_valid(0, axis);
    int inner_dim = inputs[0]->count_valid(axis + num_axes, inputs[0]->shape().dims());
    int scale_dim = inputs[0]->count_valid(axis, axis + num_axes);
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

template class SaberScale<X86, AK_FLOAT>;

}
} // namespace anakin
