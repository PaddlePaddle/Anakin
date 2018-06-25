#include "saber/funcs/impl/x86/vender_fc.h"
#include "mkl_cblas.h"

namespace anakin{
namespace saber {

typedef MKL_INT cblas_int;

template class VenderFc<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus VenderFc<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>
    ::init(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  FcParam<OpTensor> &param, Context<X86> &ctx)
{

    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus VenderFc<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>
    ::create(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  FcParam<OpTensor> &param, Context<X86> &ctx)
{
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    this->_ctx = &ctx;
    this->_param = &param;

    return SaberSuccess;
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus VenderFc<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>
    ::dispatch(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  FcParam<OpTensor> &param)
{
    if (inDtype == AK_FLOAT) {
        const float* src = static_cast<const float*>(inputs[0]->get_buf()->get_data());
        const float* weights = static_cast<const float*>(param.weights->get_buf()->get_data());
        const float* bias = NULL;
        if (param.bias)
            bias = static_cast<const float*>(param.bias->get_buf()->get_data());
        float* dst = static_cast<float*>(outputs[0]->get_buf()->get_data_mutable());

        // TODO: consistency checks
        int m = inputs[0]->count_valid(0, param.axis);
        int k = inputs[0]->count_valid(param.axis, inputs[0]->dims());
        const cblas_int MB = m;
        int channel_idx = outputs[0]->channel_index();
        Shape output_shape = outputs[0]->shape();
        const cblas_int OC = output_shape[channel_idx];
        const cblas_int IC = k;

        cblas_sgemm(CblasColMajor, param.is_transpose_weights ? CblasNoTrans : CblasTrans,
                          CblasNoTrans, OC, MB, IC,
                          1.0, weights, IC, src, IC, 0.0, dst, OC);
        if (bias) {
#pragma omp parallel for schedule(static)
            for (cblas_int mb = 0; mb < MB; mb++) {
                cblas_saxpy(OC, 1.0, bias, 1, dst + mb * OC, 1);
            }
        }
    }
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    return SaberSuccess;
}

}
} // namespace anakin
