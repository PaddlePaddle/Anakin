
#include "saber/funcs/impl/impl_define.h"
#include "saber/funcs/impl/x86/saber_softmax.h"
#include "mkl_cblas.h"
#include "mkl_vml_functions.h"

namespace anakin{
namespace saber {

template class SaberSoftmax<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberSoftmax<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        SoftmaxParam<OpTensor> &param, Context<X86> &ctx)
{
    this->_ctx = ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberSoftmax<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        SoftmaxParam<OpTensor> &param, Context<X86> &ctx)
{
//    LOG(INFO)<<"here!!!";
    this->_param = &param;
    this->_ctx = ctx;

    return SaberSuccess;
}

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
void SaberSoftmax<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::_max(
        int n, const float *x, float *max_data) {
    max_data[0] = x[0];
    for (int c = 1; c < n; ++c) {
        max_data[0] = max_data[0] > x[c] ? max_data[0] : x[c];
    }
}

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
void SaberSoftmax<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::_sub(
        int n, float alpha, const float *x, float *y) {
    for (int c = 0; c < n; ++c) {
        y[c] = x[c] - alpha;
    }
}

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
void SaberSoftmax<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::_exp(
        int n, const float *a, float *r) {
#if 1
    vsExp(n, a, r);
#else
    #pragma omp parallel for
    for (int c = 0; c < n; ++c) {
        r[c] = expf(a[c]);
    }
#endif
    return;
}

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
void SaberSoftmax<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::_sum(
        int n, const float *x, float *sum_data) {
    sum_data[0] = 0;
    for (int c = 0; c < n; ++c) {
        sum_data[0] += x[c];
    }
}

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
void SaberSoftmax<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::_scal
        (int n, float alpha, float *x) {
#if 0
    cblas_sscal(n, alpha, x, 1);
#else
#pragma omp parallel for
    for (int c = 0; c < n; ++c) {
        x[c] *= alpha;
    }
#endif
    return;
}

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberSoftmax<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        SoftmaxParam<OpTensor>& param) {
//            LOG(INFO)<<"here!!!";
    int num = inputs[0] -> num();
    int channel = inputs[0]->channel();
    float *src_ptr = inputs[0]->mutable_data();
    float *dst_ptr = outputs[0]->mutable_data();

#pragma omp parallel for schedule(static)
    for (int ou = 0; ou < num ; ou++) {
        const float *src_data = src_ptr + ou * channel;
        float *dst_data = dst_ptr + ou * channel;
        float scalar = 0;

        _max(channel, src_data, &scalar);
        _sub(channel, scalar, src_data, dst_data);
        _exp(channel, dst_data, dst_data);
        _sum(channel, dst_data, &scalar);
        _scal(channel, float(1)/scalar, dst_data);
    }
    return SaberSuccess;
}

}
} // namespace anakin
