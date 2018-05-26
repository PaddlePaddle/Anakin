
#include "saber/funcs/impl/impl_define.h"
#include "saber/funcs/impl/x86/saber_pooling.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_pool_kernel_f32.h"


namespace anakin{
namespace saber {

using namespace jit;

template class SaberPooling<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberPooling<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        PoolingParam<OpTensor> &param, Context<X86> &ctx)
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
SaberStatus SaberPooling<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        PoolingParam<OpTensor> &param,
        Context<X86> &ctx)
{
    jit_pool_conf_t jpp_;
    if(init_conf(jpp_, inputs, outputs, param) != SaberSuccess) {
        return SaberUnImplError;
    }
    kernel_ = new jit_uni_pool_kernel_f32<avx512_common>(jpp_);
    return SaberSuccess;
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberPooling<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>
    ::dispatch(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  PoolingParam<OpTensor> &param)
{
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    return SaberSuccess;
      
}

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberPooling<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init_conf(
        jit_pool_conf_t &jpp, const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        PoolingParam<OpTensor> &param) {

    using namespace utils;

    Shape src_shape(inputs[0]->shape());
    Shape dst_shape(outputs[0]->shape());
    bool ok = true
              && mayiuse(avx512_common)
              && std::is_same<LayOutType_in, NCHW_C16>::value
              && std::is_same<LayOutType_op, NCHW>::value
              && one_of(param.pooling_type, Pooling_max,
                        Pooling_average_include_padding,
                        Pooling_average_exclude_padding);
    if (!ok) {
        return SaberUnImplError;
    }

    const int simd_w = 16;
    const int ndims = 4;

    jpp.ndims = ndims;
    jpp.mb = src_shape[0];
    jpp.c = src_shape[1] * 16;
    jpp.id = (ndims == 5) ? src_shape[2] : 1;
    jpp.ih = src_shape[ndims - 2];
    jpp.iw = src_shape[ndims - 1];
    jpp.od = (ndims == 5) ? dst_shape[2] : 1;
    jpp.oh = dst_shape[ndims - 2];
    jpp.ow = dst_shape[ndims - 1];

    jpp.stride_d = 1;
    jpp.stride_h = param.stride_h;
    jpp.stride_w = param.stride_w;
    jpp.kd = 1;
    jpp.kh = param.window_h;
    jpp.kw = param.window_w;

    jpp.f_pad = 0;
    jpp.t_pad = param.pad_h;
    jpp.l_pad = param.pad_w;

    jpp.alg = param.pooling_type;

    jpp.ind_dt = AK_FLOAT;

    jpp.simple_alg = false;

    jpp.c_block = simd_w;

    jpp.nb_c = jpp.c / jpp.c_block;
    if (jpp.alg == Pooling_max) {
        jpp.ur_w = 16;
    } else {
        jpp.ur_w = 24;
    }

    if (jpp.ow < jpp.ur_w) {
        jpp.ur_w = jpp.ow;
    }
    if (jpp.l_pad > jpp.ur_w) {
        return SaberUnImplError;
    }

    jpp.ur_w_tail = jpp.ow % jpp.ur_w;
    if (jit_uni_pool_kernel_f32<avx512_common>::init_conf(jpp)) {
        return SaberSuccess;
    } else {
        return SaberUnImplError;
    }
}
template class SaberPooling<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW_C16, NCHW_C16>;
}
} // namespace anakin
