
#include "saber/funcs/impl/x86/saber_conv_pooling.h"
#include "saber/funcs/calibrate.h"
#include "saber/funcs/impl/x86/saber_conv.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/funcs_utils.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_conv_pool_optimized.h"
#include "saber/funcs/impl/x86/kernel/jit_conv_pooling_normal.h"

namespace anakin {
namespace saber {

// FP32 part
template <>
SaberStatus SaberConv2DPooling<X86, AK_FLOAT>::\
        create(const std::vector<Tensor<X86> *>& inputs,
            std::vector<Tensor<X86> *>& outputs,
            ConvPoolingParam<X86>& param, Context<X86>& ctx) {

    _ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);
    _inner_tensor.reshape(_inner_shape);
    _inner_tensor_v.resize(1);
    _inner_tensor_v[0] = &_inner_tensor;

    _vender_conv.create(inputs, _inner_tensor_v, param.conv_param, ctx);
    _vender_pool.create(_inner_tensor_v, outputs, param.pooling_param, ctx);
    return SaberSuccess;
}

template <>
SaberStatus SaberConv2DPooling<X86, AK_FLOAT>::
        init(const std::vector<Tensor<X86> *>& inputs,
            std::vector<Tensor<X86> *>& outputs,
            ConvPoolingParam<X86>& param, Context<X86>& ctx) {

    _ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);
    _inner_tensor.re_alloc(_inner_shape, AK_FLOAT);

    _inner_tensor_v.resize(1);
    _inner_tensor_v[0] = &_inner_tensor;
    _vender_conv.init(inputs, _inner_tensor_v, param.conv_param, ctx);
    _vender_pool.init(_inner_tensor_v, outputs, param.pooling_param, ctx);
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConv2DPooling<X86, AK_FLOAT>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvPoolingParam<X86>& param) {
    _vender_conv.dispatch(inputs, _inner_tensor_v, param.conv_param);
    _vender_pool.dispatch(_inner_tensor_v, outputs, param.pooling_param);
    return SaberSuccess;
}

template class SaberConv2DPooling<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberConv2DPooling, ConvPoolingParam, X86, AK_HALF);

template <>
SaberStatus SaberConv2DPooling<X86, AK_INT8>::\
create(const std::vector<Tensor<X86> *>& inputs,
       std::vector<Tensor<X86> *>& outputs,
       ConvPoolingParam<X86>& param, Context<X86>& ctx) {
    SaberStatus ret = SaberUnImplError;

    this->_ctx = &ctx;
    ConvParam<X86> conv_param(param.conv_param);
    PoolingParam<X86> pool_param = param.pooling_param;

    DataType dtype_out = outputs[0]->get_dtype();
    DataType dtype_in = inputs[0]->get_dtype();
    // check layout info
    Shape out_shape = outputs[0]->valid_shape();
    Shape in_shape = inputs[0]->valid_shape();

    LayoutType layout_in = in_shape.get_layout();
    LayoutType layout_out = out_shape.get_layout();
    if (!((dtype_in != AK_FLOAT) && (dtype_out != AK_FLOAT) &&
          (layout_in == Layout_NHWC) && (layout_out == Layout_NHWC))) {
        return ret;
    }

    if (!this->conv_pool_impl_) {
        LOG(FATAL) << "impl is NULL";
        return SaberNotInitialized;
    }

    // conv pooling op create func
    ret = this->conv_pool_impl_->create(inputs, outputs, param, ctx);
    return ret;
}

template <>
SaberStatus SaberConv2DPooling<X86, AK_INT8>::\
init(const std::vector<Tensor<X86> *>& inputs,
     std::vector<Tensor<X86> *>& outputs,
     ConvPoolingParam<X86>& param, Context<X86>& ctx) {
    SaberStatus ret = SaberUnImplError;

    this->_ctx = &ctx;
    ConvParam<X86> conv_param(param.conv_param);
    PoolingParam<X86> pool_param = param.pooling_param;

    DataType dtype_out = outputs[0]->get_dtype();
    DataType dtype_in = inputs[0]->get_dtype();
    // check layout info
    Shape out_shape = outputs[0]->valid_shape();
    Shape in_shape = inputs[0]->valid_shape();

    LayoutType layout_in = in_shape.get_layout();
    LayoutType layout_out = out_shape.get_layout();

    if (!((dtype_in != AK_FLOAT) && (dtype_out != AK_FLOAT) &&
          (layout_in == Layout_NHWC) && (layout_out == Layout_NHWC))) {
        return ret;
    }

    // init conv pool op
    if (this->conv_pool_impl_) {
        delete this->conv_pool_impl_;
    }

    // first try optimized op
    this->conv_pool_impl_ = new JitAvx512ConvPoolOptimized;
    ret = this->conv_pool_impl_->init(inputs, outputs, param, ctx);

    if (ret != SaberSuccess) {
        // then try normal op
        delete this->conv_pool_impl_;
        this->conv_pool_impl_ = new JitConvPoolingNormal<AK_INT8>;
        ret = this->conv_pool_impl_->init(inputs, outputs, param, ctx);
    }
    LOG(INFO)<<"";
    return ret;
}

template <>
SaberStatus SaberConv2DPooling<X86, AK_INT8>::\
dispatch(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvPoolingParam<X86>& param) {
    SaberStatus ret = SaberSuccess;
    if (!this->conv_pool_impl_) {
        LOG(FATAL) << "impl is NULL";
        return SaberNotInitialized;
    }

    ret = this->conv_pool_impl_->dispatch(inputs, outputs, param);
    return ret;
}


}
}
