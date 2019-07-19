#include "saber/funcs/impl/x86/kernel/jit_conv_pooling_normal.h"
#include "saber/funcs/impl/x86/saber_conv.h"
#include "saber/funcs/impl/x86/saber_pooling.h"

namespace anakin {
namespace saber {

using namespace jit;

template <>
SaberStatus JitConvPoolingNormal<AK_FLOAT>::allocate_buf(Shape buf_shape, std::vector<float> scale) {
    SaberStatus ret = SaberMemAllocFailed;

    Tensor<X86> *b_info = new Tensor<X86>(buf_shape, AK_FLOAT);
    if (buf_shape.get_layout() == Layout_NHWC) {
        delete b_info;
        b_info = new Tensor<X86>(buf_shape, AK_UINT8);
    }
    if (b_info) {
        b_info->set_scale(scale);
        buf_.push_back(b_info);
        ret = SaberSuccess;
    }
    return ret;
}

template <>
void JitConvPoolingNormal<AK_FLOAT>::release_buf() {

    for (int i = 0; i < this->buf_.size(); i++) {
        delete buf_[i];
        buf_[i] = nullptr;
    }
    std::vector<Tensor<X86> *> ().swap(buf_);
    return;
}

template <>
SaberStatus JitConvPoolingNormal<AK_FLOAT>::
        prepare_buf(Shape pool_shape, PoolingParam<X86> pool_param, std::vector<float> scale) {

    SaberStatus ret = SaberMemAllocFailed;

    // calculate the shape of buf
    Shape buf_shape({pool_shape[0], pool_shape[1],
            (pool_shape[2] - 1) * pool_param.stride_h + pool_param.window_h - 2 * pool_param.pad_h,
            (pool_shape[3] - 1) * pool_param.stride_w + pool_param.window_w - 2 * pool_param.pad_w,
            16}, Layout_NCHW_C16);

    LayoutType layout = pool_shape.get_layout();
    if (layout == Layout_NCHW_C16||layout == Layout_NCHW_C16R) {
        Shape buf_tmp({pool_shape[0], pool_shape[1],
            (pool_shape[2] - 1) * pool_param.stride_h + pool_param.window_h - 2 * pool_param.pad_h,
            (pool_shape[3] - 1) * pool_param.stride_w + pool_param.window_w - 2 * pool_param.pad_w,
            16}, Layout_NCHW_C16);
        buf_shape = buf_tmp;
    } else if (layout == Layout_NHWC) {
        Shape buf_tmp({pool_shape[0],
            (pool_shape[1] - 1) * pool_param.stride_h + pool_param.window_h - 2 * pool_param.pad_h,
            (pool_shape[2] - 1) * pool_param.stride_w + pool_param.window_w - 2 * pool_param.pad_w,
            pool_shape[3]}, Layout_NHWC);
        buf_shape = buf_tmp;
    } else {
        assert(!"not supported.");
    }

    // make sure allocate buf is successfully
    if (buf_.size() > 0 && buf_[0]->valid_shape() == buf_shape) {
        return SaberSuccess;
    }

    // release buf first
    this->release_buf();

    // allocate the buf according to the shape
    ret = allocate_buf(buf_shape, scale);
    return ret;
}

template <>
SaberStatus JitConvPoolingNormal<AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
                                               std::vector<Tensor<X86> *>& outputs,
                                               ConvPoolingParam<X86>& param, Context<X86>& ctx) {
    SaberStatus ret = SaberUnImplError;

    this->_ctx = &ctx;
    ConvParam<X86> conv_param(param.conv_param);
    PoolingParam<X86> pool_param = param.pooling_param;

    auto out_scale = outputs[0]->get_scale();
    DataType dtype_out = outputs[0]->get_dtype();
    DataType dtype_in = inputs[0]->get_dtype();
    // check layout info
    Shape out_shape = outputs[0]->valid_shape();
    Shape in_shape = inputs[0]->valid_shape();

    LayoutType layout_in = in_shape.get_layout();
    LayoutType layout_out = out_shape.get_layout();
    if (!(((dtype_in == AK_FLOAT) && (layout_in == Layout_NCHW) &&
        ((layout_out == Layout_NCHW_C16) || (layout_out == Layout_NHWC))) ||
        ((dtype_in == AK_FLOAT) && (dtype_out == AK_FLOAT) &&
        (layout_in == Layout_NCHW_C16) && (layout_out == Layout_NCHW_C16)))) {
        return ret;
    }

    if (!this->conv_impl_ || !this->pool_impl_) {
        LOG(ERROR) << "impl is NULL";
        return SaberNotInitialized;
    }

    // prepare buf
    ret = this->prepare_buf(out_shape, pool_param, out_scale);
    if (ret != SaberSuccess) {
        return ret;
    }

    // create conv act op
    ret = this->conv_impl_->create(inputs, buf_, conv_param, ctx);
    if (ret != SaberSuccess) {
        return ret;
    }

    // create pooling op
    ret = this->pool_impl_->create(buf_, outputs, pool_param, ctx);
    return ret;
}

template <>
SaberStatus JitConvPoolingNormal<AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
                                             std::vector<Tensor<X86> *>& outputs,
                                             ConvPoolingParam<X86>& param, Context<X86>& ctx) {
    SaberStatus ret = SaberUnImplError;

    this->_ctx = &ctx;
    ConvParam<X86> conv_param(param.conv_param);
    PoolingParam<X86> pool_param = param.pooling_param;

    auto out_scale = outputs[0]->get_scale();
    DataType dtype_out = outputs[0]->get_dtype();
    DataType dtype_in = inputs[0]->get_dtype();
    // check layout info
    Shape out_shape = outputs[0]->valid_shape();
    Shape in_shape = inputs[0]->valid_shape();

    LayoutType layout_in = in_shape.get_layout();
    LayoutType layout_out = out_shape.get_layout();

    if (!(((dtype_in == AK_FLOAT) && (layout_in == Layout_NCHW) &&
        ((layout_out == Layout_NCHW_C16) || (layout_out == Layout_NHWC))) ||
        ((dtype_in == AK_FLOAT) && (dtype_out == AK_FLOAT) &&
        (layout_in == Layout_NCHW_C16) && (layout_out == Layout_NCHW_C16)))) {
        return ret;
    }
    // prepare buf
    ret = this->prepare_buf(out_shape, pool_param, out_scale);
    if (ret != SaberSuccess) {
        return ret;
    }

    // init conv op
    if (this->conv_impl_) {
        delete this->conv_impl_;
    }
    this->conv_impl_ = new SaberConv2D<X86, AK_FLOAT>;
    ret = this->conv_impl_->init(inputs, buf_, conv_param, ctx);
    if (ret != SaberSuccess) {
        LOG(INFO) << "init convact impl error";
        return ret;
    }

    // init pool op
    if (this->pool_impl_) {
        delete this->pool_impl_;
    }

    if ((dtype_out == AK_FLOAT) && (layout_out == Layout_NCHW_C16 || layout_out == Layout_NCHW_C16R)) {
        this->pool_impl_ = new SaberPooling<X86, AK_FLOAT>;
    } else if ((dtype_out != AK_FLOAT) && (layout_out == Layout_NHWC)) {
        this->pool_impl_ = (Impl_pool_t*) new SaberPooling<X86, AK_INT8>;
    } else {
        LOG(INFO) << "not implemented.";
        ret = SaberUnImplError;
        return ret;
    }
    ret = this->pool_impl_->init(buf_, outputs, pool_param, ctx);

    return ret;
}

template <>
SaberStatus JitConvPoolingNormal<AK_FLOAT>::dispatch(const std::vector<Tensor<X86> *>& inputs,
             std::vector<Tensor<X86> *>& outputs,
             ConvPoolingParam<X86>& param) {
    SaberStatus ret = SaberSuccess;
    if (!this->conv_impl_ || !this->pool_impl_) {
        LOG(ERROR) << "impl is NULL";
        return SaberNotInitialized;
    }
    ConvParam<X86> conv_param(param.conv_param);
    PoolingParam<X86> pool_param = param.pooling_param;

    ret = this->conv_impl_->dispatch(inputs, buf_, conv_param);
    if (ret != SaberSuccess) {
        return ret;
    }

    ret = this->pool_impl_->dispatch(buf_, outputs, pool_param);
    return ret;
}


template <>
SaberStatus JitConvPoolingNormal<AK_INT8>::allocate_buf(Shape buf_shape, std::vector<float> scale) {
    SaberStatus ret = SaberMemAllocFailed;

    Tensor<X86> *b_info = new Tensor<X86>(buf_shape, AK_UINT8);
    if (b_info) {
        b_info->set_scale(scale);
        buf_.push_back(b_info);
        ret = SaberSuccess;
    }
    return ret;
}

template <>
void JitConvPoolingNormal<AK_INT8>::release_buf() {

    for (int i = 0; i < this->buf_.size(); i++) {
        delete buf_[i];
        buf_[i] = nullptr;
    }
    std::vector<Tensor<X86> *> ().swap(buf_);
    return;
}

template <>
SaberStatus JitConvPoolingNormal<AK_INT8>::prepare_buf(Shape pool_shape, PoolingParam<X86> pool_param, std::vector<float> scale) {

    SaberStatus ret = SaberMemAllocFailed;

    // calculate the shape of buf
    Shape buf_shape({pool_shape[0],
            (pool_shape[1] - 1) * pool_param.stride_h + pool_param.window_h - 2 * pool_param.pad_h,
            (pool_shape[2] - 1) * pool_param.stride_w + pool_param.window_w - 2 * pool_param.pad_w,
            pool_shape[3]}, Layout_NHWC);

    // make sure allocate buf is successfully
    if (buf_.size() > 0 && buf_[0]->valid_shape() == buf_shape) {
        return SaberSuccess;
    }

    // release buf first
    this->release_buf();

    // allocate the buf according to the shape
    ret = allocate_buf(buf_shape, scale);
    return ret;
}

template <>
SaberStatus JitConvPoolingNormal<AK_INT8>::create(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvPoolingParam<X86>& param, Context<X86>& ctx) {
    SaberStatus ret = SaberUnImplError;

    this->_ctx = &ctx;
    ConvParam<X86> conv_param(param.conv_param);
    PoolingParam<X86> pool_param = param.pooling_param;

    auto out_scale = outputs[0]->get_scale();
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

    if (!this->conv_impl_ || !this->pool_impl_) {
        LOG(FATAL) << "impl is NULL";
        return SaberNotInitialized;
    }

    // prepare buf
    ret = this->prepare_buf(out_shape, pool_param, out_scale);
    if (ret != SaberSuccess) {
        return ret;
    }

    // create conv act op
    ret = this->conv_impl_->create(inputs, buf_, conv_param, ctx);
    if (ret != SaberSuccess) {
        return ret;
    }

    // create pooling op
    ret = this->pool_impl_->create(buf_, outputs, pool_param, ctx);
    return ret;
}

template <>
SaberStatus JitConvPoolingNormal<AK_INT8>::init(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvPoolingParam<X86>& param, Context<X86>& ctx) {
    SaberStatus ret = SaberUnImplError;

    this->_ctx = &ctx;
    ConvParam<X86> conv_param(param.conv_param);
    PoolingParam<X86> pool_param = param.pooling_param;

    auto out_scale = outputs[0]->get_scale();
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

    // prepare buf
    ret = this->prepare_buf(out_shape, pool_param, out_scale);
    if (ret != SaberSuccess) {
        return ret;
    }
    // init conv op
    if (this->conv_impl_) {
        delete this->conv_impl_;
    }
    this->conv_impl_ = new SaberConv2D<X86, AK_INT8>;
    ret = this->conv_impl_->init(inputs, buf_, conv_param, ctx);
    if (ret != SaberSuccess) {
        LOG(FATAL) << "init convact impl error";
        return ret;
    }

    // init pool op
    if (this->pool_impl_) {
        delete this->pool_impl_;
    }

    this->pool_impl_ = new SaberPooling<X86, AK_INT8>;
    ret = this->pool_impl_->init(buf_, outputs, pool_param, ctx);

    return ret;
}

template <>
SaberStatus JitConvPoolingNormal<AK_INT8>::dispatch(const std::vector<Tensor<X86> *>& inputs,
             std::vector<Tensor<X86> *>& outputs,
             ConvPoolingParam<X86>& param) {
    SaberStatus ret = SaberSuccess;
    if (!this->conv_impl_ || !this->pool_impl_) {
        LOG(FATAL) << "impl is NULL";
        return SaberNotInitialized;
    }
    ConvParam<X86> conv_param(param.conv_param);
    PoolingParam<X86> pool_param = param.pooling_param;

    ret = this->conv_impl_->dispatch(inputs, buf_, conv_param);
    if (ret != SaberSuccess) {
        return ret;
    }

    ret = this->pool_impl_->dispatch(buf_, outputs, pool_param);
    return ret;
}


}
}
