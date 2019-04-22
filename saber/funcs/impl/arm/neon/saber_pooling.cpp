#include "saber/funcs/impl/arm/saber_pooling.h"
#include "saber/funcs/impl/arm/neon/impl/pooling_arm_impl.h"
#include "saber/funcs/type_trans.h"

namespace anakin{

namespace saber{
template <>
SaberStatus SaberPooling<ARM, AK_FLOAT>::create(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        PoolingParam<ARM> &param, Context<ARM> &ctx) {
    if (param.global_pooling) {
        _impl = pooling_global;
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
        _pool_type = "pooling_global";
#endif
        return SaberSuccess;
    }
    if (param.window_h == 2 && param.window_w == 2 && \
        param.pooling_type != Pooling_average_exclude_padding){

        if (param.stride_h == 2 && param.stride_w == 2 && \
            param.pad_h == 0 && param.pad_w == 0) {
            if (param.pooling_type == Pooling_max) {
                _impl = pooling2x2s2_max;
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                _pool_type = "pooling2x2s2_max";
#endif
            } else {
                _impl = pooling2x2s2_ave;
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                _pool_type = "pooling2x2s2_ave";
#endif
            }
        }
        return SaberSuccess;
    }

    if (param.window_h == 3 && param.window_w == 3 && \
        param.pooling_type != Pooling_average_exclude_padding) {

            if (param.stride_h == 1 && param.stride_w == 1 && \
                param.pad_h == 1 && param.pad_w == 1) {
                if (param.pooling_type == Pooling_max) {
                    _impl = pooling3x3s1p1_max;
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                    _pool_type = "pooling3x3s1p1_max";
#endif
                } else {
                    _impl = pooling3x3s1p1_ave;
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                    _pool_type = "pooling3x3s1p1_ave";
#endif
                }
                return SaberSuccess;
            }

            if (param.stride_h == 2 && param.stride_w == 2) {
                if (param.pad_h == 0 &&  param.pad_w == 0) {
                    if (param.pooling_type == Pooling_max) {
                        _impl = pooling3x3s2p0_max;
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                        _pool_type = "pooling3x3s2p0_max";
#endif
                    } else {
                        _impl = pooling3x3s2p0_ave;
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                        _pool_type = "pooling3x3s2p0_ave";
#endif
                    }
                    return SaberSuccess;
                }
                if (param.pad_h == 1 && param.pad_w == 1) {
                    if (param.pooling_type == Pooling_max) {
                        _impl = pooling3x3s2p1_max;
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                        _pool_type = "pooling3x3s2p1_max";
#endif
                    } else {
                        _impl = pooling3x3s2p1_ave;
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                        _pool_type = "pooling3x3s2p1_ave";
#endif
                    }
                    return SaberSuccess;
                }
            }
        }
        _impl = pooling_basic;
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
        _pool_type = "pooling_basic";
#endif

#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
    int win1 = inputs[0]->width();
    int hin1 = inputs[0]->height();
    int chin1 = inputs[0]->channel();
    int num1 = inputs[0]->num();
    int kw = param.window_w;
    int kh = param.window_h;
    int pw = param.pad_w;
    int sw = param.stride_w;
    int ph = param.pad_h;
    int sh = param.stride_h;
    bool global = param.global_pooling;
    LOG(INFO) << "pooling fp32 param: " << " img_num = " << num1 << " in_channels = " << chin1 \
        << " img_h = " << hin1 << " img_w = " << win1 \
        << " pad_width = " << pw << " pad_height = " << ph << " stride_width = " \
        << sw << " stride_height = " << sh << " kernel_w = " << kw << " kernel_h = " \
        << kh << " global = " <<(global > 0 ? "true" : "false") << " type = " << _pool_type;
#endif
    return SaberSuccess;
}
template <>
SaberStatus SaberPooling<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        PoolingParam<ARM> &param) {

#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    const void* din = nullptr;
    void* dout = nullptr;
    int num = inputs[0]->num();
    int chout = outputs[0]->channel();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();

    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();
    float pool_scale = 1.0f;

    DataType tensor_in_type = inputs[0]->get_dtype();
    DataType tensor_out_type = outputs[0]->get_dtype();
#ifdef ENABLE_OP_TIMER
    this->_trans_timer.clear();
    this->_trans_timer.start(*this->_ctx);
#endif

    if (tensor_in_type == AK_INT8){
        _tmp_in.set_dtype(AK_FLOAT);
        _tmp_in.reshape(inputs[0]->valid_shape());
        trans_tensor_dtype<ARM, AK_INT8, AK_FLOAT>(*inputs[0], _tmp_in, inputs[0]->get_scale()[0], 1.f, {1.f});
        din = _tmp_in.data();
    } else {
        din = inputs[0]->data();
    }
    dout = outputs[0]->mutable_data();

#ifdef ENABLE_OP_TIMER
        this->_trans_timer.end(*this->_ctx);
#endif
    _impl(din, dout, num, chout, hout, wout, chin, hin, win, 0.f, param);

#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    float trans_ts = this->_trans_timer.get_average_ms();
    float op_macs = num * chout * wout * hout * param.window_h * param.window_w;
    LOG(INFO) << "Poooling fp32 type: " << this->_pool_type.c_str() << ", name: " << \
        this->_op_name.c_str() << ", pool time: " << ts \
        << "ms, GOPs: " << 1e-9f * op_macs << ",  GOPS: " << 0.000001 * op_macs / ts;
    GOPS ops;
    ops.ops = op_macs;
    ops.ts = ts;
    OpTimer::add_timer("Pooling", ops);
    OpTimer::add_timer("total", ops);

    GOPS trans_ops;
    trans_ops.ops = 0;
    trans_ops.ts = trans_ts;
    OpTimer::add_timer("pool tensor trans", trans_ops);
#endif

    return SaberSuccess;
}

template <>
SaberStatus SaberPooling<ARM, AK_INT8>::create(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        PoolingParam<ARM> &param, Context<ARM> &ctx) {
    DataType out_type = outputs[0]->get_dtype();
    if (param.global_pooling) {
        if (out_type == AK_FLOAT){
            _impl = pooling_global_int8_o_fp32;
        } else if (out_type == AK_INT8){
            _impl = pooling_global_int8_o_int8;
        }
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
        _pool_type = "pooling_global_int8";
#endif
        return SaberSuccess;
    }
    if (param.window_h == 2 && param.window_w == 2 && \
        param.pooling_type != Pooling_average_exclude_padding){

        if (param.stride_w == 2 && param.stride_h == 2 && \
            param.pad_h == 0 && param.pad_w == 0) {
            if (param.pooling_type == Pooling_max) {
                if (out_type == AK_FLOAT){
                    _impl = pooling2x2s2_max_int8_o_fp32;
                } else if (out_type == AK_INT8){
                    _impl = pooling2x2s2_max_int8_o_int8;
                }
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                _pool_type = "pooling2x2s2_max_int8";
#endif
            } else {
                if (out_type == AK_FLOAT){
                    _impl = pooling2x2s2_ave_int8_o_fp32;
                } else if (out_type == AK_INT8){
                    _impl = pooling2x2s2_ave_int8_o_int8;
                }
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                _pool_type = "pooling2x2s2_ave_int8";
#endif
            }
        }
        return SaberSuccess;
    }
    if (param.window_h == 3 && param.window_w == 3) {
        if (param.stride_h == 1 && param.stride_w == 1 && \
            param.pad_h == 1 && param.pad_w == 1) {
            if (param.pooling_type == Pooling_max) {
                if (out_type == AK_FLOAT){
                    _impl = pooling3x3s1p1_max_int8_o_fp32;
                } else if (out_type == AK_INT8){
                    _impl = pooling3x3s1p1_max_int8_o_int8;
                }
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                _pool_type = "pooling3x3s1_max_int8";
#endif
            } else {
                if (out_type == AK_FLOAT){
                    _impl = pooling_basic_int8_o_fp32;
                } else if (out_type == AK_INT8){
                    _impl = pooling_basic_int8_o_int8;
                }
            }
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
            _pool_type = " pooling_basic_int8";
#endif
        }
        return SaberSuccess;
    }
    if (param.window_h == 2 && param.window_w == 2) {
        if (param.pad_w == 0 &&  param.pad_h == 0) {
            if (param.pooling_type == Pooling_max) {
                if (out_type == AK_FLOAT){
                    _impl =  pooling3x3s2p0_max_int8_o_fp32;
                } else if (out_type == AK_INT8){
                    _impl =pooling3x3s2p0_max_int8_o_int8;
                }
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                _pool_type = "pooling3x3s2p0_max_int8";
#endif
            } else {
                if (out_type == AK_FLOAT){
                    _impl = pooling_basic_int8_o_fp32;
                } else if (out_type == AK_INT8){
                    _impl = pooling_basic_int8_o_int8;
                }
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                _pool_type = "pooling_basic_int8";
#endif
            }
            return SaberSuccess;
        }
        if (param.pad_w == 1 && param.pad_h == 1) {
            if (param.pooling_type == Pooling_max) {
                if (out_type == AK_FLOAT){
                    _impl = pooling3x3s2p1_max_int8_o_fp32;
                } else if (out_type == AK_INT8){
                    _impl = pooling3x3s2p1_max_int8_o_int8;
                }
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                _pool_type = "pooling3x3s2p1_max_int8";
#endif
            } else {
                if (out_type == AK_FLOAT){
                    _impl = pooling_basic_int8_o_fp32;
                } else if (out_type == AK_INT8){
                    _impl = pooling_basic_int8_o_int8;
                }
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
                _pool_type = "pooling_basic_int8";
#endif
            }
            return SaberSuccess;
        }
    }
    // default
    if (out_type == AK_FLOAT){
        _impl = pooling_basic_int8_o_fp32;
    } else if (out_type == AK_INT8){
        _impl = pooling_basic_int8_o_int8;
    }
#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
    _pool_type = "pooling_basic_int8";
#endif

#if defined ENABLE_OP_TIMER || defined(ENABLE_DEBUG)
    int win1 = inputs[0]->width();
    int hin1 = inputs[0]->height();
    int chin1 = inputs[0]->channel();
    int num1 = inputs[0]->num();
    int kw = param.window_w;
    int kh = param.window_h;
    int pw = param.pad_w;
    int sw = param.stride_w;
    int ph = param.pad_h;
    int sh = param.stride_h;
    bool global = param.global_pooling;
    LOG(INFO) << "pooling int8 param: " << " img_num = " << num1 << " in_channels = " << chin1 \
        << " img_h = " << hin1 << " img_w = " << win1 \
        << " pad_width = " << pw << " pad_height = " << ph << " stride_width = " \
        << sw << " stride_height = " << sh << " kernel_w = " << kw << " kernel_h = " \
        << kh << " global = " <<(global > 0 ? "true" : "false") << " type = " << _pool_type;
#endif
    return SaberSuccess;
}

template <>
SaberStatus SaberPooling<ARM, AK_INT8>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        PoolingParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    const void* din = nullptr;
    void* dout = nullptr;
    int num = inputs[0]->num();
    int chout = outputs[0]->channel();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();

    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();
    float pool_scale = 1.0f;

    DataType tensor_in_type = inputs[0]->get_dtype();
    DataType tensor_out_type = outputs[0]->get_dtype();
#ifdef ENABLE_OP_TIMER
    this->_trans_timer.clear();
    this->_trans_timer.start(*this->_ctx);
#endif

    if (tensor_in_type == AK_FLOAT){
        _tmp_in.set_dtype(AK_INT8);
        _tmp_in.reshape(inputs[0]->valid_shape());
        trans_tensor_dtype<ARM, AK_FLOAT, AK_INT8>(*inputs[0], _tmp_in, inputs[0]->get_scale()[0], 1.f, {1.f});
        din = _tmp_in.data();
    } else {
        din = inputs[0]->data();
    }
    // get pool_sale
    if (tensor_out_type == AK_FLOAT){
        pool_scale = inputs[0]->get_scale()[0];
    } else if (tensor_out_type == AK_INT8){
        pool_scale = inputs[0]->get_scale()[0] / outputs[0]->get_scale()[0];
    }
    dout = outputs[0]->mutable_data();

#ifdef ENABLE_OP_TIMER
        this->_trans_timer.end(*this->_ctx);
#endif
    _impl(din, dout, num, chout, hout, wout, chin, hin, win, pool_scale, param);

#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    float trans_ts = this->_trans_timer.get_average_ms();
    float op_macs = num * chout * wout * hout * param.window_h * param.window_w;
    LOG(INFO) << "Poooling int8 type: " << _pool_type <<", name: " << this->_op_name.c_str() << ", pool time: " << ts \
    << "ms, GOPs: " << 1e-9f * op_macs << ",  GOPS: " << 0.000001 * op_macs / ts;
    GOPS ops;
    ops.ops = op_macs;
    ops.ts = ts;
    OpTimer::add_timer("Pooling", ops);
    OpTimer::add_timer("total", ops);

    GOPS trans_ops;
    trans_ops.ops = 0;
    trans_ops.ts = trans_ts;
    OpTimer::add_timer("pool tensor trans", trans_ops);
#endif

    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(SaberPooling, PoolingParam, ARM, AK_HALF);

} //namespace anakin

} //namespace anakin
