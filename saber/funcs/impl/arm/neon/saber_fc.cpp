#include "saber/funcs/impl/arm/saber_fc.h"
#include "saber/funcs/impl/arm/neon/impl/sgemm_prepacked.h"
#include "saber/funcs/impl/arm/neon/impl/gemm_prepacked_int8.h"
#include "saber/funcs/impl/arm/neon/impl/sgemv_arm.h"
#include "saber/funcs/impl/arm/neon/impl/gemv_arm_int8.h"
#include "saber/funcs/type_trans.h"
namespace anakin{
namespace saber {

template <typename Dtype>
void fill_bias_fc(Dtype* tensor, const Dtype* bias, const int num, const int channel);

template <>
void fill_bias_fc<float>(float* tensor, const float* bias, const int num, const int channel) {

    int cnt = channel >> 4;
    int remain = channel & 15;

    for (int j = 0; j < num; ++j) {

        const float* ptr_bias = bias;
        float* ptr_out = tensor + j * channel;

        float32x4_t vout1;
        float32x4_t vout2;
        float32x4_t vout3;
        float32x4_t vout4;

        for (int i = 0; i < cnt; ++i) {
            float32x4_t vin1 = vld1q_f32(ptr_out);
            float32x4_t vb1 = vld1q_f32(ptr_bias);

            float32x4_t vin2 = vld1q_f32(ptr_out + 4);
            float32x4_t vb2 = vld1q_f32(ptr_bias + 4);

            float32x4_t vin3 = vld1q_f32(ptr_out + 8);
            float32x4_t vb3 = vld1q_f32(ptr_bias + 8);

            float32x4_t vin4 = vld1q_f32(ptr_out + 12);
            float32x4_t vb4 = vld1q_f32(ptr_bias + 12);

            vout1 = vaddq_f32(vin1, vb1);
            vout2 = vaddq_f32(vin2, vb2);
            vout3 = vaddq_f32(vin3, vb3);
            vout4 = vaddq_f32(vin4, vb4);

            vst1q_f32(ptr_out, vout1);
            vst1q_f32(ptr_out + 4, vout2);
            vst1q_f32(ptr_out + 8, vout3);
            vst1q_f32(ptr_out + 12, vout4);

            ptr_out += 16;
            ptr_bias += 16;
        }

#if 0
        if (cnt > 0) {

            asm(
            "1: \n"
            "vld1.32 {d0-d1}, [%[ptr_out]]    @ load data\n"
            "vld1.32 {d2-d3}, [%[ptr_bias]]!  @ load data\n"
            "vadd.f32 q2, q0, q1              @ add bias\n"
            "vst1.32  {d4-d5}, [%[ptr_out]]!  @ store result\n"
            "subs   %[cnt], #1                @ loop count -1\n"
            "bne    1b                        @ jump to main loop\n"
            :[ptr_out] "+r"(ptr_out), [ptr_bias] "+r"(ptr_bias), \
                    [cnt] "+r"(cnt)
            :
            :"q0", "q1", "q2"
            );
        }
#endif
        for (; remain > 0; remain--) {
            *(ptr_out++) += *(ptr_bias++);
        }
    }
}

template <>
void fill_bias_fc<int>(int* tensor, const int* bias, const int num, const int channel) {

    int cnt = channel >> 4;
    int remain = channel & 15;

    for (int j = 0; j < num; ++j) {

        const int* ptr_bias = bias;
        int* ptr_out = tensor + j * channel;

        int32x4_t vout1;
        int32x4_t vout2;
        int32x4_t vout3;
        int32x4_t vout4;

        for (int i = 0; i < cnt; ++i) {
            int32x4_t vin1 = vld1q_s32(ptr_out);
            int32x4_t vb1 = vld1q_s32(ptr_bias);

            int32x4_t vin2 = vld1q_s32(ptr_out + 4);
            int32x4_t vb2 = vld1q_s32(ptr_bias + 4);

            int32x4_t vin3 = vld1q_s32(ptr_out + 8);
            int32x4_t vb3 = vld1q_s32(ptr_bias + 8);

            int32x4_t vin4 = vld1q_s32(ptr_out + 12);
            int32x4_t vb4 = vld1q_s32(ptr_bias + 12);

            vout1 = vaddq_s32(vin1, vb1);
            vout2 = vaddq_s32(vin2, vb2);
            vout3 = vaddq_s32(vin3, vb3);
            vout4 = vaddq_s32(vin4, vb4);

            vst1q_s32(ptr_out, vout1);
            vst1q_s32(ptr_out + 4, vout2);
            vst1q_s32(ptr_out + 8, vout3);
            vst1q_s32(ptr_out + 12, vout4);

            ptr_out += 16;
            ptr_bias += 16;
        }

#if 0
        if (cnt > 0) {

        asm(
        "1: \n"
        "vld1.32 {d0-d1}, [%[ptr_out]]    @ load data\n"
        "vld1.32 {d2-d3}, [%[ptr_bias]]!  @ load data\n"
        "vadd.s32 q2, q0, q1              @ add bias\n"
        "vst1.32  {d4-d5}, [%[ptr_out]]!  @ store result\n"
        "subs   %[cnt], #1                @ loop count -1\n"
        "bne    1b                        @ jump to main loop\n"
        :[ptr_out] "+r"(ptr_out), [ptr_bias] "+r"(ptr_bias), \
                [cnt] "+r"(cnt)
        :
        :"q0", "q1", "q2"
        );
    }
#endif
        for (; remain > 0; remain--) {
            *(ptr_out++) += *(ptr_bias++);
        }
    }
}

/****************************************** Fc Precision Is Float ******************************************/

template<>
SaberStatus SaberFc<ARM, AK_FLOAT>::create(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        FcParam<ARM>& param, Context<ARM>& ctx) {

    this->_ctx = &ctx;
    int threads = this->_ctx->get_threads();
    _m = inputs[0]->count_valid(0, param.axis);
    _k = inputs[0]->count_valid(param.axis, inputs[0]->dims());
    _n = param.num_output;
    if (_m > 1 || param.is_transpose_weights) {
        int hblock;
        hblock = get_hblock(this->_ctx->get_arch());
        int m_round = hblock * ((_m + hblock - 1) / hblock);
        this->_ctx->workspace_extend(Shape({1, 1, 1, m_round * _k}));
    }
    return SaberSuccess;
}

template<>
SaberStatus SaberFc<ARM, AK_FLOAT>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        FcParam<ARM>& param, Context<ARM>& ctx){
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberFc<ARM, AK_FLOAT>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        FcParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    bool flag_bias = (param.bias != nullptr);
    const float* din = static_cast<const float*>(inputs[0]->data());
    float* dout = static_cast<const float*>(outputs[0]->mutable_data());
    const float* weights = static_cast<const float*>(param.weights->data());
    const float* bias = flag_bias ? param.bias->data() : nullptr;
    if (_m > 1 || param.is_transpose_weights) {
        float *pre_din = static_cast<float *>(this->_ctx->get_work_space()) + \
                        this->_ctx->get_l2_cache_size() / sizeof(float);
        prepackA(pre_din, din, _k, 0, _m, 0, _k, false, this->_ctx);
        sgemm_prepack(pre_din, weights, bias, dout, _m, _n, _k, false, false,\
                         !param.is_transpose_weights, this->_ctx);
        if (flag_bias) {
            fill_bias_fc(dout, bias, _m, _n);
        }
    } else {
        sgemv(weights, din, dout, false, _n, _k, flag_bias, bias, false);
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    float op_macs = 2.f * _m * _n * _k;
    LOG(INFO) << "fc fp32: " << this->_op_name.c_str() << ", time: " << ts << ", GOPs: " << \
        1e-9f * op_macs << ", GOPS: "<< 0.000001 * op_macs / ts;
    GOPS ops;
    //fixme
    ops.ops = op_macs;
    ops.ts = ts;
    OpTimer::add_timer("fc", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
/****************************************** Fc Precision Is Int8 ******************************************/

template<>
SaberStatus SaberFc<ARM, AK_INT8>::create(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        FcParam<ARM>& param, Context<ARM>& ctx) {

    this->_ctx = &ctx;
    int threads = this->_ctx->get_threads();
    _m = inputs[0]->count_valid(0, param.axis);
    _k = inputs[0]->count_valid(param.axis, inputs[0]->dims());
    _n = param.num_output;
    if (_m > 1 || param.is_transpose_weights) {
        int hblock;
        hblock = get_hblock_int8(this->_ctx->get_arch());
        int m_round = hblock * ((_m + hblock - 1) / hblock);
        this->_ctx->workspace_extend(Shape({1, 1, 1, m_round * _k}));
    }
    //! init int32 output
    if (outputs[0]->get_dtype() != AK_INT32) {
        _tmp_out.set_dtype(AK_INT32);
        _tmp_out.reshape(outputs[0]->valid_shape());
    }
    return SaberSuccess;
}

template<>
SaberStatus SaberFc<ARM, AK_INT8>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        FcParam<ARM>& param, Context<ARM>& ctx){
        this->_ctx = &ctx;
        //! trans fc weights to int8
        if (trans_weights_dtype(*param.weights, AK_INT8, 127.f, FC_TYPE, 1) != SaberSuccess) {
            LOG(ERROR) << "ERROR: fc trans weights to int8 failed";
            return SaberInvalidValue;
        }
        bool flag_bias = (param.bias != nullptr);
        //! trans fc bias to int32
        if (flag_bias) {
            trans_fp32_bias_to_int32(*param.bias, *param.bias, inputs[0]->get_scale()[0], param.weights->get_scale());
        }
        return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberFc<ARM, AK_INT8>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        FcParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    bool flag_bias = (param.bias != nullptr);
    const void* din = nullptr;
    void* dout = nullptr;
    const int8_t* weights = static_cast<const int8_t*>(param.weights->data());
    const int32_t* bias = flag_bias ? param.bias->data() : nullptr;

    //! input dtype transfer
    if (inputs[0]->get_dtype() != AK_INT8) {
        _tmp_in.set_dtype(AK_INT8);
        _tmp_in.reshape(inputs[0]->valid_shape());
        trans_tensor_dtype<ARM, AK_FLOAT, AK_INT8>(*inputs[0], _tmp_in, inputs[0]->get_scale()[0], 1.f, {1.f});
        din = _tmp_in.data();
    } else {
        din = inputs[0]->data();
    }

    if (outputs[0]->get_dtype() == AK_INT32) {
        dout = outputs[0]->mutable_data();
    } else {
        dout = _tmp_out.mutable_data();
    }

    if (_m > 1 || param.is_transpose_weights) {
        const int8_t *pre_din = static_cast<const int8_t *>(this->_ctx->get_work_space()) + \
                        this->_ctx->get_l2_cache_size() / sizeof(float);
        prepackA_int8(pre_din, static_cast<const int8_t*>(din), _k, 0, _m, 0, _k, false);
        gemm_prepack_int8(pre_din, weights, bias, static_cast<int32_t*>(dout), _m, _n, _k, false, false,\
                         !param.is_transpose_weights, nullptr, this->_ctx);
        if (flag_bias) {
            fill_bias_fc(static_cast<int32_t*>(dout), bias, _m, _n);
        }
    } else {
        gemv_int8(weights, static_cast<const int8_t*>(din), static_cast<int32_t*>(dout), false, \
            _n, _k, nullptr, flag_bias, bias, false);
    }

    //! output dtype transfer
    if (outputs[0]->get_dtype() == AK_INT8) {
        trans_tensor_dtype<ARM, AK_INT32, AK_INT8>(_tmp_out, *outputs[0], inputs[0]->get_scale()[0], \
        outputs[0]->get_scale()[0], param.weights->get_scale());
    } else if (outputs[0]->get_dtype() == AK_FLOAT) {
        trans_tensor_dtype<ARM, AK_INT32, AK_FLOAT>(_tmp_out, *outputs[0], inputs[0]->get_scale()[0], \
            1.f, param.weights->get_scale());
    } else if (outputs[0]->get_dtype() != AK_INT32) {
        LOG(ERROR) << "unsupported precision type!!";
        return SaberInvalidValue;
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    float op_macs = 2.f * _m * _n * _k;
    LOG(INFO) << "fc int8: " << this->_op_name.c_str() << ", time: " << ts << ", GOPs: " << \
        1e-9f * op_macs << ", GOPS: "<< 0.000001 * op_macs / ts;
    GOPS ops;
    //fixme
    ops.ops = op_macs;
    ops.ts = ts;
    OpTimer::add_timer("fc", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(SaberFc, FcParam, ARM, AK_HALF);
}
} // namespace anakin
