#include "saber/lite/funcs/saber_fc.h"
#include "saber/lite/net/saber_factory_lite.h"
#ifdef USE_ARM_PLACE
#include "saber/lite/funcs/neon/impl/sgemv_arm.h"
#include "saber/lite/funcs/neon/impl/sgemm_conv.h"

namespace anakin{

namespace saber{

namespace lite{

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

SaberFc::SaberFc(ParamBase *param) {
    _param = (FcParam*)param;
    this->_flag_param = true;
}

SaberFc::~SaberFc() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberFc::load_param(ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (FcParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberFc::load_param(std::istream &stream, const float *weights) {
    int axis;
    int num_out;
    int bias_term;
    int size;
    int w_offset;
    int b_offset;
    int flag_trans;
    stream >> axis >> num_out >> bias_term >> size >> w_offset >> b_offset >> flag_trans;
    _param = new FcParam(axis, num_out, bias_term > 0, weights + w_offset, size, weights + b_offset, flag_trans > 0);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberFc::set_op_precision(DataType ptype) {
    if (ptype == AK_FLOAT || ptype == AK_INT8) {
        _precision_type = ptype;
        return SaberSuccess;
    } else {
        return SaberUnImplError;
    }
}

SaberStatus SaberFc::compute_output_shape(const std::vector<Tensor<CPU> *> &inputs,
                                          std::vector<Tensor<CPU> *> &outputs) {

    if (!this->_flag_param) {
        //printf("load fc param first\n");
        LOGE("load fc param first\n");
        return SaberNotInitialized;
    }

    Shape shape_out = inputs[0]->valid_shape();
    int m = inputs[0]->count_valid(0, _param->_axis);
    int k = inputs[0]->count_valid(_param->_axis, inputs[0]->dims());
    int n = _param->_num_output;

    shape_out.resize(_param->_axis + 1);
    shape_out[_param->_axis] = n;
    return outputs[0]->set_shape(shape_out);
}

SaberStatus SaberFc::init(const std::vector<Tensor<CPU> *> &inputs, \
    std::vector<Tensor<CPU> *> &outputs, Context &ctx) {

    if (!this->_flag_param) {
        //printf("load fc param first\n");
        LOGE("load fc param first\n");
        return SaberNotInitialized;
    }

    this->_ctx = &ctx;
    int threads = this->_ctx->get_threads();

    _m = inputs[0]->count_valid(0, _param->_axis);
    _k = inputs[0]->count_valid(_param->_axis, inputs[0]->dims());
    _n = _param->_num_output;

    if (_m > 1 || _param->_flag_trans) {
        int hblock = get_hblock(this->_ctx->get_arch());
        int m_round = hblock * ((_m + hblock - 1) / hblock);
        this->_ctx->workspace_extend(m_round * _k);
    }
    this->_flag_init = true;
    return SaberSuccess;
}

//template <typename Dtype>
SaberStatus SaberFc::dispatch(\
    const std::vector<Tensor<CPU> *>& inputs, \
    std::vector<Tensor<CPU> *>& outputs) {

    if (!this->_flag_init) {
        //printf("init fc first\n");
        LOGE("init fc first\n");
        return SaberNotInitialized;
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif

    const float* din = static_cast<const float*>(inputs[0]->data());
    float* dout = static_cast<float*>(outputs[0]->mutable_data());
    const float* weights = static_cast<const float*>(_param->_weights.data());
    const float* bias = nullptr;
    if (_param->_flag_bias) {
        bias = static_cast<const float*>(_param->_bias.data());
    }
    // printf("bias: %d, trans: %d, _m: %d,  _n: %d, _k: %d \n", _param->_flag_bias, _param->_flag_trans, _m, _n, _k);

    if (_m > 1 || _param->_flag_trans) {
        float* pre_din = static_cast<float*>(this->_ctx->get_work_space()) + this->_ctx->l2_cache_size() / sizeof(float);
        prepackA(pre_din, din, _k, 0, _m, 0, _k, false, this->_ctx);
        sgemm_prepack(pre_din, weights, bias, dout, _m, _n, _k, false, false, !_param->_flag_trans, this->_ctx);
        if (_param->_flag_bias) {
            fill_bias_fc(dout, bias, _m, _n);
        }
    } else {
        if (_param->_flag_bias) {
            sgemv_bias(false, _n, _k, weights, din, dout, bias);
        } else {
            sgemv(false, _n, _k, weights, din, dout);
        }
    }

#ifdef ENABLE_OP_TIMER
    float op_macs = _m * _n * _k;
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    GOPS ops;
    ops.ts = ts;
    ops.ops = op_macs;
    printf("fc %s: time: %f, GOPs: %f, GOPS: %f\n", this->_op_name.c_str(), ts, 1e-9f * op_macs, 0.000001 * op_macs / ts);
    OpTimer::add_timer("fc", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberFc);
} //namespace lite

} //namespace saber

} //namespace anakin

#endif

