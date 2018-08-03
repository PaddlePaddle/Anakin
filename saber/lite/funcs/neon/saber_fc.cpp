#include "saber/lite/funcs/saber_fc.h"

#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/neon/impl/sgemv_arm.h"

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

SaberFc::SaberFc(const ParamBase *param) {
    _param = (const FcParam*)param;
    this->_flag_param = true;
}

SaberStatus SaberFc::load_param(const ParamBase *param) {
    _param = (const FcParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

//SaberFc::SaberFc(int axis, int num_output, bool flag_trans, bool flag_bias, \
//    const float *weights, const float *bias) {
//
//    _axis = axis;
//    _num_output = num_output;
//    _flag_trans = flag_trans;
//    _bias_term = flag_bias;
//    _weights = weights;
//    _bias = bias;
//}
//
//SaberStatus SaberFc::load_param(int axis, int num_output, bool flag_trans, bool flag_bias, \
//    const float *weights, const float *bias) {
//
//    _axis = axis;
//    _num_output = num_output;
//    _flag_trans = flag_trans;
//    _bias_term = flag_bias;
//    _weights = weights;
//    _bias = bias;
//    return SaberSuccess;
//}

SaberStatus SaberFc::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                          std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_param) {
        printf("load fc param first\n");
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

SaberStatus SaberFc::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs, \
    std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {

    if (!this->_flag_param) {
        printf("load fc param first\n");
        return SaberNotInitialized;
    }

    this->_ctx = &ctx;
    int threads = 1;
    this->_ctx->get_mode(threads);

    _m = inputs[0]->count_valid(0, _param->_axis);
    _k = inputs[0]->count_valid(_param->_axis, inputs[0]->dims());
    _n = _param->_num_output;

    int l1_cache = Env::cur_env()._L1_cache;
    int l2_cache = Env::cur_env()._L2_cache;
    //! if L1 cache size is not provided, set to 31K
    l1_cache = l1_cache > 0? l1_cache : 31000;
    //! if L2 cache size is not provided, set to 2M
    l2_cache = l2_cache > 0? l2_cache : 2000000;

    printf("fc weights transpose: %s\n", _param->_flag_trans? "true" : "false");
    if (_m > 1 || _param->_flag_trans) {
        _gemmer.init(l1_cache, l2_cache, _m, _n, _k, false, !_param->_flag_trans, threads);
    }
    this->_flag_init = true;
    return SaberSuccess;
}

//template <typename Dtype>
SaberStatus SaberFc::dispatch(\
    const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs, \
    std::vector<Tensor<CPU, AK_FLOAT> *>& outputs) {

    if (!this->_flag_init) {
        printf("init fc first\n");
        return SaberNotInitialized;
    }

    const float* din = inputs[0]->data();
    float* dout = outputs[0]->mutable_data();
    const float* weights = _param->_weights;
    const float* bias = nullptr;
    if (_param->_flag_bias) {
        bias = _param->_bias;
    }

    if (_m > 1 || _param->_flag_trans) {
        _gemmer(din, _k, weights, (_param->_flag_trans? _n : _k), dout, _n, 1.f, 0.f, false);
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
    return SaberSuccess;
}

} //namespace lite

} //namespace saber

} //namespace anakin

#endif

