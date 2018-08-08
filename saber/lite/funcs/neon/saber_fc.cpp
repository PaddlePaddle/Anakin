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

    int cnt = channel >> 2;
    int remain = channel & 3;

    for (int j = 0; j < num; ++j) {

        const float* ptr_bias = bias;
        float* ptr_out = tensor + j * channel;

        if (cnt > 0) {
            asm(
            ".fill_bias_fc: \n"
                    "vld1.32 {d0-d1}, [%[ptr_out]]  @ load data\n"
                    "vld1.32 {d2-d3}, [%[ptr_bias]]!  @ load data\n"
                    "vadd.f32 q2, q0, q1              @ add bias\n"
                    "vst1.32  {d4-d5}, [%[ptr_out]]!  @ store result\n"
                    "subs   %[cnt], #1                @ loop count -1\n"
                    "bne    .fill_bias_fc             @ jump to main loop\n"
            :[ptr_out] "+r"(ptr_out), [ptr_bias] "+r"(ptr_bias), \
                    [cnt] "+r"(cnt)
            :
            :"q0", "q1", "q2"
            );
        }

        for (; remain > 0; remain--) {
            *(ptr_out++) += *(ptr_bias++);
        }
    }
}


SaberFc::SaberFc(int axis, int num_output, bool flag_trans, bool flag_bias, \
    const float *weights, const float *bias) {

    _axis = axis;
    _num_output = num_output;
    _flag_trans = flag_trans;
    _bias_term = flag_bias;
    _weights = weights;
    _bias = bias;
}

SaberStatus SaberFc::load_param(int axis, int num_output, bool flag_trans, bool flag_bias, \
    const float *weights, const float *bias) {

    _axis = axis;
    _num_output = num_output;
    _flag_trans = flag_trans;
    _bias_term = flag_bias;
    _weights = weights;
    _bias = bias;
    return SaberSuccess;
}

SaberStatus SaberFc::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                          std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    Shape shape_out = inputs[0]->valid_shape();
    int m = inputs[0]->count_valid(0, _axis);
    int k = inputs[0]->count_valid(_axis, inputs[0]->dims());
    int n = _num_output;

    shape_out.resize(_axis + 1);
    shape_out[_axis] = n;
    return outputs[0]->set_shape(shape_out);
}

SaberStatus SaberFc::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs, \
    std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {

    _ctx = ctx;
    int threads = _ctx.get_act_ids().size();

    _m = inputs[0]->count_valid(0, _axis);
    _k = inputs[0]->count_valid(_axis, inputs[0]->dims());
    _n = _num_output;

    int l1_cache = Env::cur_env()._L1_cache;
    int l2_cache = Env::cur_env()._L2_cache;
    //! if L1 cache size is not provided, set to 31K
    l1_cache = l1_cache > 0? l1_cache : 31000;
    //! if L2 cache size is not provided, set to 2M
    l2_cache = l2_cache > 0? l2_cache : 2000000;

    printf("fc weights transpose: %s\n", _flag_trans? "true" : "false");
    if (_m > 1 || _flag_trans) {
        _gemmer.init(l1_cache, l2_cache, _m, _n, _k, false, !_flag_trans, threads);
    }
    return SaberSuccess;
}

//template <typename Dtype>
SaberStatus SaberFc::dispatch(\
    const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs, \
    std::vector<Tensor<CPU, AK_FLOAT> *>& outputs) {

    const float* din = inputs[0]->data();
    float* dout = outputs[0]->mutable_data();
    const float* weights = _weights;
    const float* bias = nullptr;
    if (_bias_term) {
        bias = _bias;
    }

    if (_m > 1 || _flag_trans) {
        _gemmer(din, _k, weights, (_flag_trans? _n : _k), dout, _n, 1.f, 0.f, false);
        if (_bias_term) {
            fill_bias_fc(dout, bias, _m, _n);
        }
    } else {
        if (_bias_term) {
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

