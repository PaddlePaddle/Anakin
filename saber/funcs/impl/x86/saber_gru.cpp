

#include "saber/funcs/impl/x86/saber_gru.h"
#include "saber/core/tensor_op.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "mkl_cblas.h"
#include <immintrin.h>
#include "avx_mathfun.h"
namespace anakin {

namespace saber {


#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0
#define EXP_MAX_INPUT 40.0

inline __m256 InValidAct(__m256 a) {
            CHECK_EQ(0,1)<<"InValidAct";
}

inline __m256 Exp(__m256 a) {
    return exp256_ps(a);
    //    return exp(a);
}

inline __m256 Relu(const __m256 a) {
    __m256 tmp = _mm256_set1_ps(0.0f);
    return _mm256_max_ps(a, tmp);
}

inline __m256 Sigmoid_fluid(const __m256 a) {
    __m256 max = _mm256_set1_ps(SIGMOID_THRESHOLD_MAX);
    __m256 min = _mm256_set1_ps(SIGMOID_THRESHOLD_MIN);
    __m256 tmp = _mm256_max_ps(a, min);
    tmp = _mm256_min_ps(tmp, max);
    tmp = _mm256_sub_ps(_mm256_set1_ps(0.0f), tmp);
    tmp = Exp(tmp);
    tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp);
    tmp = _mm256_div_ps(_mm256_set1_ps(1.0f), tmp);
    return tmp;
}

inline __m256 Sigmoid(const __m256 a) {
    __m256  tmp = a;
    tmp = _mm256_sub_ps(_mm256_set1_ps(0.0f), tmp);
    tmp = Exp(tmp);
    tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp);
    tmp = _mm256_div_ps(_mm256_set1_ps(1.0f), tmp);
    return tmp;
}

inline __m256 Tanh_fluid(const __m256 a) {
    __m256 max = _mm256_set1_ps(EXP_MAX_INPUT);
    __m256 tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), a);
    tmp = _mm256_min_ps(tmp, max);
    tmp = Exp(tmp);
    return _mm256_sub_ps(_mm256_div_ps(_mm256_set1_ps(2.0f),
                                       _mm256_add_ps(_mm256_set1_ps(1.0f), tmp)),
                         _mm256_set1_ps(1.0f));
}

inline __m256 Tanh(const __m256 a) {
    __m256 tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), a);
    tmp = Exp(tmp);
    return _mm256_sub_ps(_mm256_div_ps(_mm256_set1_ps(2.0f),
                                       _mm256_add_ps(_mm256_set1_ps(1.0f), tmp)),
                         _mm256_set1_ps(1.0f));
}

__m256 Identity(const __m256 a) {
    return a;
}

static  __m256 ( *act_funcs[10])(const __m256)={&InValidAct,&Sigmoid,&Relu,&Tanh,&InValidAct,\
                       &InValidAct,&Identity,&Sigmoid_fluid,&Tanh_fluid};

//inline
static void gemm(const bool TransA, const bool TransB, int m, int n, int k, const float alpha,
                 const float* a, const float* b, const float beta, float* c) {
    //    cout << "(" << m << "," << n << "," << k << ")" << endl;
    int lda = (!TransA/* == CblasNoTrans*/) ? k : m;
    int ldb = (!TransB/* == CblasNoTrans*/) ? n : k;
    CBLAS_TRANSPOSE cuTransA =
        (!TransA/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cuTransB =
        (!TransB/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);
};

template <typename Dtype>
inline Dtype Sigmoid(const Dtype a) {
    return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-a));
}

template <typename Dtype>
inline Dtype Tanh_fluid(const Dtype a) {
    Dtype tmp = -2.0 * a;
    tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
    return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

template <typename Dtype>
inline Dtype Tanh(const Dtype a) {
    Dtype tmp = -2.0 * a;
    return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

template<>
SaberStatus SaberGru<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::naiv_gru(\
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        GruParam<OpTensor>& param) {
    CHECK_NE(param._formula, GRU_CUDNN) << "X86 gru not support cudnn formula now";
    const OpDataType* weight_h = _weights_h2h.data();
    const OpDataType* weight_w = _weights_i2h.data();
    const OpDataType* bias = _weights_bias.data();

    std::vector<int> offset_vec = inputs[0]->get_seq_offset();
    bool is_hw2seq = offset_vec.size() > 2;

    int word_sum = is_hw2seq ? offset_vec[offset_vec.size() - 1] : inputs[0]->channel();
    const OutDataType* h_init = nullptr;

    if (inputs.size() > 1) {
        h_init = inputs[1]->data();
    } else if (param.init_hidden() != nullptr) {
        h_init = param.init_hidden()->data();
    }

    const InDataType* x = inputs[0]->data();
    OutDataType* out = outputs[0]->mutable_data();

    bool is_reverse = param._is_reverse;


    std::vector<int> length_vec(offset_vec.size() - 1);
    int batch_size = offset_vec.size() - 1;
    int seqsum = 0;
    int max_seq_len = 0;

    for (int i = 0; i < offset_vec.size() - 1; ++i) {
        int len = offset_vec[i + 1] - offset_vec[i];
        max_seq_len = max_seq_len > len ? max_seq_len : len;
        length_vec[i] = len;
        seqsum += len;
    }

    //        Shape wx_shaep(1,seqsum,3,_aligned_hidden_size_iter_num,_aligned_size);
    _temp_wx.try_expand_size(seqsum * 3 * _hidden_size);
    _temp_wh.try_expand_size(batch_size * 2 * _hidden_size);
    _temp_whr.try_expand_size(batch_size * _hidden_size);

    OutDataType* temp_wh = _temp_wh.mutable_data();
    OutDataType* temp_wx = _temp_wx.mutable_data();
    OutDataType* temp_whr = _temp_whr.mutable_data();


//    LOG(INFO) << "gemm b" << inputs[0]->valid_shape().count() << "," <<
//              _weights_i2h.valid_shape().count() << "," << _temp_wx.valid_shape().count();
    //wx
    gemm(false, false, seqsum, 3 * _hidden_size, _word_size, 1.f, x, weight_w, 0.f, temp_wx);

    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;
    const OpDataType* b_r = bias + r_offset * _hidden_size;
    const OpDataType* b_z = bias + z_offset * _hidden_size;
    const OpDataType* b_o = bias + o_offset * _hidden_size;


    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        int batch_offset = offset_vec[batch_id];
        int batch_length = length_vec[batch_id];

        for (int seq_id_in_batch = 0; seq_id_in_batch < length_vec[batch_id]; ++seq_id_in_batch) {
            int seqid = batch_offset + seq_id_in_batch;
            int last_seq_id = seqid - 1;

            if (is_reverse) {
                seqid = batch_offset + batch_length - 1 - seq_id_in_batch;
                last_seq_id = seqid + 1;
            }

            const OutDataType* hin;
            OutDataType* hout = seqid * _hidden_size + out;

            if (seq_id_in_batch == 0) {
                hin = h_init + batch_id * _hidden_size;

            } else {
                hin = out + last_seq_id * _hidden_size;
            }

            gemm(false, false, 1, 2 * _hidden_size, _hidden_size, 1.0, hin,
                 weight_h + _hidden_size * _hidden_size,
                 0.f, temp_wh);

            OutDataType r;
            OutDataType z;
            OutDataType _h;
            OutDataType* w_x_r = temp_wx + r_offset * _hidden_size
                                 + seqid * _hidden_size * 3;
            OutDataType* w_x_z = temp_wx + z_offset * _hidden_size
                                 + seqid * _hidden_size * 3;
            OutDataType* w_x_o = temp_wx + o_offset * _hidden_size
                                 + seqid * _hidden_size * 3;

            OutDataType* w_h_r = temp_wh + 0 * _hidden_size;
            OutDataType* w_h_z = temp_wh + 1 * _hidden_size;
            OpDataType* w_o = weight_h;

            for (int frame_id = 0; frame_id < _hidden_size; ++frame_id) {
                r = w_x_r[frame_id] + w_h_r[frame_id] + b_r[frame_id]; //h_out=gate_r
                r = Sigmoid(r);
                hout[frame_id] = r * hin[frame_id];
            }

            gemm(false, false, 1, _hidden_size, _hidden_size, 1.0, hout, w_o, 0.f, temp_whr);

            for (int frame_id = 0; frame_id < _hidden_size; ++frame_id) {
                z = Sigmoid(w_x_z[frame_id] + w_h_z[frame_id] + b_z[frame_id]);
                _h = w_x_o[frame_id] + temp_whr[frame_id] + b_o[frame_id];
                _h = tanh(_h);
                hout[frame_id] = (1 - z) * hin[frame_id] + z * _h;
            }
        }

    }

    return SaberSuccess;
};

template<>
SaberStatus SaberGru<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::batch_gru(\
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        GruParam<OpTensor>& param) {
    CHECK_NE(param._formula, GRU_CUDNN) << "X86 gru not support cudnn formula now";
    const OpDataType* weight_h = _weights_h2h.data();
    const OpDataType* weight_w = _weights_i2h.data();
    const OpDataType* bias = _weights_bias.data();

    std::vector<int> offset_vec = inputs[0]->get_seq_offset();
    bool is_hw2seq = offset_vec.size() > 2;
    int word_sum = is_hw2seq ? offset_vec[offset_vec.size() - 1] : inputs[0]->channel();

    const InDataType* x = inputs[0]->data();
    OutDataType* out = outputs[0]->mutable_data();
    bool is_reverse = param._is_reverse;


    std::vector<int> length_vec(offset_vec.size() - 1);
    int batch_size = offset_vec.size() - 1;
    int seqsum = 0;
    int max_seq_len = 0;

    for (int i = 0; i < offset_vec.size() - 1; ++i) {
        int len = offset_vec[i + 1] - offset_vec[i];
        max_seq_len = max_seq_len > len ? max_seq_len : len;
        length_vec[i] = len;
        seqsum += len;
    }

    const OutDataType* h_init = nullptr;

    if (inputs.size() > 1) {
        h_init = inputs[1]->data();
    } else if (param.init_hidden() != nullptr) {
        CHECK_EQ(param.init_hidden()->valid_shape().count(),
                 batch_size * _hidden_size) << "hinit must match batchsize";
        h_init = param.init_hidden()->data();
    }


    //        Shape wx_shaep(1,seqsum,3,_aligned__hidden_size_iter_num,_aligned_size);
    _temp_wx.try_expand_size(seqsum * 3 * _hidden_size);
    _temp_wh.try_expand_size(batch_size * 2 * _hidden_size);
    _temp_whr.try_expand_size(batch_size * _hidden_size);

    OutDataType* temp_wh = _temp_wh.mutable_data();
    OutDataType* temp_wx = _temp_wx.mutable_data();
    OutDataType* temp_whr = _temp_whr.mutable_data();
    ///////////////////////////////////////////////////////
    std::vector<int> emit_offset_vec;
    int emit_length = 0;

    utils::SeqSortedseqTranseUtil transe_util;
    bool transform = transe_util.get_sorted_map(offset_vec, emit_offset_vec, emit_length);
    //    print_vec(_map_vec.data(),_map_vec.size(),"map ");
    float* inner_h_out = out;
    float* inner_x = x;
    const float* inner_h_init = h_init;

    //    print_vec(x,word_sum*word_size,"before  x");
    if (transform) {
        _temp_out.try_expand_size(seqsum * _hidden_size * param._num_direction);
        _temp_x.try_expand_size(seqsum * _word_size);
        inner_h_out = _temp_out.mutable_data();
        inner_x = _temp_x.mutable_data();
        transe_util.seq_2_sorted_seq(x, inner_x, _word_size);

        if (inner_h_init != nullptr) {
            _temp_h_init.try_expand_size(batch_size * _hidden_size);
            transe_util.hidden_2_sorted_hidden(inner_h_init, _temp_h_init.mutable_data(), _hidden_size);
            inner_h_init = _temp_h_init.data();
        }

    }

    gemm(false, false, word_sum, 3 * _hidden_size, _word_size, 1.f, inner_x, weight_w, 0.f,
         temp_wx);

    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;
    const float* b_r = bias + r_offset * _hidden_size;
    const float* b_z = bias + z_offset * _hidden_size;
    const float* b_o = bias + o_offset * _hidden_size;

    for (int word_id = 0; word_id < emit_length; word_id++) {
        int emit_word_id_start = emit_offset_vec[word_id];
        int emit_word_id_end = emit_offset_vec[word_id + 1];
        int emit_word_length = emit_word_id_end - emit_word_id_start;
        const float* hin;

        if (word_id == 0) {
            hin = inner_h_init;
        } else {
            hin = inner_h_out + emit_offset_vec[word_id - 1] * _hidden_size;
        }

        float* hout = emit_offset_vec[word_id] * _hidden_size + inner_h_out;

        gemm(false, false, emit_word_length, 2 * _hidden_size, _hidden_size, 1.0, hin,
             weight_h + _hidden_size * _hidden_size,
             0.f, temp_wh);

        float r;
        float z;
        float _h;

        const float* w_o = weight_h;

        for (int emit_word_id = emit_word_id_start; emit_word_id < emit_word_id_end; emit_word_id++) {
            int emit_id_offset = emit_word_id - emit_word_id_start;
            float* w_x_r = temp_wx + r_offset * _hidden_size
                           + emit_word_id * _hidden_size * 3;
            float* w_h_r = temp_wh + 0 * _hidden_size
                           + emit_id_offset * _hidden_size * 2;
            float* emit_hout = hout + emit_id_offset * _hidden_size;
            const float* emit_hin = hin + emit_id_offset * _hidden_size;

            for (int frame_id = 0; frame_id < _hidden_size; ++frame_id) {
                r = w_x_r[frame_id] + w_h_r[frame_id] + b_r[frame_id]; //h_out=gate_r
                r = Sigmoid(r);
                emit_hout[frame_id] = r * emit_hin[frame_id];
            }
        }

        gemm(false, false, emit_word_length, _hidden_size, _hidden_size, 1.0, hout, w_o, 0.f,
             temp_whr);

        for (int emit_word_id = emit_word_id_start; emit_word_id < emit_word_id_end; emit_word_id++) {
            int emit_offset = emit_word_id - emit_word_id_start;
            float* w_x_z = temp_wx + z_offset * _hidden_size
                           + emit_word_id * _hidden_size * 3;
            float* w_x_o = temp_wx + o_offset * _hidden_size
                           + emit_word_id * _hidden_size * 3;

            float* w_h_z = temp_wh + 1 * _hidden_size
                           + emit_offset * _hidden_size * 2;
            float* w_h_o = temp_whr + emit_offset * _hidden_size;
            float* emit_hout = hout + emit_offset * _hidden_size;
            const float* emit_hin = hin + emit_offset * _hidden_size;

            for (int frame_id = 0; frame_id < _hidden_size; ++frame_id) {

                z = Sigmoid(w_x_z[frame_id] + w_h_z[frame_id] + b_z[frame_id]);
                _h = w_x_o[frame_id] + w_h_o[frame_id] + b_o[frame_id];
                _h = Tanh(_h);
                emit_hout[frame_id] = (1 - z) * emit_hin[frame_id] + z * _h;
            }
        }

    }

    if (transform) {
        transe_util.sorted_seq_2_seq(inner_h_out, out, _hidden_size);
    }
}

template<>
SaberStatus SaberGru<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::
naiv_256(const std::vector<DataTensor_in*>& inputs,
         std::vector<DataTensor_out*>& outputs,
         GruParam<OpTensor>& param) {
    CHECK_NE(param._formula, GRU_CUDNN) << "X86 gru not support cudnn formula now";
    const OpDataType* weight_h = _weights_h2h.data();
    const OpDataType* weight_w = _weights_i2h.data();
    const OpDataType* bias = _weights_bias.data();

    std::vector<int> offset_vec = inputs[0]->get_seq_offset();
    bool is_hw2seq = offset_vec.size() > 2;
    int word_sum = is_hw2seq ? offset_vec[offset_vec.size() - 1] : inputs[0]->channel();

    const OutDataType* h_init = nullptr;

    if (inputs.size() > 1) {
        h_init = inputs[1]->data();
    } else if (param.init_hidden() != nullptr) {
        h_init = param.init_hidden()->data();
    }

    const InDataType* x = inputs[0]->data();
    OutDataType* out = outputs[0]->mutable_data();
    bool is_reverse = param._is_reverse;

    std::vector<int> length_vec(offset_vec.size() - 1);
    int batch_size = offset_vec.size() - 1;
    int seqsum = 0;
    int max_seq_len = 0;

    for (int i = 0; i < offset_vec.size() - 1; ++i) {
        int len = offset_vec[i + 1] - offset_vec[i];
        length_vec[i] = len;
        max_seq_len = max_seq_len > len ? max_seq_len : len;
        seqsum += len;
    }

    _temp_wx.try_expand_size(seqsum * 3 * _hidden_size);
    _temp_wh.try_expand_size(batch_size * 2 * _hidden_size);
    _temp_whr.try_expand_size(batch_size * _hidden_size);

    OutDataType* temp_wh = _temp_wh.mutable_data();
    OutDataType* temp_wx = _temp_wx.mutable_data();
    OutDataType* temp_whr = _temp_whr.mutable_data();
    /////////////////////////////////////////////////
    //wx
    gemm(false, false, seqsum, 3 * _hidden_size, _word_size, 1.f, x, weight_w, 0.f, temp_wx);
    //    for(float i :_temp_WX){
    //        cout<<" "<<i;
    //    }

    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;
    const __m256* b_r = (__m256*)(bias + r_offset * _hidden_size);
    const __m256* b_z = (__m256*)(bias + z_offset * _hidden_size);
    const __m256* b_o = (__m256*)(bias + o_offset * _hidden_size);

    int mod_num = _hidden_size % 8;

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        int batch_offset = offset_vec[batch_id];
        int batch_length = length_vec[batch_id];

        for (int seq_id_in_batch = 0; seq_id_in_batch < length_vec[batch_id]; ++seq_id_in_batch) {
            int seqid = batch_offset + seq_id_in_batch;
            int last_seq_id = seqid - 1;

            if (is_reverse) {
                seqid = batch_offset + batch_length - 1 - seq_id_in_batch;
                last_seq_id = seqid + 1;
            }

            const float* hin;
            float* hout = seqid * _hidden_size + out;

            if (seq_id_in_batch == 0) {
                hin = h_init + batch_id * _hidden_size;

            } else {
                hin = out + last_seq_id * _hidden_size;
            }

            //wh
            gemm(false, false, 1, 2 * _hidden_size, _hidden_size, 1.0, hin,
                 weight_h + _hidden_size * _hidden_size,
                 0.f, temp_wh);


            __m256 r;
            __m256 z;
            __m256 _h;

            __m256* hout_256 = (__m256*) hout;
            const __m256* hin_256 = (__m256*) hin;

            __m256* w_x_r = (__m256*)(temp_wx + r_offset * _hidden_size
                                      + seqid * _hidden_size * 3);
            __m256* w_x_z = (__m256*)(temp_wx + z_offset * _hidden_size
                                      + seqid * _hidden_size * 3);
            __m256* w_x_o = (__m256*)(temp_wx + o_offset * _hidden_size
                                      + seqid * _hidden_size * 3);

            __m256* w_h_r = (__m256*)(temp_wh + 0 * _hidden_size);
            __m256* w_h_z = (__m256*)(temp_wh + 1 * _hidden_size);

            for (int frame_id = 0; frame_id < _hidden_size / 8; ++frame_id) {
                r = w_x_r[frame_id] + w_h_r[frame_id] + b_r[frame_id]; //h_out=gate_r
                r = Sigmoid(r);
                //            printit("wxr ",&r);
                hout_256[frame_id] = r * hin_256[frame_id];
            }

            //        cout << "hout = " << hout[0] << endl;
            gemm(false, false, 1, _hidden_size, _hidden_size, 1.0, hout, weight_h, 0.f, temp_whr);

            __m256* temp_wrh_256 = (__m256*) temp_whr;

            for (int frame_id = 0; frame_id < _hidden_size / 8; ++frame_id) {
                z = Sigmoid(w_x_z[frame_id] + w_h_z[frame_id] + b_z[frame_id]);
                _h = w_x_o[frame_id] + temp_wrh_256[frame_id] + b_o[frame_id];

                _h = Tanh(_h);

                hout_256[frame_id] = (1 - z) * hin_256[frame_id] + z * _h;
            }

        }
    }
};

template<>
SaberStatus SaberGru<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::
naiv_256_s_aligned(const std::vector<DataTensor_in*>& inputs,
                   std::vector<DataTensor_out*>& outputs,
                   GruParam<OpTensor>& param) {
    CHECK_NE(param._formula, GRU_CUDNN) << "X86 gru not support cudnn formula now";
    const OpDataType* weight_h = _aligned_weights_h2h.data();
    const OpDataType* weight_w = _aligned_weights_i2h.data();
    const OpDataType* bias = _aligned_weights_bias.data();

    std::vector<int> offset_vec = inputs[0]->get_seq_offset();
    std::vector<int> length_vec(offset_vec.size() - 1);
    int batch_size = offset_vec.size() - 1;
    int seqsum = 0;
    int max_seq_len = 0;
    bool is_hw2seq = offset_vec.size() > 2;
    int word_sum = is_hw2seq ? offset_vec[offset_vec.size() - 1] : inputs[0]->channel();
    __m256 (*gate_act)(const __m256)=act_funcs[param._gate_activity];
    __m256 (*hid_act)(const __m256)=act_funcs[param._h_activity];
    utils::AlignedUtils aligned_utils;
    utils::VectorPrint vector_print;
    const OutDataType* h_init = nullptr;

    if (inputs.size() > 1) {
        h_init = inputs[1]->data();
        _aligned_init_hidden.try_expand_size(batch_size * _aligned_hidden_size);
        aligned_utils.aligned_last_dim(h_init, _aligned_init_hidden.mutable_data(),
                                       batch_size * _hidden_size, _hidden_size, _aligned_hidden_size);
        h_init = _aligned_init_hidden.data();
    } else if (param.init_hidden() != nullptr) {
        h_init = param.init_hidden()->data();
        //FIXME:is it correct
    } else {
        _aligned_init_hidden.try_expand_size(batch_size * _aligned_hidden_size);
        h_init = _aligned_init_hidden.data();
    }

    const InDataType* x = inputs[0]->data();
    OutDataType* out = outputs[0]->mutable_data();
    bool is_reverse = param._is_reverse;

    for (int i = 0; i < offset_vec.size() - 1; ++i) {
        int len = offset_vec[i + 1] - offset_vec[i];
        length_vec[i] = len;
        max_seq_len = max_seq_len > len ? max_seq_len : len;
        seqsum += len;
    }

    _temp_wx.try_expand_size(seqsum * 3 * _aligned_hidden_size);
    _temp_wh.try_expand_size(batch_size * 2 * _aligned_hidden_size);
    _temp_whr.try_expand_size(batch_size * _aligned_hidden_size);
    _temp_out.try_expand_size(seqsum * _aligned_hidden_size);
    OutDataType* temp_wh = _temp_wh.mutable_data();
    OutDataType* temp_wx = _temp_wx.mutable_data();
    OutDataType* temp_whr = _temp_whr.mutable_data();
    /////////////////////////////////////////////////
    //wx
    gemm(false, false, seqsum, 3 * _aligned_hidden_size, _word_size, 1.f, x, weight_w, 0.f, temp_wx);
    //    for(float i :_temp_WX){
    //        cout<<" "<<i;
    //    }

    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;
    const __m256* b_r = (__m256*)(bias + r_offset * _aligned_hidden_size);
    const __m256* b_z = (__m256*)(bias + z_offset * _aligned_hidden_size);
    const __m256* b_o = (__m256*)(bias + o_offset * _aligned_hidden_size);

    int mod_num = _hidden_size % 8;

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        int batch_offset = offset_vec[batch_id];
        int batch_length = length_vec[batch_id];

        for (int seq_id_in_batch = 0; seq_id_in_batch < length_vec[batch_id]; ++seq_id_in_batch) {
            int seqid = batch_offset + seq_id_in_batch;
            int last_seq_id = seqid - 1;

            if (is_reverse) {
                seqid = batch_offset + batch_length - 1 - seq_id_in_batch;
                last_seq_id = seqid + 1;
            }

            const float* hin;
            float* hout = seqid * _aligned_hidden_size + _temp_out.mutable_data();

            if (seq_id_in_batch == 0) {
                hin = h_init + batch_id * _aligned_hidden_size;
            } else {
                hin = _temp_out.mutable_data() + last_seq_id * _aligned_hidden_size;
            }

            //wh
            gemm(false, false, 1, 2 * _aligned_hidden_size, _hidden_size, 1.0, hin,
                 weight_h + _hidden_size * _aligned_hidden_size,
                 0.f, temp_wh);

            __m256 r;
            __m256 z;
            __m256 _h;

            __m256* hout_256 = (__m256*) hout;
            const __m256* hin_256 = (__m256*) hin;

            __m256* w_x_r = (__m256*)(temp_wx + r_offset * _aligned_hidden_size
                                      + seqid * _aligned_hidden_size * 3);
            __m256* w_x_z = (__m256*)(temp_wx + z_offset * _aligned_hidden_size
                                      + seqid * _aligned_hidden_size * 3);
            __m256* w_x_o = (__m256*)(temp_wx + o_offset * _aligned_hidden_size
                                      + seqid * _aligned_hidden_size * 3);

            __m256* w_h_r = (__m256*)(temp_wh + 0 * _aligned_hidden_size);
            __m256* w_h_z = (__m256*)(temp_wh + 1 * _aligned_hidden_size);

            for (int frame_id = 0; frame_id < _aligned_hidden_size / 8; ++frame_id) {
                r = w_x_r[frame_id] + w_h_r[frame_id] + b_r[frame_id]; //h_out=gate_r
                r = gate_act(r);
                //                vector_print.print_float(&r);

                //            printit("wxr ",&r);
                hout_256[frame_id] = r * hin_256[frame_id];
            }

            //        cout << "hout = " << hout[0] << endl;
            gemm(false, false, 1, _aligned_hidden_size, _hidden_size, 1.0, hout, weight_h, 0.f, temp_whr);

            __m256* temp_wrh_256 = (__m256*) temp_whr;

            for (int frame_id = 0; frame_id < _aligned_hidden_size / 8; ++frame_id) {
                z = gate_act(w_x_z[frame_id] + w_h_z[frame_id] + b_z[frame_id]);
                _h = w_x_o[frame_id] + temp_wrh_256[frame_id] + b_o[frame_id];

                _h = hid_act(_h);

                hout_256[frame_id] = (1 - z) * hin_256[frame_id] + z * _h;
            }

        }
    }

    aligned_utils.unaligned_last_dim(_temp_out.data(), out, seqsum * _hidden_size, _hidden_size,
                                     _aligned_hidden_size);
};

template<>
SaberStatus SaberGru<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::
batch_256_s_aligned(const std::vector<DataTensor_in*>& inputs,
                    std::vector<DataTensor_out*>& outputs,
                    GruParam<OpTensor>& param) {
    CHECK_NE(param._formula, GRU_CUDNN) << "X86 gru not support cudnn formula now";
    const OpDataType* weight_h = _aligned_weights_h2h.data();
    const OpDataType* weight_w = _aligned_weights_i2h.data();
    const OpDataType* bias = _aligned_weights_bias.data();
    __m256 (*gate_act)(const __m256)=act_funcs[param._gate_activity];
    __m256 (*hid_act)(const __m256)=act_funcs[param._h_activity];
    std::vector<int> offset_vec = inputs[0]->get_seq_offset();
    std::vector<int> length_vec(offset_vec.size() - 1);
    int batch_size = offset_vec.size() - 1;
    int seqsum = 0;
    int max_seq_len = 0;
    bool is_hw2seq = offset_vec.size() > 2;
    int word_sum = is_hw2seq ? offset_vec[offset_vec.size() - 1] : inputs[0]->channel();
    utils::AlignedUtils aligned_utils;
    utils::VectorPrint vector_print;
    const OutDataType* h_init = nullptr;

    const InDataType* x = inputs[0]->data();
    OutDataType* out = outputs[0]->mutable_data();
    bool is_reverse = param._is_reverse;

    if (inputs.size() > 1) {
        h_init = inputs[1]->data();
        _aligned_init_hidden.try_expand_size(batch_size * _aligned_hidden_size);
        aligned_utils.aligned_last_dim(h_init, _aligned_init_hidden.mutable_data(),
                                       batch_size * _hidden_size, _hidden_size, _aligned_hidden_size);
        h_init = _aligned_init_hidden.data();
    } else if (param.init_hidden() != nullptr) {
        h_init = param.init_hidden()->data();
        //FIXME:is it correct?
    } else {
        _aligned_init_hidden.try_expand_size(batch_size * _aligned_hidden_size);
        h_init = _aligned_init_hidden.data();
    }

    std::vector<int> emit_offset_vec;
    int emit_length = 0;
    utils::SeqSortedseqTranseUtil transe_util(is_reverse);
    bool transform = transe_util.get_sorted_map(offset_vec, emit_offset_vec, emit_length);

    float* inner_h_out = out;
    float* inner_x = x;
    const float* inner_h_init = h_init;

    for (int i = 0; i < offset_vec.size() - 1; ++i) {
        int len = offset_vec[i + 1] - offset_vec[i];
        length_vec[i] = len;
        max_seq_len = max_seq_len > len ? max_seq_len : len;
        seqsum += len;
    }

    _temp_wx.try_expand_size(seqsum * 3 * _aligned_hidden_size);
    _temp_wh.try_expand_size(batch_size * 2 * _aligned_hidden_size);
    _temp_whr.try_expand_size(batch_size * _aligned_hidden_size);
    _temp_out.try_expand_size(seqsum * _aligned_hidden_size * param._num_direction);

    if (transform) {
        _temp_x.try_expand_size(seqsum * _word_size);
        inner_h_out = _temp_out.mutable_data();
        inner_x = _temp_x.mutable_data();
        transe_util.seq_2_sorted_seq(x, inner_x, _word_size);

        if (inner_h_init != nullptr) {
            _temp_h_init.try_expand_size(batch_size * _aligned_hidden_size);
            transe_util.hidden_2_sorted_hidden(inner_h_init, _temp_h_init.mutable_data(), _aligned_hidden_size);
            inner_h_init = _temp_h_init.data();
        }

    } else if (_hidden_size != _aligned_hidden_size) {
        inner_h_out = _temp_out.mutable_data();
    }

    OutDataType* temp_wh = _temp_wh.mutable_data();
    OutDataType* temp_wx = _temp_wx.mutable_data();
    OutDataType* temp_whr = _temp_whr.mutable_data();
    /////////////////////////////////////////////////
    //wx
    gemm(false, false, seqsum, 3 * _aligned_hidden_size, _word_size, 1.f, inner_x, weight_w, 0.f,
         temp_wx);

    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;
    const __m256* b_r = (__m256*)(bias + r_offset * _aligned_hidden_size);
    const __m256* b_z = (__m256*)(bias + z_offset * _aligned_hidden_size);
    const __m256* b_o = (__m256*)(bias + o_offset * _aligned_hidden_size);

    int mod_num = _hidden_size % 8;

    int reverse_out_offset=seqsum;


    for (int word_id = 0; word_id < emit_length; word_id++) {
        int real_word_id = word_id;
        int last_word_id = word_id - 1;

        if (param._is_reverse&&batch_size==1) {
            real_word_id = emit_length - word_id - 1;
            last_word_id = real_word_id + 1;
        }

        int emit_word_id_start = emit_offset_vec[real_word_id];
        int emit_word_id_end = emit_offset_vec[real_word_id + 1];
        int emit_word_length = emit_word_id_end - emit_word_id_start;
        const float* hin;

        if (word_id == 0) {
            hin = inner_h_init;
        } else {
//            if(is_reverse){
//                hin = inner_h_out + reverse_out_offset * _aligned_hidden_size;
//            }else{
//                hin = inner_h_out + emit_offset_vec[last_word_id] * _aligned_hidden_size;
//            }
            hin = inner_h_out + emit_offset_vec[last_word_id] * _aligned_hidden_size;
        }

        float* hout = nullptr;
//        if(is_reverse){
//            reverse_out_offset-=emit_word_length;
//            hout=reverse_out_offset*_aligned_hidden_size + inner_h_out;
//        } else{
            hout=emit_offset_vec[real_word_id] * _aligned_hidden_size + inner_h_out;
//        }
        //wh
        gemm(false, false, emit_word_length, 2 * _aligned_hidden_size, _aligned_hidden_size, 1.0, hin,
             weight_h + _hidden_size * _aligned_hidden_size,
             0.f, temp_wh);

        volatile __m256 r;
        volatile __m256 z;
        volatile __m256 _h;
        __m256* hout_256 = (__m256*) hout;
        const __m256* hin_256 = (__m256*) hin;

        for (int emit_word_id = emit_word_id_start; emit_word_id < emit_word_id_end; emit_word_id++) {
            int emit_id_offset = emit_word_id - emit_word_id_start;
            __m256* w_x_r = (__m256*)(temp_wx + r_offset * _aligned_hidden_size
                                      + emit_word_id * _aligned_hidden_size * 3);
            __m256* w_h_r = (__m256*)(temp_wh + 0 * _aligned_hidden_size
                                      + emit_id_offset * _aligned_hidden_size * 2);
            __m256* emit_hout = (__m256*)(hout + emit_id_offset * _aligned_hidden_size);
            const __m256* emit_hin = (__m256*)(hin + emit_id_offset * _aligned_hidden_size);

            for (int frame_id = 0; frame_id < _aligned_hidden_size / 8; ++frame_id) {
                r = w_x_r[frame_id] + w_h_r[frame_id] + b_r[frame_id]; //h_out=gate_r
                r = gate_act(r);

                emit_hout[frame_id] = r * emit_hin[frame_id];
            }

        }

        //        cout << "hout = " << hout[0] << endl;
        gemm(false, false, emit_word_length, _aligned_hidden_size, _aligned_hidden_size, 1.0, hout,
             weight_h, 0.f, temp_whr);


        for (int emit_word_id = emit_word_id_start; emit_word_id < emit_word_id_end; emit_word_id++) {
            int emit_offset = emit_word_id - emit_word_id_start;
            __m256* w_x_z = (__m256*)(temp_wx + z_offset * _aligned_hidden_size
                                      + emit_word_id * _aligned_hidden_size * 3);
            __m256* w_x_o = (__m256*)(temp_wx + o_offset * _aligned_hidden_size
                                      + emit_word_id * _aligned_hidden_size * 3);

            __m256* w_h_z = (__m256*)(temp_wh + 1 * _aligned_hidden_size
                                      + emit_offset * _aligned_hidden_size * 2);

            __m256* w_h_o = (__m256*)(temp_whr + emit_offset * _aligned_hidden_size);
            __m256* emit_hout = (__m256*)(hout + emit_offset * _aligned_hidden_size) ;
            const __m256* emit_hin = (__m256*)(hin + emit_offset * _aligned_hidden_size) ;

            for (int frame_id = 0; frame_id < _aligned_hidden_size / 8; ++frame_id) {

                z = gate_act(w_x_z[frame_id] + w_h_z[frame_id] + b_z[frame_id]);
                _h = w_x_o[frame_id] + w_h_o[frame_id] + b_o[frame_id];
                _h = hid_act(_h);
                //                vector_print.print_float(&z);
                emit_hout[frame_id] = (1 - z) * emit_hin[frame_id] + z * _h;
            }
        }

    }

    if (transform){
        transe_util.sorted_seq_2_seq(inner_h_out, out, _hidden_size, _aligned_hidden_size);
    }else if (_hidden_size != _aligned_hidden_size) {
        aligned_utils.unaligned_last_dim(_temp_out.data(), out, seqsum * _hidden_size, _hidden_size,
                                         _aligned_hidden_size);
    }
    return SaberSuccess;
};
template<>
SaberStatus SaberGru<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        GruParam<OpTensor>& param) {
//        return batch_gru(inputs, outputs, param);
    if(inputs[0]->get_seq_offset().size()>2) {
        return batch_256_s_aligned(inputs, outputs, param);
    }else {
        return naiv_256_s_aligned(inputs, outputs, param);
    }
//        return naiv_256_s_aligned(inputs, outputs, param);
//            return naiv_gru(inputs, outputs, param);
};

template class SaberGru<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}
}