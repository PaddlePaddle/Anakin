
#include "saber/funcs/impl/x86/saber_crf_decoding.h"
#include "saber/saber_funcs_param.h"
#include "x86_utils.h"
#include <cstring>
#include <limits>
#include <cmath>
#include <immintrin.h>

namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberCrfDecoding<X86, OpDtype>::init(
        const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        CrfDecodingParam<X86> &param, Context<X86> &ctx) {
            
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberCrfDecoding<X86, OpDtype>::create(
        const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        CrfDecodingParam<X86> &param,
        Context<X86> &ctx) {

    CHECK_EQ(inputs[0]->get_dtype(), OpDtype) << "inputs data type should be same with OpDtype";
    CHECK_EQ(outputs[0]->get_dtype(), OpDtype) << "outputs data type should be same with OpDtype";
    
    this->_ctx = &ctx;
    _track.re_alloc(inputs[0]->valid_shape(), AK_INT32);

#ifdef __AVX2__
    int tag_num = inputs[0]->channel();
    _aligned_tag_num = (tag_num % 8) ? (tag_num / 8 + 1) * 8 : tag_num;
    // get transposed transition weight
    const OpDataType *transition_ptr = (const OpDataType*)param.transition_weight()->data();
    Shape trans_shape({tag_num + 2, _aligned_tag_num, 1, 1}, Layout_NCHW);
    _trans.re_alloc(trans_shape, OpDtype);
    OpDataType *transition = (OpDataType*)_trans.mutable_data();
    memcpy(transition, transition_ptr, sizeof(OpDataType) * tag_num);
    memcpy(transition + _aligned_tag_num, transition_ptr + tag_num, sizeof(OpDataType) * tag_num);
    for (int i = 0; i < tag_num; i++) {
        for (int j = 0; j < tag_num; j++) {
            transition[(i + 2) * _aligned_tag_num + j] = transition_ptr[(j + 2) * tag_num + i];
        }
        for (int j = tag_num; j < _aligned_tag_num; j++) {
            transition[(i + 2) * _aligned_tag_num + j] = 0;
        }
    }

    Shape emis_shape({inputs[0]->num(), _aligned_tag_num, 1, 1}, Layout_NCHW);
    _emis.re_alloc(emis_shape, OpDtype);
    _alpha.re_alloc(emis_shape, OpDtype);
#else
    _alpha.re_alloc(inputs[0]->valid_shape(), OpDtype);
#endif
    return SaberSuccess;
}

template<typename Dtype>
void decoding(Dtype* path, const Dtype* emission, const Dtype* transition,
                        Dtype* alpha_value, int* track_value, int aligned_tag_num, int seq_len, int tag_num) {
#ifdef __AVX2__
    const Dtype* x = emission;
    const Dtype* w = transition;
    const int state_trans_base_idx = 2;

    {
        __m256 *ww = (__m256*)w;
        __m256 *xx = (__m256*)x;
        __m256 *aa = (__m256*)alpha_value;
        for (int i = 0; i < aligned_tag_num / 8; ++i) {
            aa[i] = ww[i] + xx[i];
        }
    }

    int tail = ((aligned_tag_num == tag_num) ? 8 : tag_num % 8); 

    for (int k = 1; k < seq_len; ++k) {
        for (int i = 0; i < tag_num; ++i) {
            Dtype max_score = -std::numeric_limits<Dtype>::max();
            int max_j = 0;

            __m256 *aa = (__m256*)(alpha_value + (k - 1) * aligned_tag_num);
            __m256 *ww = (__m256*)(w + (i + state_trans_base_idx) * aligned_tag_num);
            __m256 score_v;
            Dtype *score = (Dtype*)(&score_v);
            for (size_t j = 0; j < aligned_tag_num / 8 - 1; ++j) {
                score_v = aa[j] + ww[j];
                for (int m = 0; m < 8; m++) {
                    if (score[m] > max_score) {
                        max_score = score[m];
                        max_j = j * 8 + m;
                    }
                }
            }
            int tail_idx = aligned_tag_num / 8 - 1;
            score_v = aa[tail_idx] + ww[tail_idx];
            for (int m = 0; m < tail; m++) {
                if (score[m] > max_score) {
                    max_score = score[m];
                    max_j = tail_idx * 8 + m;
                }
            }

            alpha_value[k * aligned_tag_num + i] = max_score + x[k * aligned_tag_num + i];
            track_value[k * tag_num + i] = max_j;
        }
    }

    Dtype max_score = -std::numeric_limits<Dtype>::max();
    int max_i = 0;
    __m256* aa = (__m256*)(alpha_value + (seq_len - 1) * aligned_tag_num);
    __m256* ww = (__m256*)(w + aligned_tag_num);
    __m256 score_v;
    Dtype *score = (Dtype*)(&score_v);
    for (size_t i = 0; i < aligned_tag_num / 8 - 1; ++i) {
        score_v = aa[i] + ww[i];
        for (int m = 0; m < 8; m++) {
            if (score[m] > max_score) {
                max_score = score[m];
                max_i = i * 8 + m;
            }
        }
    }
    int tail_idx = aligned_tag_num / 8 - 1;
    score_v = aa[tail_idx] + ww[tail_idx];
    for (int m = 0; m < tail; m++) {
        if (score[m] > max_score) {
            max_score = score[m];
            max_i = tail_idx * 8 + m;
        }
    }

    path[seq_len - 1] = max_i;
    for (int k = seq_len - 1; k >= 1; --k) {
        path[k - 1] = max_i = track_value[k * tag_num + max_i];
    }
#else
    const Dtype* x = emission;
    const Dtype* w = transition;
    const int state_trans_base_idx = 2;

    for (int i = 0; i < tag_num; ++i) alpha_value[i] = w[i] + x[i];

    for (int k = 1; k < seq_len; ++k) {
        for (int i = 0; i < tag_num; ++i) {
            Dtype max_score = -std::numeric_limits<Dtype>::max();
            int max_j = 0;
            for (size_t j = 0; j < tag_num; ++j) {
                Dtype score = alpha_value[(k - 1) * tag_num + j] +
                          w[(j + state_trans_base_idx) * tag_num + i];
                if (score > max_score) {
                    max_score = score;
                    max_j = j;
                }
            }
            alpha_value[k * tag_num + i] = max_score + x[k * tag_num + i];
            track_value[k * tag_num + i] = max_j;
        }
    }
    Dtype max_score = -std::numeric_limits<Dtype>::max();
    int max_i = 0;
    for (size_t i = 0; i < tag_num; ++i) {
        Dtype score = alpha_value[(seq_len - 1) * tag_num + i] + w[tag_num + i];
        if (score > max_score) {
            max_score = score;
            max_i = i;
        }
    }
    path[seq_len - 1] = max_i;
    for (int k = seq_len - 1; k >= 1; --k) {
        path[k - 1] = max_i = track_value[k * tag_num + max_i];
    }
#endif
}

template <DataType OpDtype>
SaberStatus SaberCrfDecoding<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        CrfDecodingParam<X86> &param) {

    std::vector<std::vector<int>> seq_offset = inputs[0]->get_seq_offset();

    const OpDataType *emission_ptr = (const OpDataType*)inputs[0]->data();
    int tag_num = inputs[0]->channel();
    const OpDataType *transition_ptr = (const OpDataType*)param.transition_weight()->data();
    int slice_size = inputs[0]->channel() * inputs[0]->height() * inputs[0]->width();
   
#ifdef __AVX2__
    if (tag_num % 8) {
        transition_ptr = (OpDataType*)_trans.data();

        // align emission to AVX2 register width
        OpDataType *emission = (OpDataType*)_emis.mutable_data();
        for (int i = 0; i < inputs[0]->num(); i++) {
          OpDataType* to = emission + i * _aligned_tag_num;
          OpDataType* from = emission_ptr + i * tag_num;
          memcpy(to, from, tag_num * sizeof(OpDataType));
          for (int j = tag_num; j < _aligned_tag_num; j++) {
              to[j] = 0;
          }
        }
        emission_ptr = emission;
        slice_size = _aligned_tag_num;
    }
#endif
    OpDataType *decoded_path = (OpDataType*) outputs[0]->mutable_data();
    int seq_num = seq_offset[0].size() - 1;
    int nthreads = omp_get_max_threads();

    if (nthreads > seq_num) {
        nthreads = seq_num;
    }
    #pragma omp parallel for num_threads(nthreads) if(seq_num > 1)
    for (int i = 0; i < seq_num; ++i) {
        int seq_len = seq_offset[0][i+1] - seq_offset[0][i];
       // LOG(INFO) << "slice_size: " << slice_size << ", seq_num: " << seq_num << ", seq_len: " << seq_len;
        decoding<OpDataType>(decoded_path, emission_ptr, transition_ptr,
                 (OpDataType*)_alpha.mutable_data(), (int*)_track.mutable_data(),
                 _aligned_tag_num, seq_len, tag_num);

        decoded_path += seq_len;
        emission_ptr += slice_size * seq_len;
    }
    //LOG(INFO) << "dispatch success ";
    return SaberSuccess;
}

template class SaberCrfDecoding<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberCrfDecoding, CrfDecodingParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberCrfDecoding, CrfDecodingParam, X86, AK_INT8);
}
} // namespace anakin
