#include "mkl_cblas.h"
#include "mkl_vml_functions.h"

#include "saber/funcs/impl/x86/vender_gru.h"
#include "sequence2batch.h"
#include "saber/funcs/impl/x86/activation_functions.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/funcs/impl/x86/kernel/jit_generator.h"

namespace anakin {
namespace saber {

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus VenderGru<X86, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>::init(
                const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
GruParam<OpTensor>& param, Context<X86>& ctx) {
    this->_ctx = ctx;
    this->max_thread_num_ = omp_get_max_threads();
    hidden_size_ = outputs[0]->channel();
    word_size_ = inputs[0]->channel();

    int aligned_size = 8;
    aligned_hidden_size_ = (hidden_size_ % aligned_size) ? ((hidden_size_ / aligned_size) + 1) *
                           aligned_size : hidden_size_;

    avx2_available_ = jit::mayiuse(jit::avx2);
    // LOG(ERROR) << "AVX2 available: " << avx2_available_;

    if (param.formula == GRU_ORIGIN) {
        OpDataType* weights_data = const_cast<float*>(param.weight()->data());

        OpDataType* wx = weights_data;
        OpDataType* wch = wx + word_size_ * hidden_size_ * 3;
        OpDataType* wh = wch + hidden_size_ * hidden_size_;

        OpDataType* aligned_wx = nullptr;
        OpDataType* aligned_wch = nullptr;
        OpDataType* aligned_wh = nullptr;

        int delta = aligned_hidden_size_ - hidden_size_;


        if (aligned_bias_ == nullptr) {
            aligned_bias_ = (OpDataType*)zmalloc(3 * aligned_hidden_size_ * sizeof(float), 4096);
            const OpDataType* bias_data = param.bias()->data();

            for (int i = 0; i < 3; i++) {
                memcpy(aligned_bias_ + i * aligned_hidden_size_, bias_data + i * hidden_size_,
                       hidden_size_ * sizeof(float));

                if (delta > 0) {
                    memset(aligned_bias_ + i * aligned_hidden_size_ + hidden_size_, 0, delta * sizeof(float));
                }
            }
        } else {
            LOG(ERROR) << "aligned bias in init should not be a non-nullptr";
        }


        if (delta > 0) {
            aligned_wx = (OpDataType*)zmalloc(word_size_ * aligned_hidden_size_ * 3 * sizeof(float), 4096);
            aligned_wch = (OpDataType*)zmalloc(aligned_hidden_size_ * aligned_hidden_size_ * sizeof(float),
                                               4096);
            aligned_wh = (OpDataType*)zmalloc(2 * aligned_hidden_size_ * aligned_hidden_size_ * sizeof(float),
                                              4096);

            for (int i = 0; i < word_size_; i++) {
                float* aligned_row = aligned_wx + i * aligned_hidden_size_ * 3;
                float* row = wx + i * hidden_size_ * 3;

                for (int j = 0; j < 3; j++) {
                    memcpy(aligned_row + j * aligned_hidden_size_, row + j * hidden_size_,
                           hidden_size_ * sizeof(float));
                    memset(aligned_row + j * aligned_hidden_size_ + hidden_size_, 0, delta * sizeof(float));
                }
            }

            for (int i = 0; i < aligned_hidden_size_; i++) {
                float* aligned_row = aligned_wch + i * aligned_hidden_size_;
                float* row = wch + i * hidden_size_;

                if (i < hidden_size_) {
                    memcpy(aligned_row, row, hidden_size_ * sizeof(float));
                    memset(aligned_row + hidden_size_, 0, delta * sizeof(float));
                } else {
                    memset(aligned_row, 0, aligned_hidden_size_ * sizeof(float));
                }
            }

            for (int i = 0; i < aligned_hidden_size_; i++) {
                float* aligned_row = aligned_wh + i * aligned_hidden_size_ * 2;
                float* row = wh + i * hidden_size_ * 2;

                if (i < hidden_size_) {
                    for (int j = 0; j < 2; j++) {
                        memcpy(aligned_row + j * aligned_hidden_size_, row + j * hidden_size_,
                               hidden_size_ * sizeof(float));
                        memset(aligned_row + j * aligned_hidden_size_ + hidden_size_, 0, delta * sizeof(float));
                    }
                } else {
                    memset(aligned_row, 0, 2 * aligned_hidden_size_ * sizeof(float));
                }
            }
        } else {
            aligned_wx = wx;
            aligned_wch = wch;
            aligned_wh = wh;
        }

        if (weight_x_packed_) {
            cblas_sgemm_free(weight_x_packed_);
            weight_x_packed_ = nullptr;
        }

        if (weight_ru_packed_) {
            cblas_sgemm_free(weight_ru_packed_);
            weight_ru_packed_ = nullptr;
        }

        if (weight_c_packed_) {
            cblas_sgemm_free(weight_c_packed_);
            weight_c_packed_ = nullptr;
        }

        weight_x_packed_ = cblas_sgemm_alloc(CblasBMatrix, inputs[0]->num(), 3 * aligned_hidden_size_,
                                             word_size_);

        if (!weight_x_packed_) {
            LOG(ERROR) << "cannot alloc weight_x_packed_ for gru";
            return SaberOutOfMem;
        }

        cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, inputs[0]->num(),
                         3 * aligned_hidden_size_, word_size_, 1.0,
                         aligned_wx, 3 * aligned_hidden_size_, weight_x_packed_);

        weight_ru_packed_ = cblas_sgemm_alloc(CblasBMatrix, 1, 2 * aligned_hidden_size_,
                                              aligned_hidden_size_);

        if (!weight_ru_packed_) {
            LOG(ERROR) << "cannot alloc weight_ru_packed_ for gru";
            return SaberOutOfMem;
        }

        cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, 1, 2 * aligned_hidden_size_,
                         aligned_hidden_size_, 1.0,
                         aligned_wh, 2 * aligned_hidden_size_, weight_ru_packed_);

        weight_c_packed_ = cblas_sgemm_alloc(CblasBMatrix, 1, aligned_hidden_size_, aligned_hidden_size_);

        if (!weight_c_packed_) {
            LOG(ERROR) << "cannot alloc weight_c_packed_ for gru";
            return SaberOutOfMem;
        }

        cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, 1, aligned_hidden_size_,
                         aligned_hidden_size_, 1.0,
                         aligned_wch, aligned_hidden_size_, weight_c_packed_);

        if (delta > 0) {
            zfree(aligned_wx);
            wx = nullptr;
            zfree(aligned_wh);
            wh = nullptr;
            zfree(aligned_wch);
            wch = nullptr;
        }
    } else {
        LOG(ERROR) << "only support GRU_ORIGIN now";
        return SaberUnImplError;
    }

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus VenderGru<X86, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>::create(
                const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
                GruParam<OpTensor>& param,
Context<X86>& ctx) {
    batched_h.try_expand_size(inputs[0]->num() * aligned_hidden_size_ * param.num_direction);
    batched_x.try_expand_size(inputs[0]->num() * word_size_);
    batched_xx.try_expand_size(inputs[0]->num() * 3 * aligned_hidden_size_);

    return SaberSuccess;
}

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus VenderGru<X86, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
                const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
GruParam<OpTensor>& param) {

    const OpDataType* bias = param.bias()->data();
    std::vector<int> seq_offset = inputs[0]->get_seq_offset();
    int word_sum = inputs[0]->num();
    const InDataType* x = inputs[0]->data();
    OutDataType* out = outputs[0]->mutable_data();
    bool is_reverse = param.is_reverse;
    int batch_size = seq_offset.size() - 1;

    batched_h.try_expand_size(inputs[0]->num() * aligned_hidden_size_ * param.num_direction);
    OutDataType* batched_h_data = batched_h.mutable_data();
    batched_x.try_expand_size(inputs[0]->num() * word_size_);
    batched_xx.try_expand_size(inputs[0]->num() * 3 * aligned_hidden_size_);
    InDataType* batched_xx_data = batched_xx.mutable_data();

    // input sequence to batch
    math::SequenceToBatch batch_value;
    batch_value.create_batch(word_sum, batch_size, seq_offset, is_reverse);

    int bat_length = batch_value.get_batch_num();
    std::vector<int> bat_offset(bat_length + 1);
    batch_value.get_batch_offset(bat_offset);
    batch_value.seq_2_bat(x, batched_x.mutable_data(), word_size_);

    int delta = aligned_hidden_size_ - hidden_size_;

    // init h
    Shape h_init_shape(batch_size, aligned_hidden_size_, 1, 1);
    aligned_init_hidden.try_expand_size(h_init_shape);
    const OutDataType* h0 = nullptr;

    if (param.init_hidden() != nullptr) {
        CHECK_EQ(param.init_hidden()->valid_shape().count(),
                 batch_size * hidden_size_) << "hidden init must match batch size";
        h0 = param.init_hidden()->data();
        OpTensor h_init_tmp(h_init_shape);
        float* aligned_init = h_init_tmp.mutable_data();
        int delta = aligned_hidden_size_ - hidden_size_;

        if (delta > 0) {
            for (int i = 0; i < batch_size; i++) {
                float* aligned_row = aligned_init + i * aligned_hidden_size_;
                const float* row = h0 + i * hidden_size_;
                memcpy(aligned_row, row, hidden_size_ * sizeof(float));
                memset(aligned_row + hidden_size_, 0, delta * sizeof(float));
            }

            batch_value.hidden_2_bat(h_init_tmp.data(), aligned_init_hidden.mutable_data(),
                                     aligned_hidden_size_);
            h0 = aligned_init_hidden.data();
        } else {
            batch_value.hidden_2_bat(h0, aligned_init_hidden.mutable_data(), aligned_hidden_size_);
            h0 = aligned_init_hidden.data();
        }
    } else {
        fill_tensor_host_const(aligned_init_hidden, 0);
        h0 = aligned_init_hidden.data();
    }

    // batched_xx = batched_x * [Wcx, Wrx, Wux]
    cblas_sgemm_compute(CblasRowMajor,
                        CblasNoTrans,
                        CblasPacked,
                        word_sum,
                        3 * aligned_hidden_size_,
                        word_size_,
                        batched_x.data(),
                        word_size_,
                        weight_x_packed_,
                        3 * aligned_hidden_size_,
                        0.f,
                        batched_xx_data,
                        3 * aligned_hidden_size_);

    // batched_xx += bias
    int xx_num = inputs[0]->num();
    int hidden_stride = 3 * aligned_hidden_size_;
    #pragma omp parallel for if(this->max_thread_num_ > 1)

    for (int i = 0; i < xx_num; i++) {
        cblas_saxpy(hidden_stride, 1, aligned_bias_, 1, batched_xx_data + i * hidden_stride, 1);
    }

    int c_offset = 0;
    int r_offset = 1;
    int u_offset = 2;

    for (int word_id = 0; word_id < bat_length; word_id++) {
        int bat_word_id_start = bat_offset[word_id];
        int bat_word_id_end = bat_offset[word_id + 1];
        int bat_word_length = bat_word_id_end - bat_word_id_start;
        const float* ht_1;

        if (word_id == 0) {
            ht_1 = h0;
        } else {
            ht_1 = batched_h_data + bat_offset[word_id - 1] * aligned_hidden_size_;
        }

        float* ht = batched_h_data + bat_offset[word_id] * aligned_hidden_size_;

        // xx = xx + ht_1 * Wh
        cblas_sgemm_compute(CblasRowMajor,
                            CblasNoTrans,
                            CblasPacked,
                            bat_word_length,
                            2 * aligned_hidden_size_,
                            aligned_hidden_size_,
                            ht_1,
                            aligned_hidden_size_,
                            weight_ru_packed_,
                            2 * aligned_hidden_size_,
                            1.f,
                            batched_xx_data + bat_word_id_start * hidden_stride + r_offset * aligned_hidden_size_,
                            hidden_stride);

        // compute reset gate output r and rh
        if (avx2_available_) {
            for (int bat_word_id = bat_word_id_start; bat_word_id < bat_word_id_end; bat_word_id++) {
                int intra_bat_offset = bat_word_id - bat_word_id_start;
                __m256* r = (__m256*)(batched_xx_data + bat_word_id * hidden_stride + r_offset *
                                      aligned_hidden_size_);
                __m256* hit = (__m256*)(ht + intra_bat_offset * aligned_hidden_size_);
                __m256* hit_1 = (__m256*)(ht_1 + intra_bat_offset * aligned_hidden_size_);

                for (int i = 0; i < aligned_hidden_size_ / 8; ++i) {
                    r[i] = math::avx_activation(r[i], param.gate_activity);
                    hit[i] = r[i] * hit_1[i];
                }
            }
        } else {
            for (int bat_word_id = bat_word_id_start; bat_word_id < bat_word_id_end; bat_word_id++) {
                int intra_bat_offset = bat_word_id - bat_word_id_start;
                float* r = (float*)(batched_xx_data + bat_word_id * hidden_stride + r_offset *
                                    aligned_hidden_size_);
                float* hit = (float*)(ht + intra_bat_offset * aligned_hidden_size_);
                float* hit_1 = (float*)(ht_1 + intra_bat_offset * aligned_hidden_size_);

                for (int i = 0; i < aligned_hidden_size_; ++i) {
                    math::activation(1, r + i, r + i, param.gate_activity);
                    hit[i] = r[i] * hit_1[i];
                }
            }
        }

        // xx = xx + rh * Wch
        cblas_sgemm_compute(CblasRowMajor,
                            CblasNoTrans,
                            CblasPacked,
                            bat_word_length,
                            aligned_hidden_size_,
                            aligned_hidden_size_,
                            ht,
                            aligned_hidden_size_,
                            weight_c_packed_,
                            aligned_hidden_size_,
                            1.f,
                            batched_xx_data + bat_word_id_start * hidden_stride + c_offset * aligned_hidden_size_,
                            hidden_stride);

        // compute candidate activation output and h
        if (avx2_available_) {
            for (int bat_word_id = bat_word_id_start; bat_word_id < bat_word_id_end; bat_word_id++) {
                int intra_bat_offset = bat_word_id - bat_word_id_start;
                int h_word_id_offset = bat_word_id * hidden_stride;
                __m256* u = (__m256*)(batched_xx_data + h_word_id_offset + u_offset * aligned_hidden_size_);
                __m256* c = (__m256*)(batched_xx_data + h_word_id_offset + c_offset * aligned_hidden_size_);
                __m256* hit = (__m256*)(ht + intra_bat_offset * aligned_hidden_size_);
                __m256* hit_1 = (__m256*)(ht_1 + intra_bat_offset * aligned_hidden_size_);

                for (int i = 0; i < aligned_hidden_size_ / 8; ++i) {
                    u[i] = math::avx_activation(u[i], param.gate_activity);
                    c[i] = math::avx_activation(c[i], param.h_activity);
                    hit[i] = (c[i] - hit_1[i]) * u[i] + hit_1[i];
                }
            }
        } else {
            for (int bat_word_id = bat_word_id_start; bat_word_id < bat_word_id_end; bat_word_id++) {
                int intra_bat_offset = bat_word_id - bat_word_id_start;
                int h_word_id_offset = bat_word_id * hidden_stride;
                float* u = (float*)(batched_xx_data + h_word_id_offset + u_offset * aligned_hidden_size_);
                float* c = (float*)(batched_xx_data + h_word_id_offset + c_offset * aligned_hidden_size_);
                float* hit = (float*)(ht + intra_bat_offset * aligned_hidden_size_);
                float* hit_1 = (float*)(ht_1 + intra_bat_offset * aligned_hidden_size_);

                for (int i = 0; i < aligned_hidden_size_; ++i) {
                    math::activation(1, u + i, u + i, param.gate_activity);
                    math::activation(1, c + i, c + i, param.h_activity);
                    hit[i] = (c[i] - hit_1[i]) * u[i] + hit_1[i];
                }
            }
        }
    }

    // batch to sequence
    batch_value.bat_2_seq(batched_h_data, out, hidden_size_, aligned_hidden_size_);

    return SaberSuccess;
}

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus VenderGru<X86, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>::check_conf(
                const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
GruParam<OpTensor>& param) {
    return SaberSuccess;
}

template class VenderGru<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} // namespace saber
} // namespace anakin
