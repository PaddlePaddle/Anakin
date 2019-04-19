#include "saber/core/tensor_op.h"
#include "saber/funcs/impl/x86/vender_lstm.h"
#include "saber/funcs/impl/x86/sequence2batch.h"
#include "saber/funcs/impl/x86/kernel/jit_generator.h"
#include "saber/funcs/impl/x86/saber_normal_activation.h"

namespace anakin {
namespace saber {


template <>
SaberStatus VenderLstm<X86, AK_FLOAT>::create(
    const std::vector<OpTensor*>& inputs,
    std::vector<OpTensor*>& outputs,
    LstmParam<X86>& param,
    Context<X86>& ctx) {
    utils::try_expand_tensor(batched_h, param.num_direction * param.num_layers * inputs[0]->num() *
                             aligned_hidden_size_);
    utils::try_expand_tensor(batched_c, param.num_direction * param.num_layers * inputs[0]->num() *
                             aligned_hidden_size_);
    utils::try_expand_tensor(batched_x, inputs[0]->num() * word_size_);
    utils::try_expand_tensor(batched_x_reverse, inputs[0]->num() * word_size_);
    utils::try_expand_tensor(batched_xx, param.num_direction * param.num_layers * inputs[0]->num() * 4 *
                             aligned_hidden_size_);
    return SaberSuccess;
}
template <>
SaberStatus VenderLstm<X86, AK_FLOAT>::init(
    const std::vector<OpTensor*>& inputs,
    std::vector<OpTensor*>& outputs,
    LstmParam<X86>& param, Context<X86>& ctx) {
#ifdef USE_SGX
    const char *ret = "1";
#else
    const char* ret = std::getenv("OMP_NUM_THREADS");
#endif
    this->_ctx = &ctx;
    this->max_thread_num_ = ret ? atoi(ret) : anakin_get_max_threads();
    int layer_num_ = param.num_layers;
    int direc_num_ = param.num_direction;
    hidden_size_ = outputs[0]->channel() / direc_num_;
    word_size_ = inputs[0]->channel();
    std::vector<std::vector<int>> seq_offset_vec = inputs[0]->get_seq_offset();
    std::vector<int> seq_offset = seq_offset_vec[seq_offset_vec.size() - 1];
    int batch_size_ = seq_offset.size() - 1;
    int bias_size_ = 4;

    if (param.with_peephole) {
        bias_size_ = 7;
    }

    int aligned_size_ = 8;
    aligned_hidden_size_ = (hidden_size_ % aligned_size_) ?
                           ((hidden_size_ / aligned_size_) + 1) * aligned_size_ : hidden_size_;
    int delta = aligned_hidden_size_ - hidden_size_;
    int Wx_stride_l0 = word_size_ * 4 * hidden_size_;
    int W_stride_l0 = (word_size_ + hidden_size_) * 4 * hidden_size_;

    if (param.skip_input) {
        if ((4 * hidden_size_) != word_size_) {
            LOG(ERROR) << "input width should be 4 * hidden_size in skip input mode";
            return SaberInvalidValue;
        }

        Wx_stride_l0 = 0;
        W_stride_l0 = hidden_size_ * 4 * hidden_size_;
    }

    int Wx_stride_ln = hidden_size_ * 4 * hidden_size_;
    int W_stride_ln = (hidden_size_ + hidden_size_) * 4 * hidden_size_;
    int W_stride = W_stride_l0 + (layer_num_ - 1) * W_stride_ln;
    avx2_available_ = jit::mayiuse(jit::avx2);

    if (aligned_bias_ == nullptr) {
        aligned_bias_ = (OpDataType*)zmalloc(direc_num_ * layer_num_ * bias_size_ * aligned_hidden_size_ *
                                             sizeof(float), 4096);
    } else {
        LOG(ERROR) << "aligned bias in init should not be a non-nullptr";
    }

    weight_x_packed_.clear();
    weight_h_packed_.clear();

    for (int d = 0; d < direc_num_; d++) {
        const OpDataType* weights_data = static_cast<const OpDataType*>(param.weight()->data()) + d *
                                         W_stride;
        const OpDataType* bias_data = static_cast<const OpDataType*>(param.bias()->data()) + d * layer_num_
                                      * bias_size_ * hidden_size_;
        OpDataType* aligned_bias_data = aligned_bias_ + d * layer_num_ * bias_size_ * aligned_hidden_size_;

        for (int l = 0; l < layer_num_; l++) {
            // align bias
            for (int i = 0; i < bias_size_; i++) {
                memcpy(aligned_bias_data + l * bias_size_ * aligned_hidden_size_ + i * aligned_hidden_size_,
                       bias_data + l * bias_size_ * hidden_size_ + i * hidden_size_, hidden_size_ * sizeof(float));

                if (delta > 0) {
                    memset(aligned_bias_data + l * bias_size_ * aligned_hidden_size_ + i * aligned_hidden_size_ +
                           hidden_size_,
                           0, delta * sizeof(float));
                }
            }

            // align weights
            OpDataType* aligned_wx_tmp;
            OpDataType* aligned_wh_tmp;
            const OpDataType* wx = (l == 0) ? weights_data : weights_data + W_stride_l0 + (l - 1) * W_stride_ln;
            const OpDataType* wh = (l == 0) ? wx + Wx_stride_l0 : wx + Wx_stride_ln;
            int Wx_row = (l == 0) ? word_size_ : aligned_hidden_size_;

            if (delta > 0) {
                aligned_wx_tmp = (OpDataType*)zmalloc(Wx_row * aligned_hidden_size_ * 4 * sizeof(float), 4096);
                aligned_wh_tmp = (OpDataType*)zmalloc(4 * aligned_hidden_size_ * aligned_hidden_size_ * sizeof(
                        float), 4096);

                if (!(param.skip_input && l == 0)) {
                    for (int i = 0; i < Wx_row; i++) {
                        OpDataType* aligned_row = aligned_wx_tmp + i * aligned_hidden_size_ * 4;
                        const OpDataType* row = wx + i * hidden_size_ * 4;

                        if (i < hidden_size_ || l == 0) {
                            for (int j = 0; j < 4; j++) {
                                memcpy(aligned_row + j * aligned_hidden_size_, row + j * hidden_size_,
                                       hidden_size_ * sizeof(float));
                                memset(aligned_row + j * aligned_hidden_size_ + hidden_size_, 0, delta * sizeof(float));
                            }
                        } else {
                            memset(aligned_row, 0, 4 * aligned_hidden_size_ * sizeof(float));
                        }
                    }
                }

                for (int i = 0; i < aligned_hidden_size_; i++) {
                    OpDataType* aligned_row = aligned_wh_tmp + i * aligned_hidden_size_ * 4;
                    const OpDataType* row = wh + i * hidden_size_ * 4;

                    if (i < hidden_size_) {
                        for (int j = 0; j < 4; j++) {
                            memcpy(aligned_row + j * aligned_hidden_size_, row + j * hidden_size_,
                                   hidden_size_ * sizeof(float));
                            memset(aligned_row + j * aligned_hidden_size_ + hidden_size_, 0, delta * sizeof(float));
                        }
                    } else {
                        memset(aligned_row, 0, 4 * aligned_hidden_size_ * sizeof(float));
                    }
                }
            } else {
                aligned_wx_tmp = const_cast<OpDataType*>(wx);
                aligned_wh_tmp = const_cast<OpDataType*>(wh);
            }

            if (batch_size_ > 1) {
                OpDataType* weight_x_packed_tmp;
                OpDataType* weight_h_packed_tmp;
                weight_x_packed_tmp = cblas_sgemm_alloc(CblasBMatrix, inputs[0]->num(), 4 * aligned_hidden_size_,
                                                        Wx_row);

                if (!weight_x_packed_tmp) {
                    LOG(ERROR) << "cannot alloc weight_x_packed_ for lstm";
                    return SaberOutOfMem;
                }

                cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, inputs[0]->num(),
                                 4 * aligned_hidden_size_, Wx_row, 1.0,
                                 aligned_wx_tmp, 4 * aligned_hidden_size_, weight_x_packed_tmp);
                weight_h_packed_tmp = cblas_sgemm_alloc(CblasBMatrix, 1, 4 * aligned_hidden_size_,
                                                        aligned_hidden_size_);

                if (!weight_h_packed_tmp) {
                    LOG(ERROR) << "cannot alloc weight_h_packed_ for lstm";
                    return SaberOutOfMem;
                }

                cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, 1, 4 * aligned_hidden_size_,
                                 aligned_hidden_size_, 1.0,
                                 aligned_wh_tmp, 4 * aligned_hidden_size_, weight_h_packed_tmp);
                weight_x_packed_.push_back(weight_x_packed_tmp);
                weight_h_packed_.push_back(weight_h_packed_tmp);

                if (delta > 0) {
                    zfree(aligned_wx_tmp);
                    wx = nullptr;
                    zfree(aligned_wh_tmp);
                    wh = nullptr;
                }
            }

            if (aligned_wx_tmp != nullptr) {
                aligned_wx_.push_back(aligned_wx_tmp);
            }

            if (aligned_wh_tmp != nullptr) {
                aligned_wh_.push_back(aligned_wh_tmp);
            }
        }
    }

    return create(inputs, outputs, param, ctx);
}
template <>
SaberStatus VenderLstm<X86, AK_FLOAT>::single_batch(
    const std::vector<OpTensor*>& inputs,
    std::vector<OpTensor*>& outputs,
    LstmParam<X86>& param) {
    int direc_thread = 0;
    int wave_front_thread_num__num_ = 0;
    int layer_num_ = param.num_layers;
    int direc_num_ = param.num_direction;
    std::vector<std::vector<int>> seq_offset_vec = inputs[0]->get_seq_offset();
    std::vector<int> seq_offset = seq_offset_vec[seq_offset_vec.size() - 1];
    int batch_size_ = seq_offset.size() - 1;
    int word_sum = inputs[0]->num();
    bool is_reverse = param.is_reverse;
    int delta = aligned_hidden_size_ - hidden_size_;
    int bias_size_ = 4;

    if (param.with_peephole) {
        bias_size_ = 7;
    }

    int i_offset = 1;
    int c_offset = 2;
    int o_offset = 3;
    OpDataType* out = static_cast<OpDataType*>(outputs[0]->mutable_data());
    OpDataType* out_c = static_cast<OpDataType*>(outputs[1]->mutable_data());
    batched_h.set_shape(Shape({direc_num_, layer_num_, word_sum, aligned_hidden_size_}));
    batched_c.set_shape(Shape({direc_num_, layer_num_, word_sum, aligned_hidden_size_}));
    // init h
    aligned_init_hidden_ = (OpDataType*)zmalloc(direc_num_ * layer_num_ * aligned_hidden_size_ * sizeof(
                               float), 4096);
    aligned_init_hidden_c = (OpDataType*)zmalloc(direc_num_ * layer_num_ * aligned_hidden_size_ *
                            sizeof(float), 4096);

    if (param.init_hidden() != nullptr) {
        CHECK_EQ(param.init_hidden()->valid_shape().count(),
                 direc_num_ * layer_num_ * hidden_size_ * 2) << "hidden init size not matched";

        for (int d = 0; d < direc_num_; d++) {
            const OpDataType* h0 = static_cast<const OpDataType*>(param.init_hidden()->data()) + d * layer_num_
                                   * 2 *  hidden_size_;
            const OpDataType* c0 = static_cast<const OpDataType*>(param.init_hidden()->data()) + hidden_size_ +
                                   d * layer_num_ * 2 *
                                   hidden_size_;
            OpDataType* aligned_h0 = aligned_init_hidden_ + d * layer_num_ * aligned_hidden_size_;
            OpDataType* aligned_c0 = aligned_init_hidden_c + d * layer_num_ * aligned_hidden_size_;

            if (delta > 0) {
                for (int l = 0; l < layer_num_; l++) {
                    memcpy(aligned_h0 + l * aligned_hidden_size_, h0 + 2 * l * hidden_size_,
                           hidden_size_ * sizeof(float));
                    memset(aligned_h0 + l * aligned_hidden_size_ + hidden_size_, 0, delta * sizeof(float));
                    memcpy(aligned_c0 + l * aligned_hidden_size_, c0 + 2 * l * hidden_size_,
                           hidden_size_ * sizeof(float));
                    memset(aligned_c0 + l * aligned_hidden_size_ + hidden_size_, 0, delta * sizeof(float));
                }
            } else {
                for (int l = 0; l < layer_num_; l++) {
                    memcpy(aligned_h0 + l * aligned_hidden_size_, h0 + 2 * l * hidden_size_,
                           hidden_size_ * sizeof(float));
                    memcpy(aligned_c0 + l * aligned_hidden_size_, c0 + 2 * l * hidden_size_,
                           hidden_size_ * sizeof(float));
                }
            }
        }
    } else {
        memset(aligned_init_hidden_, 0, direc_num_ * layer_num_ * aligned_hidden_size_ * sizeof(float));
        memset(aligned_init_hidden_c, 0, direc_num_ * layer_num_ * aligned_hidden_size_ * sizeof(float));
    }

    OpDataType* Wx_x = nullptr;

    if (!param.skip_input) {
        Wx_x = (OpDataType*)zmalloc(direc_num_ * 4 * aligned_hidden_size_ * word_sum * sizeof(float), 4096);
        memset(Wx_x, 0, direc_num_ * word_sum * 4 * aligned_hidden_size_ * sizeof(float));

        for (int direction = 0; direction < direc_num_; direction++) {
            cblas_sgemm_compute(CblasRowMajor,
                                CblasNoTrans,
                                CblasNoTrans,
                                word_sum,
                                4 * aligned_hidden_size_,
                                word_size_,
                                static_cast<const OpDataType*>(inputs[0]->data()),
                                word_size_,
                                aligned_wx_[direction * layer_num_],
                                4 * aligned_hidden_size_,
                                0.f,
                                Wx_x + direction * word_sum * 4 * aligned_hidden_size_,
                                4 * aligned_hidden_size_);
        }
    }

    // cell_exec start
    auto cell_exec = [&](int layer_idx, int word_idx, int direction, bool reverse) {
        OpDataType* p = (OpDataType*)zmalloc(aligned_hidden_size_ * sizeof(float), 4096);
        OpDataType* act = (OpDataType*)zmalloc(aligned_hidden_size_ * sizeof(float), 4096);
        word_idx = (reverse) ? (word_sum - 1 - word_idx) : word_idx;
        const OpDataType* x = (layer_idx == 0) ? static_cast<const OpDataType*>
                              (inputs[0]->data()) + word_idx * word_size_ :
                              static_cast <OpDataType*>(batched_h.mutable_data())
                              + direction * layer_num_ * word_sum * aligned_hidden_size_
                              + (layer_idx - 1) * word_sum * aligned_hidden_size_ + word_idx * aligned_hidden_size_;
        const OpDataType* init_h = aligned_init_hidden_ + direction * layer_num_ * aligned_hidden_size_ +
                                   layer_idx * aligned_hidden_size_;
        const OpDataType* init_c = aligned_init_hidden_c + direction * layer_num_ * aligned_hidden_size_ +
                                   layer_idx * aligned_hidden_size_;
        const OpDataType* bias = aligned_bias_ + direction * layer_num_ * bias_size_ * aligned_hidden_size_
                                 + layer_idx * bias_size_ * aligned_hidden_size_;
        OpDataType* xx = static_cast<OpDataType*>(batched_xx.mutable_data()) + direction * layer_num_ *
                         word_sum * 4 *
                         aligned_hidden_size_ +
                         layer_idx * word_sum * 4 * aligned_hidden_size_ + word_idx * 4 * aligned_hidden_size_;
        OpDataType* ht = static_cast<OpDataType*>(batched_h.mutable_data()) + direction * layer_num_ *
                         word_sum * aligned_hidden_size_
                         +
                         layer_idx * word_sum * aligned_hidden_size_ + word_idx * aligned_hidden_size_;
        OpDataType* ct = static_cast<OpDataType*>(batched_c.mutable_data()) + direction * layer_num_ *
                         word_sum * aligned_hidden_size_
                         +
                         layer_idx * word_sum * aligned_hidden_size_ + word_idx * aligned_hidden_size_;
        const OpDataType* ht_1 = nullptr;
        const OpDataType* ct_1 = nullptr;

        if (reverse) {
            ht_1 = (word_idx == (word_sum - 1)) ? init_h : static_cast<OpDataType*>
                   (batched_h.mutable_data()) + direction * layer_num_ *
                   word_sum * aligned_hidden_size_ +
                   layer_idx * word_sum * aligned_hidden_size_ + (word_idx + 1) * aligned_hidden_size_;
            ct_1 = (word_idx == (word_sum - 1)) ? init_c : static_cast<OpDataType*>
                   (batched_c.mutable_data()) + direction * layer_num_ *
                   word_sum * aligned_hidden_size_ +
                   layer_idx * word_sum * aligned_hidden_size_ + (word_idx + 1) * aligned_hidden_size_;
        } else {
            ht_1 = (word_idx == 0) ? init_h : static_cast<OpDataType*>(batched_h.mutable_data()) + direction *
                   layer_num_ * word_sum *
                   aligned_hidden_size_ +
                   layer_idx * word_sum * aligned_hidden_size_ + (word_idx - 1) * aligned_hidden_size_;
            ct_1 = (word_idx == 0) ? init_c : static_cast<OpDataType*>(batched_c.mutable_data()) + direction *
                   layer_num_ * word_sum *
                   aligned_hidden_size_ +
                   layer_idx * word_sum * aligned_hidden_size_ + (word_idx - 1) * aligned_hidden_size_;
        }

        int x_stride = (layer_idx == 0) ? word_size_ : aligned_hidden_size_;

        if (layer_idx == 0) {
            // xx += bias
            if (param.skip_input) {
                if (delta > 0) {
                    for (int i = 0; i < 4; i++) {
                        memcpy(xx + i * aligned_hidden_size_, x + i * hidden_size_, hidden_size_ * sizeof(float));
                        memset(xx + i * aligned_hidden_size_ + hidden_size_, 0, delta * sizeof(float));
                    }
                } else {
                    memcpy(xx, x, word_size_ * sizeof(float));
                }
            } else {
                // non skip input
                memcpy(xx, Wx_x + direction * 4 * aligned_hidden_size_ * word_sum + word_idx * 4 *
                       aligned_hidden_size_, 4 * aligned_hidden_size_ * sizeof(float));
            }
        } else {
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        1,
                        4 * aligned_hidden_size_,
                        x_stride,
                        1,
                        x,
                        x_stride,
                        aligned_wx_[direction * layer_num_ + layer_idx],
                        4 * aligned_hidden_size_,
                        0.f,
                        xx,
                        4 * aligned_hidden_size_);
        }

        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    1,
                    4 * aligned_hidden_size_,
                    aligned_hidden_size_,
                    1,
                    ht_1,
                    aligned_hidden_size_,
                    aligned_wh_[direction * layer_num_ + layer_idx],
                    4 * aligned_hidden_size_,
                    1.f,
                    xx,
                    4 * aligned_hidden_size_);

        if (param.with_peephole) {
            // caculate ft
            vsMul(aligned_hidden_size_, ct_1, bias + 4 *  aligned_hidden_size_, p);
            cblas_saxpby(aligned_hidden_size_, 1, p, 1, 1, xx, 1);
            // caculate it
            vsMul(aligned_hidden_size_, ct_1, bias + (4 + i_offset) * aligned_hidden_size_, p);
            cblas_saxpby(aligned_hidden_size_, 1, p, 1, 1, xx + i_offset * aligned_hidden_size_, 1);
        }

        // xx += bias
        cblas_saxpy(4 * aligned_hidden_size_, 1, bias, 1, xx, 1);
#if defined(__AVX2__) and defined(__FMA__)

        // compute forget gate input gate and Cct
        if (avx2_available_) {
            __m256* ft = (__m256*)(xx);
            __m256* it = (__m256*)(xx + i_offset * aligned_hidden_size_);
            __m256* Cct = (__m256*)(xx + c_offset * aligned_hidden_size_);
            __m256* ct_temp = (__m256*)(ct);
            __m256* ct_1_temp = (__m256*)(ct_1);

            for (int i = 0; i < aligned_hidden_size_ / 8; ++i) {
                ft[i] = Activate_inner(ft[i], param.gate_activity);
                it[i] = Activate_inner(it[i], param.gate_activity);
                Cct[i] = Activate_inner(Cct[i], param.candidate_activity);
                ct_temp[i] = Cct[i] * ft[i] + ct_1_temp[i] * it[i];
            }
        } else {
            float* ft = (float*)(xx);
            float* it = (float*)(xx + i_offset * aligned_hidden_size_);
            float* Cct = (float*)(xx + c_offset * aligned_hidden_size_);
            float* ct_temp = (float*)(ct);
            float* ct_1_temp = (float*)(ct_1);

            for (int i = 0; i < aligned_hidden_size_; ++i) {
                ft[i] = Activate_inner(ft[i], param.gate_activity);
                it[i] = Activate_inner(it[i], param.gate_activity);
                Cct[i] = Activate_inner(Cct[i], param.candidate_activity);
                ct_temp[i] = Cct[i] * ft[i] + ct_1_temp[i] * it[i];
            }
        }

#else
        float* ft = (float*)(xx);
        float* it = (float*)(xx + i_offset * aligned_hidden_size_);
        float* Cct = (float*)(xx + c_offset * aligned_hidden_size_);
        float* ct_temp = (float*)(ct);
        float* ct_1_temp = (float*)(ct_1);

        for (int i = 0; i < aligned_hidden_size_; ++i) {
            ft[i] = Activate_inner(ft[i], param.gate_activity);
            it[i] = Activate_inner(it[i], param.gate_activity);
            Cct[i] = Activate_inner(Cct[i], param.candidate_activity);
            ct_temp[i] = Cct[i] * ft[i] + ct_1_temp[i] * it[i];
        }

#endif

        // peephole for ot
        if (param.with_peephole) {
            // p = Ct * Woc
            vsMul(aligned_hidden_size_, ct, bias + 6 * aligned_hidden_size_, p);
            // Wo[ht_1, xt] + Ct * Woc
            cblas_saxpby(aligned_hidden_size_, 1, p, 1, 1, xx + o_offset * aligned_hidden_size_, 1);
        }

        memcpy(act, ct, aligned_hidden_size_ * sizeof(float));
        // compute output gate
#if defined(__AVX2__) and defined(__FMA__)

        if (avx2_available_) {
            __m256* ot = (__m256*)(xx + o_offset * aligned_hidden_size_);
            __m256* Hht = (__m256*)(act);
            __m256* ht_temp = (__m256*)(ht);

            for (int i = 0; i < aligned_hidden_size_ / 8; ++i) {
                ot[i] = Activate_inner(ot[i], param.gate_activity);
                Hht[i] = Activate_inner(Hht[i], param.cell_activity);
                ht_temp[i] = ot[i] * Hht[i];
            }
        } else {
            float* ot = (float*)(xx + o_offset * aligned_hidden_size_);
            float* Hht = (float*)(act);
            float* ht_temp = (float*)(ht);

            for (int i = 0; i < aligned_hidden_size_; ++i) {
                ot[i] = Activate_inner(ot[i], param.gate_activity);
                Hht[i] = Activate_inner(Hht[i], param.cell_activity);
                ht_temp[i] = ot[i] * Hht[i];
            }
        }

#else
        float* ot = (float*)(xx + o_offset * aligned_hidden_size_);
        float* Hht = (float*)(act);
        float* ht_temp = (float*)(ht);

        for (int i = 0; i < aligned_hidden_size_; ++i) {
            ot[i] = Activate_inner(ot[i], param.gate_activity);
            Hht[i] = Activate_inner(Hht[i], param.cell_activity);
            ht_temp[i] = ot[i] * Hht[i];
        }

#endif

        if (p) {
            zfree(p);
            p = nullptr;
        }

        if (act) {
            zfree(act);
            act = nullptr;
        }
    };
    int min_dim, max_dim;
    bool l_gt_t;

    if (layer_num_ > word_sum) {
        min_dim = word_sum;
        max_dim = layer_num_;
        l_gt_t = true;
    } else {
        min_dim = layer_num_;
        max_dim = word_sum;
        l_gt_t = false;
    }

    wave_front_thread_num_ = min_dim < (max_thread_num_ / direc_num_) ? min_dim :
                             (max_thread_num_ / direc_num_);
    mkl_thread_num_ = max_thread_num_ / (wave_front_thread_num_ * direc_num_) > 1 ? max_thread_num_ /
                      (wave_front_thread_num_ * direc_num_) : 1;

    // in single layer condition
    if (layer_num_ == 1) {
        wave_front_thread_num_ = 1;
        direction_parallel_num_ = 1;
        mkl_thread_num_ = max_thread_num_ / direc_num_;
        // in multilayer and long sequence condition
    } else if (layer_num_ >= 3 && word_sum > 14) {
        mkl_thread_num_ = 1;
    }

    mkl_set_num_threads(mkl_thread_num_);
    // omp_set_num_threads(1);
    auto grid_exec = [&](int direction, bool reverse) {
        #pragma omp parallel num_threads(wave_front_thread_num_)
        {
            for (int n = 0; n < min_dim; n++) {
                #pragma omp for

                for (int t = 0; t <= n; t++) {
                    cell_exec(n - t, t, direction, reverse);
                }
            }

            for (int n = 0; n < max_dim - min_dim; n++) {
                #pragma omp for

                for (int t = 0; t < min_dim; t++) {
                    if (l_gt_t) {
                        cell_exec(min_dim + n - t, t, direction, reverse);
                    } else {
                        cell_exec(t, min_dim + n - t, direction, reverse);
                    }
                }
            }

            for (int n = min_dim - 1; n > 0; n--) {
                #pragma omp for

                for (int t = 0; t < n; t++) {
                    cell_exec(layer_num_ - n + t, word_sum - t - 1, direction, reverse);
                }
            }
        }
    };

    if (direc_num_ == 1) {
        grid_exec(0, is_reverse);
    } else {
        #pragma omp parallel sections num_threads(direction_parallel_num_)
        {
            #pragma omp section
            {
                grid_exec(0, false);
            }
            #pragma omp section
            {
                grid_exec(1, true);
            }
        }
    }

    outputs[0]->re_alloc(Shape({word_sum, direc_num_ * hidden_size_, 1, 1}));
    outputs[0]->reshape(Shape({word_sum, direc_num_ * hidden_size_, 1, 1}));
    out = static_cast<OpDataType*>(outputs[0]->mutable_data());
    #pragma omp parallel for

    for (int d = 0; d < direc_num_; d++) {
        for (int i = 0; i < word_sum; i++) {
            int in_start = i * aligned_hidden_size_;
            int out_start = i * direc_num_ * hidden_size_ + d * hidden_size_;
            memcpy(out + out_start, static_cast<OpDataType*>(batched_h.mutable_data()) +
                   d * layer_num_ * word_sum * aligned_hidden_size_ +
                   (layer_num_ - 1) * word_sum * aligned_hidden_size_ + in_start,
                   hidden_size_ * sizeof(float));
            memcpy(out_c + out_start, static_cast<OpDataType*>(batched_c.mutable_data()) +
                   d * layer_num_ * word_sum * aligned_hidden_size_ +
                   (layer_num_ - 1) * word_sum * aligned_hidden_size_ + in_start,
                   hidden_size_ * sizeof(float));
        }
    }

    if (Wx_x) {
        zfree(Wx_x);
        Wx_x = nullptr;
    }

    return SaberSuccess;
}
template <>
SaberStatus VenderLstm<X86, AK_FLOAT>::dispatch(
    const std::vector<OpTensor*>& inputs,
    std::vector<OpTensor*>& outputs,
    LstmParam<X86>& param) {
    int layer_num_ = param.num_layers;
    int direc_num_ = param.num_direction;
    std::vector<std::vector<int>> seq_offset_vec = inputs[0]->get_seq_offset();
    std::vector<int> seq_offset = seq_offset_vec[seq_offset_vec.size() - 1];
    int batch_size_ = seq_offset.size() - 1;
    int word_sum = inputs[0]->num();
    bool is_reverse = param.is_reverse;
    int delta = aligned_hidden_size_ - hidden_size_;
    int bias_size_ = 4;

    if (param.with_peephole) {
        bias_size_ = 7;
    }

    int i_offset = 1;
    int c_offset = 2;
    int o_offset = 3;
    anakin_set_nested(1);
    mkl_set_dynamic(0);

    if (batch_size_ == 1) {
        return single_batch(inputs, outputs, param);
    } else {
        // sequence to batch
        batched_h.set_shape(Shape({direc_num_, layer_num_, word_sum, aligned_hidden_size_}));
        batched_c.set_shape(Shape({direc_num_, layer_num_, word_sum, aligned_hidden_size_}));
        batched_x.set_shape(Shape({word_sum, word_size_, 1, 1}));
        batched_xx.set_shape(Shape({direc_num_, layer_num_, inputs[0]->num(), 4 * aligned_hidden_size_}));
        std::vector<std::vector<int>> seq_to_batch_meta;
        std::vector<std::vector<int>> seq_to_batch_reverse_meta;
        seq_to_batch_meta.push_back(seq_offset);
        math::Seq2BatchFunctor<AK_FLOAT, NCHW> to_batch;

        if (direc_num_ == 1) {
            to_batch(inputs[0], &batched_x, seq_to_batch_meta, true, is_reverse, 1);
        } else {
            seq_to_batch_reverse_meta.push_back(seq_offset);
            batched_x_reverse.set_shape(Shape({word_sum, word_size_, 1, 1}));
            to_batch(inputs[0], &batched_x, seq_to_batch_meta, true, false, 1);
            to_batch(inputs[0], &batched_x_reverse, seq_to_batch_reverse_meta, true, true, 1);
        }

        auto bat_offset = seq_to_batch_meta[0];
        size_t bat_num = bat_offset.size() - 1;
        // init h
        aligned_init_hidden_ = (OpDataType*)zmalloc(direc_num_ * layer_num_ * batch_size_ *
                               aligned_hidden_size_ * sizeof(float), 4096);
        aligned_init_hidden_c = (OpDataType*)zmalloc(direc_num_ * layer_num_ * batch_size_ *
                                aligned_hidden_size_ * sizeof(float), 4096);

        if (param.init_hidden() != nullptr) {
            CHECK_EQ(param.init_hidden()->valid_shape().count(),
                     direc_num_ * layer_num_ * batch_size_ * 2 * hidden_size_) << "hidden init size not matched";
            std::vector<int> order(seq_to_batch_meta[2]);

            for (int d = 0; d < direc_num_; d++) {
                const OpDataType* h0 = static_cast<const OpDataType*>(param.init_hidden()->data()) + d * layer_num_
                                       * 2 * batch_size_ *
                                       hidden_size_;
                OpDataType* aligned_h0 = aligned_init_hidden_ + d * layer_num_ * batch_size_ * aligned_hidden_size_;
                const OpDataType* c0 = static_cast<const OpDataType*>(param.init_hidden()->data()) + d * layer_num_
                                       * 2 * batch_size_ * hidden_size_
                                       + batch_size_ * hidden_size_;
                OpDataType* aligned_c0 = aligned_init_hidden_c + d * layer_num_ * batch_size_ *
                                         aligned_hidden_size_;

                if (delta > 0) {
                    for (int l = 0; l < layer_num_; l++) {
                        for (int i = 0; i < batch_size_; i++) {
                            memcpy(aligned_h0 + l * batch_size_ * aligned_hidden_size_ + i * aligned_hidden_size_,
                                   h0 + l * 2 * batch_size_ * hidden_size_ + order[i] * hidden_size_, hidden_size_ * sizeof(float));
                            memset(aligned_h0 + l * batch_size_ * aligned_hidden_size_ + hidden_size_, 0,
                                   delta * sizeof(float));
                            memcpy(aligned_c0 + l * batch_size_ * aligned_hidden_size_ + i * aligned_hidden_size_,
                                   c0 + l * 2 * batch_size_ * hidden_size_ + order[i] * hidden_size_, hidden_size_ * sizeof(float));
                            memset(aligned_c0 + l * batch_size_ * aligned_hidden_size_ + hidden_size_, 0,
                                   delta * sizeof(float));
                        }
                    }
                } else {
                    for (int l = 0; l < layer_num_; l++) {
                        for (int i = 0; i < batch_size_; i++) {
                            memcpy(aligned_h0 + l * batch_size_ * aligned_hidden_size_ + i * aligned_hidden_size_,
                                   h0 + l * 2 * batch_size_ * hidden_size_ + order[i] * hidden_size_, hidden_size_ * sizeof(float));
                            memcpy(aligned_c0 + l * batch_size_ * aligned_hidden_size_ + i * aligned_hidden_size_,
                                   c0 + l * 2 * batch_size_ * hidden_size_ + order[i] * hidden_size_, hidden_size_ * sizeof(float));
                        }
                    }
                }
            }
        } else {
            memset(aligned_init_hidden_, 0,
                   direc_num_ * layer_num_ * batch_size_ * aligned_hidden_size_ * sizeof(float));
            memset(aligned_init_hidden_c, 0,
                   direc_num_ * layer_num_ * batch_size_ * aligned_hidden_size_ * sizeof(float));
        }

        // cell execution start
        auto cell_exec = [&](int layer_idx, int word_idx, int direction) {
            OpDataType* p = (OpDataType*)zmalloc(aligned_hidden_size_ * sizeof(float), 4096);
            OpDataType* act = (OpDataType*)zmalloc(aligned_hidden_size_ * sizeof(float), 4096);
            int bat_start = bat_offset[word_idx];
            int bat_end = bat_offset[word_idx + 1];
            int bat_length = bat_end - bat_start;
            const OpDataType* batched_x_data = (direction == 0) ? static_cast<const OpDataType*>
                                               (batched_x.data()) : static_cast<const OpDataType*>(batched_x_reverse.data());
            const OpDataType* x = (layer_idx == 0) ? batched_x_data + bat_start * word_size_ :
                                  static_cast< OpDataType*>(batched_h.mutable_data())
                                  + direction * layer_num_ * word_sum * aligned_hidden_size_
                                  + (layer_idx - 1) * word_sum * aligned_hidden_size_ + bat_start * aligned_hidden_size_;
            const OpDataType* init_h = aligned_init_hidden_ + direction * layer_num_ * batch_size_ *
                                       aligned_hidden_size_
                                       + layer_idx * batch_size_ * aligned_hidden_size_;
            const OpDataType* init_c = aligned_init_hidden_c + direction * layer_num_ * batch_size_ *
                                       aligned_hidden_size_
                                       + layer_idx * batch_size_ * aligned_hidden_size_;
            const OpDataType* bias = aligned_bias_ + direction * layer_num_ * bias_size_ * aligned_hidden_size_
                                     + layer_idx
                                     * bias_size_ * aligned_hidden_size_;
            OpDataType* xx = static_cast<OpDataType*>(batched_xx.mutable_data()) + direction * layer_num_ *
                             word_sum * 4 *
                             aligned_hidden_size_ +
                             layer_idx * word_sum * 4 * aligned_hidden_size_ + bat_start * 4 * aligned_hidden_size_;
            const OpDataType* ht_1 = (word_idx == 0) ? init_h : static_cast<OpDataType*>
                                     (batched_h.mutable_data()) + direction *
                                     layer_num_ * word_sum * aligned_hidden_size_ +
                                     layer_idx * word_sum * aligned_hidden_size_ + bat_offset[word_idx - 1] * aligned_hidden_size_;
            const OpDataType* ct_1 = (word_idx == 0) ? init_c : static_cast<OpDataType*>
                                     (batched_c.mutable_data()) + direction *
                                     layer_num_ * word_sum * aligned_hidden_size_ +
                                     layer_idx * word_sum * aligned_hidden_size_ + bat_offset[word_idx - 1] * aligned_hidden_size_;
            OpDataType* ht = static_cast<OpDataType*>(batched_h.mutable_data()) + direction * layer_num_ *
                             word_sum * aligned_hidden_size_
                             +
                             layer_idx * word_sum * aligned_hidden_size_ + bat_start * aligned_hidden_size_;
            OpDataType* ct = static_cast<OpDataType*>(batched_c.mutable_data()) + direction * layer_num_ *
                             word_sum * aligned_hidden_size_
                             +
                             layer_idx * word_sum * aligned_hidden_size_ + bat_start * aligned_hidden_size_;
            int x_stride = (layer_idx == 0) ? word_size_ : aligned_hidden_size_;

            if (layer_idx == 0 && param.skip_input) {
                // xx += bias
                if (delta > 0) {
                    for (int i = 0; i < bat_length; i++) {
                        for (int j = 0; j < 4; j++) {
                            memcpy(xx + i * 4 * aligned_hidden_size_ + j * aligned_hidden_size_,
                                   x + i * 4 * hidden_size_ + j * hidden_size_,
                                   hidden_size_ * sizeof(float));
                            memset(xx + i * 4 * aligned_hidden_size_ + j * aligned_hidden_size_ + hidden_size_,
                                   0, delta * sizeof(float));
                        }
                    }
                } else {
                    cblas_saxpy(bat_length * word_size_, 1, x, 0, xx, 1);
                }
            } else {
                cblas_sgemm_compute(CblasRowMajor,
                                    CblasNoTrans,
                                    CblasPacked,
                                    bat_length,
                                    4 * aligned_hidden_size_,
                                    x_stride,
                                    x,
                                    x_stride,
                                    weight_x_packed_[direction * layer_num_ + layer_idx],
                                    4 * aligned_hidden_size_,
                                    0.f,
                                    xx,
                                    4 * aligned_hidden_size_);
            }

            // batched_xx += ht_1 * Wh
            cblas_sgemm_compute(CblasRowMajor,
                                CblasNoTrans,
                                CblasPacked,
                                bat_length,
                                4 * aligned_hidden_size_,
                                aligned_hidden_size_,
                                ht_1,
                                aligned_hidden_size_,
                                weight_h_packed_[direction * layer_num_ + layer_idx],
                                4 * aligned_hidden_size_,
                                1.f,
                                xx,
                                4 * aligned_hidden_size_);

            // batched_xx += bias
            for (int s = 0; s < bat_length; s++) {
                cblas_saxpy(4 * aligned_hidden_size_, 1, bias, 1, xx + s * 4 * aligned_hidden_size_, 1);
            }

            // compute four gate
#if defined(__AVX2__) and defined(__FMA__)

            if (avx2_available_) {
                for (int s = 0; s < bat_length; s++) {
                    const OpDataType* cit_1 = ct_1 + s * aligned_hidden_size_;
                    OpDataType* cit = ct + s * aligned_hidden_size_;

                    if (param.with_peephole) {
                        // caculate ft
                        vsMul(aligned_hidden_size_, cit_1, bias + 4 * aligned_hidden_size_, p);
                        cblas_saxpby(aligned_hidden_size_, 1, p, 1, 1, xx + s * 4 * aligned_hidden_size_, 1);
                        // caculate it
                        vsMul(aligned_hidden_size_, cit_1, bias + (4 + i_offset) * aligned_hidden_size_, p);
                        cblas_saxpby(aligned_hidden_size_, 1, p, 1, 1,
                                     xx + i_offset * aligned_hidden_size_ + s * 4 * aligned_hidden_size_, 1);
                    }

                    __m256* ft = (__m256*)(xx + s * 4 * aligned_hidden_size_);
                    __m256* it = (__m256*)(xx + s * 4 * aligned_hidden_size_ + i_offset * aligned_hidden_size_);
                    __m256* Cct = (__m256*)(xx + s * 4 * aligned_hidden_size_ + c_offset * aligned_hidden_size_);
                    __m256* m_cit = (__m256*)(cit);
                    __m256* m_cit_1 = (__m256*)(cit_1);

                    for (int i = 0; i < aligned_hidden_size_ / 8; ++i) {
                        ft[i] = Activate_inner(ft[i], param.gate_activity);
                        it[i] = Activate_inner(it[i], param.gate_activity);
                        Cct[i] = Activate_inner(Cct[i], param.candidate_activity);
                        m_cit[i] = ft[i] * Cct[i] + it[i] * m_cit_1[i];
                    }

                    // peephole for ot
                    if (param.with_peephole) {
                        // p = Ct * Woc
                        vsMul(aligned_hidden_size_, cit, bias + 6 * aligned_hidden_size_, p);
                        // Wo[ht_1, xt] + Ct * Woc
                        cblas_saxpby(aligned_hidden_size_, 1, p, 1, 1,
                                     xx + s * 4 * aligned_hidden_size_ + o_offset * aligned_hidden_size_, 1);
                    }

                    memcpy(act, cit, aligned_hidden_size_ * sizeof(float));
                    // compute output gate
                    __m256* ot = (__m256*)(xx + s * 4 * aligned_hidden_size_ + o_offset * aligned_hidden_size_);
                    __m256* Hht = (__m256*)(act);
                    __m256* hit_temp = (__m256*)(ht + s * aligned_hidden_size_);

                    for (int i = 0; i < aligned_hidden_size_ / 8; ++i) {
                        ot[i] = Activate_inner(ot[i], param.gate_activity);
                        Hht[i] = Activate_inner(Hht[i], param.cell_activity);
                        hit_temp[i] = ot[i] * Hht[i];
                    }
                }
            } else {
                for (int s = 0; s < bat_length; s++) {
                    const OpDataType* cit_1 = ct_1 + s * aligned_hidden_size_;
                    OpDataType* cit = ct + s * aligned_hidden_size_;
                    OpDataType* hit = ht + s * aligned_hidden_size_;

                    if (param.with_peephole) {
                        // caculate ft
                        vsMul(aligned_hidden_size_, cit_1, bias + 4 * aligned_hidden_size_, p);
                        cblas_saxpby(aligned_hidden_size_, 1, p, 1, 1, xx + s * 4 * aligned_hidden_size_, 1);
                        // caculate it
                        vsMul(aligned_hidden_size_, cit_1, bias + (4 + i_offset) * aligned_hidden_size_, p);
                        cblas_saxpby(aligned_hidden_size_, 1, p, 1, 1,
                                     xx + i_offset * aligned_hidden_size_ + s * 4 * aligned_hidden_size_, 1);
                    }

                    float* ft = (float*)(xx + s * 4 * aligned_hidden_size_);
                    float* it = (float*)(xx + s * 4 * aligned_hidden_size_ + i_offset * aligned_hidden_size_);
                    float* Cct = (float*)(xx + s * 4 * aligned_hidden_size_ + c_offset * aligned_hidden_size_);
                    float* m_cit = (float*)(cit);
                    float* m_cit_1 = (float*)(cit_1);

                    for (int i = 0; i < aligned_hidden_size_; ++i) {
                        ft[i] = Activate_inner(ft[i], param.gate_activity);
                        it[i] = Activate_inner(it[i], param.gate_activity);
                        Cct[i] = Activate_inner(Cct[i], param.candidate_activity);
                        m_cit[i] = ft[i] * Cct[i] + it[i] * m_cit_1[i];
                    }

                    // peephole for ot
                    if (param.with_peephole) {
                        // p = Ct * Woc
                        vsMul(aligned_hidden_size_, cit, bias + 6 * aligned_hidden_size_, p);
                        // Wo[ht_1, xt] + Ct * Woc
                        cblas_saxpby(aligned_hidden_size_, 1, p, 1, 1,
                                     xx + s * 4 * aligned_hidden_size_ + o_offset * aligned_hidden_size_, 1);
                    }

                    memcpy(act, cit, aligned_hidden_size_ * sizeof(float));
                    float* ot = (float*)(xx + s * 4 * aligned_hidden_size_ + o_offset * aligned_hidden_size_);
                    float* Hht = (float*)(act);
                    float* hit_temp = (float*)(ht + s * aligned_hidden_size_);

                    for (int i = 0; i < aligned_hidden_size_; ++i) {
                        ot[i] = Activate_inner(ot[i], param.gate_activity);
                        Hht[i] = Activate_inner(Hht[i], param.cell_activity);
                        hit_temp[i] = ot[i] * Hht[i];
                    }
                }
            }

#else

        for (int s = 0; s < bat_length; s++) {
            const OpDataType* cit_1 = ct_1 + s * aligned_hidden_size_;
            OpDataType* cit = ct + s * aligned_hidden_size_;
            OpDataType* hit = ht + s * aligned_hidden_size_;

            if (param.with_peephole) {
                // caculate ft
                vsMul(aligned_hidden_size_, cit_1, bias + 4 * aligned_hidden_size_, p);
                cblas_saxpby(aligned_hidden_size_, 1, p, 1, 1, xx + s * 4 * aligned_hidden_size_, 1);
                // caculate it
                vsMul(aligned_hidden_size_, cit_1, bias + (4 + i_offset) * aligned_hidden_size_, p);
                cblas_saxpby(aligned_hidden_size_, 1, p, 1, 1,
                             xx + i_offset * aligned_hidden_size_ + s * 4 * aligned_hidden_size_, 1);
            }

            float* ft = (float*)(xx + s * 4 * aligned_hidden_size_);
            float* it = (float*)(xx + s * 4 * aligned_hidden_size_ + i_offset * aligned_hidden_size_);
            float* Cct = (float*)(xx + s * 4 * aligned_hidden_size_ + c_offset * aligned_hidden_size_);
            float* m_cit = (float*)(cit);
            float* m_cit_1 = (float*)(cit_1);

            for (int i = 0; i < aligned_hidden_size_; ++i) {
                ft[i] = Activate_inner(ft[i], param.gate_activity);
                it[i] = Activate_inner(it[i], param.gate_activity);
                Cct[i] = Activate_inner(Cct[i], param.candidate_activity);
                m_cit[i] = ft[i] * Cct[i] + it[i] * m_cit_1[i];
            }

            // peephole for ot
            if (param.with_peephole) {
                // p = Ct * Woc
                vsMul(aligned_hidden_size_, cit, bias + 6 * aligned_hidden_size_, p);
                // Wo[ht_1, xt] + Ct * Woc
                cblas_saxpby(aligned_hidden_size_, 1, p, 1, 1,
                             xx + s * 4 * aligned_hidden_size_ + o_offset * aligned_hidden_size_, 1);
            }

            memcpy(act, cit, aligned_hidden_size_ * sizeof(float));
            float* ot = (float*)(xx + s * 4 * aligned_hidden_size_ + o_offset * aligned_hidden_size_);
            float* Hht = (float*)(act);
            float* hit_temp = (float*)(ht + s * aligned_hidden_size_);

            for (int i = 0; i < aligned_hidden_size_; ++i) {
                ot[i] = Activate_inner(ot[i], param.gate_activity);
                Hht[i] = Activate_inner(Hht[i], param.cell_activity);
                hit_temp[i] = ot[i] * Hht[i];
            }
        }

#endif

            if (p) {
                zfree(p);
                p = nullptr;
            }

            if (act) {
                zfree(act);
                act = nullptr;
            }
        };
        // cell execution end
        int min_dim, max_dim;
        bool l_gt_t;

        if (layer_num_ > bat_num) {
            min_dim = bat_num;
            max_dim = layer_num_;
            l_gt_t = true;
        } else {
            min_dim = layer_num_;
            max_dim = bat_num;
            l_gt_t = false;
        }

        wave_front_thread_num_ = min_dim < (max_thread_num_ / direc_num_) ? min_dim :
                                 (max_thread_num_ / direc_num_);
        mkl_thread_num_ = max_thread_num_ / (wave_front_thread_num_ * direc_num_) > 1 ? max_thread_num_ /
                          (wave_front_thread_num_ * direc_num_) : 1;

        // in single layer condition
        if (layer_num_ == 1) {
            wave_front_thread_num_ = 1;
            direction_parallel_num_ = 1;
            mkl_thread_num_ = max_thread_num_ / direc_num_;
            // in multilayer and long sequence condition
        } else if (layer_num_ >= 3 && bat_num > 14) {
            mkl_thread_num_ = 1;
        }

        mkl_set_num_threads(mkl_thread_num_);
        // omp_set_num_threads(wave_front_thread_num_);
        auto grid_exec = [&](int direction) {
            #pragma omp parallel num_threads(wave_front_thread_num_)
            {
                for (int n = 0; n < min_dim; n++) {
                    #pragma omp for

                    for (int t = 0; t <= n; t++) {
                        cell_exec(n - t, t, direction);
                    }
                }

                for (int n = 0; n < max_dim - min_dim; n++) {
                    #pragma omp for

                    for (int t = 0; t < min_dim; t++) {
                        if (l_gt_t) {
                            cell_exec(min_dim + n - t, t, direction);
                        } else {
                            cell_exec(t, min_dim + n - t, direction);
                        }
                    }
                }

                for (int n = min_dim - 1; n > 0; n--) {
                    #pragma omp for

                    for (int t = 0; t < n; t++) {
                        cell_exec(layer_num_ - n + t, bat_num - t - 1, direction);
                    }
                }
            }
        };

        if (direc_num_ == 1) {
            grid_exec(0);
        } else {
            #pragma omp parallel sections num_threads(direction_parallel_num_)
            {
                #pragma omp section
                {
                    grid_exec(0);
                }
                #pragma omp section
                {
                    grid_exec(1);
                }
            }
        }

        outputs[0]->re_alloc(Shape({word_sum, direc_num_ * hidden_size_, 1, 1}));
        outputs[0]->reshape(Shape({word_sum, direc_num_ * hidden_size_, 1, 1}));
        // batch to sequence
        OpTensor batched_out(static_cast<OpDataType*>(batched_h.mutable_data()) + (layer_num_ - 1)
                             * word_sum * aligned_hidden_size_, X86(), 0, Shape({word_sum, aligned_hidden_size_, 1, 1}));
        OpTensor batched_out_c(static_cast<OpDataType*>(batched_c.mutable_data()) + (layer_num_ - 1)
                               * word_sum * aligned_hidden_size_, X86(), 0, Shape({word_sum, aligned_hidden_size_, 1, 1}));
        math::Batch2SeqFunctor<AK_FLOAT, NCHW> to_seq;

        if (direc_num_ == 1) {
            to_seq(&batched_out, outputs[0], seq_to_batch_meta);
            to_seq(&batched_out_c, outputs[1], seq_to_batch_meta);
        } else {
            OpTensor batched_out_reverse(static_cast<OpDataType*>(batched_h.mutable_data()) + layer_num_ *
                                         word_sum*
                                         aligned_hidden_size_
                                         + (layer_num_ - 1) * word_sum * aligned_hidden_size_, X86(), 0, Shape({word_sum, aligned_hidden_size_, 1,
                                                 1}));
            OpTensor batched_out_c_reverse(static_cast<OpDataType*>(batched_c.mutable_data()) + layer_num_ *
                                           word_sum*
                                           aligned_hidden_size_
                                           + (layer_num_ - 1) * word_sum * aligned_hidden_size_, X86(), 0, Shape({word_sum, aligned_hidden_size_, 1,
                                                   1}));
            to_seq(&batched_out, outputs[0], seq_to_batch_meta, 1, 0, hidden_size_);
            to_seq(&batched_out_reverse, outputs[0], seq_to_batch_reverse_meta, 1, hidden_size_, hidden_size_);
            to_seq(&batched_out_c, outputs[1], seq_to_batch_meta, 1, 0, hidden_size_);
            to_seq(&batched_out_c_reverse, outputs[1], seq_to_batch_reverse_meta, 1, hidden_size_,
                   hidden_size_);
        }
    }

    return SaberSuccess;
}
template class VenderLstm<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(VenderLstm, LstmParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(VenderLstm, LstmParam, X86, AK_INT8);
} // namespace saber
} // namespace anakin
