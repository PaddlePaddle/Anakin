#include "saber/funcs/impl/x86/saber_lstmp.h"
#include "mkl_cblas.h"
#include "mkl.h"
#include "saber_normal_activation.h"
#include "debug.h"
#include "timer.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#define SABER_X86_TYPE __m512
#elif defined(__AVX2__) and defined(__FMA__)
#include <immintrin.h>
#define SABER_X86_TYPE __m256
#elif defined(__SSE4_2__) and defined(__FMA__)
#include <immintrin.h>
#define SABER_X86_TYPE __m128
#else
#define SABER_X86_TYPE float
#endif

namespace anakin {

namespace saber {

static void gemm(const bool TransA, const bool TransB, int m, int n, int k, const float alpha,
                 const float* a, const float* b, const float beta, float* c) {
    //    cout << "(" << m << "," << n << "," << k << ")" << endl;
    int lda = (!TransA/* == CblasNoTrans*/) ? k : m;
    int ldb = (!TransB/* == CblasNoTrans*/) ? n : k;
    CBLAS_TRANSPOSE cu_trans_a =
        (!TransA/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cu_trans_b =
        (!TransB/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    Context<X86> ctx(0, 0, 0);
    SaberTimer<X86> timer;
    timer.start(ctx);
    cblas_sgemm(CblasRowMajor, cu_trans_a, cu_trans_b, m, n, k, alpha, a, k, b, n, beta, c, n);
    timer.end(ctx);
    double ms = timer.get_average_ms();
    double work_load = (double)m * n * k * 2;
    double speed = work_load / ms / 1000.0 / 1000.0;
    LOG(INFO) << "mkl_cblas_sgemm " << m << "," << n << "," << k << "," << ms << "," << speed;
};

static void s8s8s32_gemm(const bool TransA, const bool TransB, int m, int n, int k,
                         const float alpha,
                         const int8_t* a, const int8_t* b, const float beta, int32_t* c) {

};


template<>
SaberStatus SaberLstmp<X86, AK_FLOAT>:: create(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        LstmParam<X86>& param,
        Context<X86>& ctx) {
    return SaberSuccess;
};


template<>
SaberStatus SaberLstmp<X86, AK_FLOAT>::init(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        LstmParam<X86>& param,
        Context<X86>& ctx) {
    _inner_hidden_dim = param.cell_dim;
    _output_hidden_dim = param.project_dim;

    CHECK_GT(param.cell_dim, 0);
    CHECK_GT(param.project_dim, 0);
    CHECK_EQ(param.cell_dim % (sizeof(SABER_X86_TYPE) / sizeof(float)), 0);

    int word_dim = inputs[0]->channel();
    const float* weights_x_ptr = static_cast<const float*>(param.weight()->data());
    const float* weights_h_ptr = weights_x_ptr + word_dim * _inner_hidden_dim * 4;
    const float* weights_project_ptr = weights_h_ptr + _output_hidden_dim * _inner_hidden_dim * 4;
    int word_num = inputs[0]->num();
    const int skip_num = param.skip_num;
    _wx_gemm_fp32.init(false, false,word_num, 4 * _inner_hidden_dim, word_dim,ctx,weights_x_ptr,PACKED_MKLGEMM);
    _wh_gemm_fp32.init(false, false,skip_num, 4 * _inner_hidden_dim, _output_hidden_dim,ctx,weights_h_ptr,PACKED_MKLGEMM);
    _wp_gemm_fp32.init(false, false,skip_num, _output_hidden_dim, _inner_hidden_dim,ctx,weights_project_ptr,PACKED_MKLGEMM);
    return create(inputs, outputs, param, ctx);
} ;

template <typename BIT, typename OpDataType, bool first_iter>
static inline void cal_lstm_batch(int emit_word_id_size, OpDataType* temp_wx,
                                  const OpDataType* weight_peephole,
                                  OpDataType* hout, OpDataType* inner_cell, const OpDataType* b_i_in, const OpDataType* b_f_in,
                                  const OpDataType* b_c_in,
                                  const OpDataType* b_o_in, int hidden_size) {

    const int inner_iter_num = hidden_size / (sizeof(BIT) / sizeof(OpDataType));
    const BIT* b_i = (BIT*)b_i_in;
    const BIT* b_f = (BIT*)b_f_in;
    const BIT* b_c = (BIT*)b_c_in;
    const BIT* b_o = (BIT*)b_o_in;
    const int max_thread_nums=anakin_get_max_threads();
    for (int emit_word_id = 0; emit_word_id < emit_word_id_size; emit_word_id++) {
        int emit_wx_offset = emit_word_id * hidden_size * 4;
        const BIT* w_x_i = (BIT*)(temp_wx + 0 * hidden_size + emit_wx_offset);
        const BIT* w_x_f = (BIT*)(temp_wx + 1 * hidden_size + emit_wx_offset);
        const BIT* w_x_c = (BIT*)(temp_wx + 2 * hidden_size + emit_wx_offset);
        const BIT* w_x_o = (BIT*)(temp_wx + 3 * hidden_size + emit_wx_offset);

        const BIT* w_ci = (BIT*)(weight_peephole + 0 * hidden_size);
        const BIT* w_cf = (BIT*)(weight_peephole + 1 * hidden_size);
        const BIT* w_co = (BIT*)(weight_peephole + 2 * hidden_size);

        BIT* gate_h_p = (BIT*)(hout + emit_word_id * hidden_size);
        BIT* gate_c_p = (BIT*)(inner_cell + emit_word_id * hidden_size);

        if (first_iter) {
#pragma omp parallel for schedule(static) if (max_thread_nums > 1)
            for (int frame_id = 0; frame_id < inner_iter_num; ++frame_id) {
                BIT gate_i = Sigmoid(w_x_i[frame_id] + b_i[frame_id]);
                BIT gate_f = Sigmoid(w_x_f[frame_id] + b_f[frame_id]);
                BIT gate_c_s = Tanh(w_x_c[frame_id] + b_c[frame_id]);
                BIT gate_c = gate_i * gate_c_s;
                BIT gate_o = Sigmoid(w_x_o[frame_id] + gate_c * w_co[frame_id] + b_o[frame_id]);
                gate_c_p[frame_id] = gate_c;
                gate_h_p[frame_id] = gate_o * Tanh(gate_c);
            }
        } else {
#pragma omp parallel for schedule(static) if (max_thread_nums > 1)
            for (int frame_id = 0; frame_id < inner_iter_num; ++frame_id) {
                BIT c_1 = gate_c_p[frame_id];
                BIT gate_i = Sigmoid(w_x_i[frame_id] + b_i[frame_id] + w_ci[frame_id] * c_1);
                BIT gate_f = Sigmoid(w_x_f[frame_id] + b_f[frame_id] + w_cf[frame_id] * c_1);
                BIT gate_c_s = Tanh(w_x_c[frame_id] + b_c[frame_id]);
                BIT gate_c = gate_f * c_1 + gate_i * gate_c_s;
                BIT gate_o = Sigmoid(w_x_o[frame_id] + b_o[frame_id] + gate_c * w_co[frame_id]);

                gate_c_p[frame_id] = gate_c;
                gate_h_p[frame_id] = gate_o * Tanh(gate_c);
            }
        }
    }
}

template<>
SaberStatus SaberLstmp<X86, AK_FLOAT>::
dispatch(const std::vector<Tensor<X86>*>& inputs,
         std::vector<Tensor<X86>*>& outputs,
         LstmParam<X86>& param) {
    auto offset_vec = inputs[0]->get_seq_offset();
    CHECK_EQ(offset_vec.size(), 1);
    auto offset = offset_vec[0];
    CHECK_EQ(offset.size(), 2);
    const int skip_num = param.skip_num;
    CHECK_GT(skip_num, 1);
    int word_num = inputs[0]->num();
    int word_dim = inputs[0]->channel();
    int iter_num = utils::round_up(word_num, skip_num) / skip_num;

    utils::try_expand_tensor(_wx_tensor, word_num * 4 * _inner_hidden_dim);
    utils::try_expand_tensor(_temp_hidden_tensor, skip_num * _inner_hidden_dim);
    utils::try_expand_tensor(_temp_cell_tensor, skip_num * _inner_hidden_dim);

    float* wx_ptr = static_cast<float*>(_wx_tensor.mutable_data());
    const float* x_ptr = static_cast<const float*>(inputs[0]->data());
    const float* weights_x_ptr = static_cast<const float*>(param.weight()->data());
    const float* weights_h_ptr = weights_x_ptr + word_dim * _inner_hidden_dim * 4;
    const float* weights_project_ptr = weights_h_ptr + _output_hidden_dim * _inner_hidden_dim * 4;
    const float* weights_bias_ptr = static_cast<const float*>(param.bias()->data());
    const float* weights_bias_i_ptr = weights_bias_ptr;
    const float* weights_bias_f_ptr = weights_bias_i_ptr + _inner_hidden_dim;
    const float* weights_bias_c_ptr = weights_bias_f_ptr + _inner_hidden_dim;
    const float* weights_bias_o_ptr = weights_bias_c_ptr + _inner_hidden_dim;
    const float* weights_peephole_ptr = weights_bias_ptr + _inner_hidden_dim * 4;
    float* output_ptr = static_cast<float*>(outputs[0]->mutable_data());
    float* temp_hidden_out = static_cast<float*>(_temp_hidden_tensor.mutable_data());
    float* temp_cell_out = static_cast<float*>(_temp_cell_tensor.mutable_data());
//    gemm(false, false, word_num, 4 * _inner_hidden_dim, word_dim, 1.f, x_ptr, weights_x_ptr, 0.f,
//         wx_ptr);
    _wx_gemm_fp32.dispatch(1.f,0.f,word_num,x_ptr, weights_x_ptr,wx_ptr);

    for (int i = 0; i < iter_num; i++) {
        const int run_batch_dim = (i == (iter_num - 1)) ? (word_num - skip_num * i) : skip_num;
        float* wx_iter = wx_ptr + i * skip_num * 4 * _inner_hidden_dim;

        if (i >= 1) {
            float* hidden_in = output_ptr + (i - 1) * skip_num * _output_hidden_dim;
//            gemm(false, false, run_batch_dim, 4 * _inner_hidden_dim, _output_hidden_dim, 1.f, hidden_in,
//                 weights_h_ptr,
//                 1.f, wx_iter);
            _wh_gemm_fp32.dispatch(1.f,1.f,run_batch_dim,hidden_in,weights_h_ptr,wx_iter);

            cal_lstm_batch<SABER_X86_TYPE, float, false>(run_batch_dim, wx_iter, weights_peephole_ptr,
                    temp_hidden_out, temp_cell_out, weights_bias_i_ptr, weights_bias_f_ptr, weights_bias_c_ptr,
                    weights_bias_o_ptr, _inner_hidden_dim);

        } else {
            cal_lstm_batch<SABER_X86_TYPE, float, true>(run_batch_dim, wx_iter, weights_peephole_ptr,
                    temp_hidden_out, temp_cell_out, weights_bias_i_ptr, weights_bias_f_ptr, weights_bias_c_ptr,
                    weights_bias_o_ptr, _inner_hidden_dim);
        }

        float* hidden_out = output_ptr + i * skip_num * _output_hidden_dim;
//        gemm(false, false, run_batch_dim, _output_hidden_dim, _inner_hidden_dim, 1.f, temp_hidden_out,
//             weights_project_ptr, 0.f, hidden_out);
        _wp_gemm_fp32.dispatch(1.f,0.f,run_batch_dim,temp_hidden_out,weights_project_ptr,hidden_out);
        vsTanh(run_batch_dim * _output_hidden_dim, hidden_out, hidden_out);
    }

    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    return SaberSuccess;
}

template<>
SaberStatus SaberLstmp<X86, AK_INT8>:: create(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        LstmParam<X86>& param,
        Context<X86>& ctx) {

    return SaberSuccess;
};


template<>
SaberStatus SaberLstmp<X86, AK_INT8>::init(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        LstmParam<X86>& param,
        Context<X86>& ctx) {
    _inner_hidden_dim = param.cell_dim;
    _output_hidden_dim = param.project_dim;

    CHECK_GT(param.cell_dim, 0);
    CHECK_GT(param.project_dim, 0);
    CHECK_EQ(param.cell_dim % (sizeof(SABER_X86_TYPE) / sizeof(float)), 0);

    int word_num = inputs[0]->num();
    int word_channel = inputs[0]->channel();
    float* weights_x_ptr = static_cast<float*>(param.weight()->data());
    float* weights_h_ptr = weights_x_ptr + word_channel * _inner_hidden_dim * 4;
    float* weights_project_ptr = weights_h_ptr + _output_hidden_dim * _inner_hidden_dim * 4;
    float* weights_bias_ptr = static_cast<float*>(param.bias()->data());
    Shape shape_x({1, 1, word_num, word_channel});
    Shape shape_h({1, 1, param.skip_num, _output_hidden_dim});
    Shape shape_wh({1, 1, param.skip_num, 4 * _inner_hidden_dim});
    Shape shape_iter_project({1, 1, param.skip_num, _inner_hidden_dim});
    Shape shape_weights_wx({1, 1, word_channel, 4 * _inner_hidden_dim});
    Shape shape_weights_wh({1, 1, _output_hidden_dim, 4 * _inner_hidden_dim});
    Shape shape_weights_project({1, 1, _inner_hidden_dim, _output_hidden_dim});
    _inner_x_int8.re_alloc(shape_x, AK_INT8);
    _inner_h_int8.re_alloc(shape_h, AK_INT8);
    _inner_wh_int32.re_alloc(shape_wh, AK_INT32);
    _inner_project_scale.re_alloc(shape_iter_project, AK_INT8);
    _int8_weights_wx.re_alloc(shape_weights_wx, AK_INT8);
    _int8_weights_wh.re_alloc(shape_weights_wh, AK_INT8);
    _int8_weights_project.re_alloc(shape_weights_project, AK_INT8);
    utils::ScaleUtils::scale_gemm_xw_weights_to_nchw_host(_int8_weights_wx,
            Tensor<X86>(static_cast<void*>(weights_x_ptr), X86(), 0, shape_weights_wx, AK_FLOAT));
    utils::ScaleUtils::scale_gemm_xw_weights_to_nchw_host(_int8_weights_wh,
            Tensor<X86>(static_cast<void*>(weights_h_ptr), X86(), 0, shape_weights_wh, AK_FLOAT));
    utils::ScaleUtils::scale_gemm_xw_weights_to_nchw_host(_int8_weights_project,
            Tensor<X86>(static_cast<void*>(weights_project_ptr), X86(), 0, shape_weights_project, AK_FLOAT));

    auto input_scale = inputs[0]->get_scale();
    CHECK_EQ(input_scale.size(), 1);

    CHECK_EQ(_int8_weights_wx.get_scale().size(), 4 * _inner_hidden_dim);

    for (auto i : _int8_weights_wx.get_scale()) {
        _inner_scale_wx.push_back(input_scale[0]*i);
    }

    _inner_scale_wh.resize(4 * _inner_hidden_dim);
    _inner_scale_project.resize(_output_hidden_dim);
    //my intrinsic gemm init
    int word_dim = inputs[0]->channel();
    _wx_gemm_me.init(4 * _inner_hidden_dim, word_dim, _int8_weights_wx);
    _wh_gemm_me.init(4 * _inner_hidden_dim, _output_hidden_dim, _int8_weights_wh);
    _project_gemm_me.init(_output_hidden_dim, _inner_hidden_dim, _int8_weights_project);


    _temp_hidden_tensor.re_alloc(Shape({1, 1, param.skip_num, _inner_hidden_dim}), AK_FLOAT);
    _temp_cell_tensor.re_alloc(Shape({1, 1, param.skip_num, _inner_hidden_dim}), AK_FLOAT);


    int8_t* weights_x_int8_ptr = static_cast<int8_t*>(_int8_weights_wx.data());
    int8_t* weights_h_int8_ptr = static_cast<int8_t*>(_int8_weights_wh.data());
    int8_t* weights_p_int8_ptr = static_cast<int8_t*>(_int8_weights_project.data());

    if (jit::mayiuse(jit::avx512_core_vnni)) {
        _wx_gemm.init(false, false, word_num, 4 * _inner_hidden_dim, word_dim, ctx, weights_x_int8_ptr,PACKED_MKLGEMM);
        _wh_gemm.init(false, false, param.skip_num, 4 * _inner_hidden_dim, _output_hidden_dim, ctx,
                      weights_h_int8_ptr,PACKED_MKLGEMM);
        _wp_gemm.init(false, false, param.skip_num, _output_hidden_dim, _inner_hidden_dim, ctx,
                      weights_p_int8_ptr,PACKED_MKLGEMM);
    }

    LOG(INFO) << "create Lstmp";
    return create(inputs, outputs, param, ctx);
} ;


template<>
SaberStatus SaberLstmp<X86, AK_INT8>::
dispatch(const std::vector<Tensor<X86>*>& inputs,
         std::vector<Tensor<X86>*>& outputs,
         LstmParam<X86>& param) {
    if (jit::mayiuse(jit::avx512_core_vnni)) {
        auto offset_vec = inputs[0]->get_seq_offset();
        CHECK_EQ(offset_vec.size(), 1);
        auto offset = offset_vec[0];
        CHECK_EQ(offset.size(), 2);
        const int skip_num = param.skip_num;
        CHECK_GT(skip_num, 1);
        int word_num = inputs[0]->num();
        int word_dim = inputs[0]->channel();
        int iter_num = utils::round_up(word_num, skip_num) / skip_num;

        utils::try_expand_tensor(_wx_tensor, word_num * 4 * _inner_hidden_dim);
        utils::try_expand_tensor(_temp_hidden_tensor, skip_num * _inner_hidden_dim);
        utils::try_expand_tensor(_temp_cell_tensor, skip_num * _inner_hidden_dim);

        float* wx_ptr = static_cast<float*>(_wx_tensor.mutable_data());
        const float* x_ptr = static_cast<const float*>(inputs[0]->data());
        const int8_t* weights_x_ptr = static_cast<const int8_t*>(_int8_weights_wx.data());
        const int8_t* weights_h_ptr = static_cast<const int8_t*>(_int8_weights_wh.data());
        const int8_t* weights_project_ptr_int8 = static_cast<const int8_t*>(_int8_weights_project.data());
        const float* weights_project_ptr = static_cast<const float*>(param.weight()->data())
                                           + word_dim * _inner_hidden_dim * 4 +
                                           _output_hidden_dim * _inner_hidden_dim * 4;
        const float* weights_bias_ptr = static_cast<const float*>(param.bias()->data());
        const float* weights_bias_i_ptr = weights_bias_ptr;
        const float* weights_bias_f_ptr = weights_bias_i_ptr + _inner_hidden_dim;
        const float* weights_bias_c_ptr = weights_bias_f_ptr + _inner_hidden_dim;
        const float* weights_bias_o_ptr = weights_bias_c_ptr + _inner_hidden_dim;
        const float* weights_peephole_ptr = weights_bias_ptr + _inner_hidden_dim * 4;
        float* output_ptr = static_cast<float*>(outputs[0]->mutable_data());
        float* temp_hidden_out = static_cast<float*>(_temp_hidden_tensor.mutable_data());
        float* temp_cell_out = static_cast<float*>(_temp_cell_tensor.mutable_data());

        if (inputs[0]->get_dtype() == AK_FLOAT) {
            utils::ScaleUtils::scale_fp32_int8(_inner_x_int8, *inputs[0]);
            const int8_t* x_int8_ptr = static_cast<const int8_t*>(_inner_x_int8.data());
            _wx_gemm.dispatch(1.f, 0.f,word_num, x_int8_ptr, weights_x_ptr, (int32_t*) wx_ptr);
            utils::ScaleUtils::cvt_int32_fp32((int32_t*) wx_ptr, _inner_scale_wx, word_num,
                                              4 * _inner_hidden_dim);
        } else {
            LOG(FATAL) << "not impl";
        }

        for (int i = 0; i < iter_num; i++) {
            const int run_batch_dim = (i == (iter_num - 1)) ? (word_num - skip_num * i) : skip_num;
            float* wx_iter = wx_ptr + i * skip_num * 4 * _inner_hidden_dim;

            if (i >= 1) {
                float* hidden_in = output_ptr + (i - 1) * skip_num * _output_hidden_dim;
                utils::ScaleUtils::scale_fp32_int8(_inner_h_int8, hidden_in, run_batch_dim * _output_hidden_dim);
                float scale_x = _inner_h_int8.get_scale()[0];
                std::vector<float> scale_weights_h = _int8_weights_wh.get_scale();
                CHECK_EQ(scale_weights_h.size(), 4 * _inner_hidden_dim);

                for (int i = 0; i < 4 * _inner_hidden_dim; i++) {
                    _inner_scale_wh[i] = scale_x * scale_weights_h[i];
                }

                _wh_gemm.dispatch(1.f, 0.f,run_batch_dim, static_cast<int8_t*>(_inner_h_int8.data()), weights_h_ptr,
                                  static_cast<int*>(_inner_wh_int32.data()));
                utils::ScaleUtils::cvt_int32_fp32(static_cast<int*>(_inner_wh_int32.data()), _inner_scale_wh,
                                                  run_batch_dim,
                                                  4 * _inner_hidden_dim);
                float* wh_fp32 = static_cast<float*>(_inner_wh_int32.data());

                for (int i = 0; i < run_batch_dim * 4 * _inner_hidden_dim; i++) {
                    wx_iter[i] += wh_fp32[i];
                }

                cal_lstm_batch<SABER_X86_TYPE, float, false>(run_batch_dim, wx_iter, weights_peephole_ptr,
                        temp_hidden_out, temp_cell_out, weights_bias_i_ptr,
                        weights_bias_f_ptr, weights_bias_c_ptr,
                        weights_bias_o_ptr, _inner_hidden_dim);

            } else {
                cal_lstm_batch<SABER_X86_TYPE, float, true>(run_batch_dim, wx_iter, weights_peephole_ptr,
                        temp_hidden_out, temp_cell_out, weights_bias_i_ptr,
                        weights_bias_f_ptr, weights_bias_c_ptr,
                        weights_bias_o_ptr, _inner_hidden_dim);
            }

            float* hidden_out = output_ptr + i * skip_num * _output_hidden_dim;

            utils::ScaleUtils::scale_fp32_int8(_inner_project_scale, temp_hidden_out,
                                               run_batch_dim * _inner_hidden_dim);
            float scale_x = _inner_project_scale.get_scale()[0];
            std::vector<float> scale_vec = _int8_weights_project.get_scale();

            for (int i = 0; i < _output_hidden_dim; i++) {
                _inner_scale_project[i] = scale_x * scale_vec[i];
            }


            _wp_gemm.dispatch(1.f, 0.f,run_batch_dim, static_cast<int8_t*>(_inner_project_scale.data()),
                              weights_project_ptr_int8,
                              (int*) hidden_out);
            utils::ScaleUtils::cvt_int32_fp32((int*)(hidden_out), _inner_scale_project,
                                              run_batch_dim,
                                              _output_hidden_dim);

            vsTanh(run_batch_dim * _output_hidden_dim, hidden_out, hidden_out);
        }

        outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
        return SaberSuccess;
    } else {
        auto offset_vec = inputs[0]->get_seq_offset();
        CHECK_EQ(offset_vec.size(), 1);
        auto offset = offset_vec[0];
        CHECK_EQ(offset.size(), 2);
        const int skip_num = param.skip_num;
        CHECK_GT(skip_num, 1);
        int word_num = inputs[0]->num();
        int word_dim = inputs[0]->channel();
        int iter_num = utils::round_up(word_num, skip_num) / skip_num;


        utils::try_expand_tensor(_wx_tensor, word_num * 4 * _inner_hidden_dim);
        utils::try_expand_tensor(_temp_hidden_tensor, skip_num * _inner_hidden_dim);
        utils::try_expand_tensor(_temp_cell_tensor, skip_num * _inner_hidden_dim);

        float* wx_ptr = static_cast<float*>(_wx_tensor.mutable_data());
        const float* x_ptr = static_cast<const float*>(inputs[0]->data());
        const int8_t* weights_x_ptr = static_cast<const int8_t*>(_int8_weights_wx.data());
        const float* weights_h_ptr = static_cast<const float*>(param.weight()->data()) + word_dim *
                                     _inner_hidden_dim * 4;
        const float* weights_project_ptr = weights_h_ptr + _output_hidden_dim * _inner_hidden_dim * 4;
        const int8_t* weights_project_ptr_int8 = static_cast<const int8_t*>(_int8_weights_project.data());
        const float* weights_bias_ptr = static_cast<const float*>(param.bias()->data());
        const float* weights_bias_i_ptr = weights_bias_ptr;
        const float* weights_bias_f_ptr = weights_bias_i_ptr + _inner_hidden_dim;
        const float* weights_bias_c_ptr = weights_bias_f_ptr + _inner_hidden_dim;
        const float* weights_bias_o_ptr = weights_bias_c_ptr + _inner_hidden_dim;
        const float* weights_peephole_ptr = weights_bias_ptr + _inner_hidden_dim * 4;
        float* output_ptr = static_cast<float*>(outputs[0]->mutable_data());
        float* temp_hidden_out = static_cast<float*>(_temp_hidden_tensor.mutable_data());
        float* temp_cell_out = static_cast<float*>(_temp_cell_tensor.mutable_data());

        if (inputs[0]->get_dtype() == AK_FLOAT) {
            utils::ScaleUtils::scale_fp32_int8(_inner_x_int8, *inputs[0]);
            _wx_gemm_me.dispatch(word_num, 4 * _inner_hidden_dim, word_dim, _inner_x_int8, _wx_tensor);
            utils::ScaleUtils::cvt_int32_fp32((int32_t*)wx_ptr, _inner_scale_wx, word_num,
                                              4 * _inner_hidden_dim);

        } else {
            LOG(FATAL) << "not impl";
        }

        for (int i = 0; i < iter_num; i++) {
            const int run_batch_dim = (i == (iter_num - 1)) ? (word_num - skip_num * i) : skip_num;
            float* wx_iter = wx_ptr + i * skip_num * 4 * _inner_hidden_dim;

            if (i >= 1) {
                float* hidden_in = output_ptr + (i - 1) * skip_num * _output_hidden_dim;
                utils::ScaleUtils::scale_fp32_int8(_inner_h_int8, hidden_in, run_batch_dim * _output_hidden_dim);
                float scale_x = _inner_h_int8.get_scale()[0];
                std::vector<float> scale_weights_h = _int8_weights_wh.get_scale();
                CHECK_EQ(scale_weights_h.size(), 4 * _inner_hidden_dim);

                for (int i = 0; i < 4 * _inner_hidden_dim; i++) {
                    _inner_scale_wh[i] = scale_x * scale_weights_h[i];
                }

                _wh_gemm_me.dispatch(run_batch_dim, 4 * _inner_hidden_dim, _output_hidden_dim, _inner_h_int8,
                                     _inner_wh_int32);

                utils::ScaleUtils::cvt_int32_fp32(static_cast<int*>(_inner_wh_int32.data()), _inner_scale_wh,
                                                  run_batch_dim,
                                                  4 * _inner_hidden_dim);
                float* wh_fp32 = static_cast<float*>(_inner_wh_int32.data());

                for (int i = 0; i < run_batch_dim * 4 * _inner_hidden_dim; i++) {
                    wx_iter[i] += wh_fp32[i];
                }

                cal_lstm_batch<SABER_X86_TYPE, float, false>(run_batch_dim, wx_iter, weights_peephole_ptr,
                        temp_hidden_out, temp_cell_out, weights_bias_i_ptr, weights_bias_f_ptr, weights_bias_c_ptr,
                        weights_bias_o_ptr, _inner_hidden_dim);

            } else {
                cal_lstm_batch<SABER_X86_TYPE, float, true>(run_batch_dim, wx_iter, weights_peephole_ptr,
                        temp_hidden_out, temp_cell_out, weights_bias_i_ptr, weights_bias_f_ptr, weights_bias_c_ptr,
                        weights_bias_o_ptr, _inner_hidden_dim);
            }

            float* hidden_out = output_ptr + i * skip_num * _output_hidden_dim;

            utils::ScaleUtils::scale_fp32_int8(_inner_project_scale, temp_hidden_out,
                                               run_batch_dim * _inner_hidden_dim);
            float scale_x = _inner_project_scale.get_scale()[0];
            std::vector<float> scale_vec = _int8_weights_project.get_scale();

            for (int i = 0; i < _output_hidden_dim; i++) {
                _inner_scale_project[i] = scale_x * scale_vec[i];
            }

            Tensor<X86> temp_tensor(hidden_out, X86(), 0, Shape({1, 1, run_batch_dim, _output_hidden_dim}),
                                    AK_INT32);
            _project_gemm_me.dispatch(run_batch_dim, _output_hidden_dim, _inner_hidden_dim,
                                      _inner_project_scale, temp_tensor);
            utils::ScaleUtils::cvt_int32_fp32((int*)(hidden_out), _inner_scale_project,
                                              run_batch_dim,
                                              _output_hidden_dim);

            vsTanh(run_batch_dim * _output_hidden_dim, hidden_out, hidden_out);
        }

        outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
        return SaberSuccess;
    }
}


DEFINE_OP_TEMPLATE(SaberLstmp, LstmParam, X86, AK_HALF);

}
}
