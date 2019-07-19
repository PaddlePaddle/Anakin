#include "saber/funcs/impl/x86/saber_gru.h"
#include "saber/core/tensor_op.h"
#include "mkl_cblas.h"
#include "saber/funcs/impl/x86/saber_normal_activation.h"
#ifdef USE_SGX
extern "C" void mkl_free_buffers();
#endif

namespace anakin {

namespace saber {

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
    cblas_sgemm(CblasRowMajor, cuTransA, cuTransB, m, n, k, alpha, a, k, b, n, beta, c, n);
#ifdef USE_SGX
    mkl_free_buffers();
#endif
};

template <typename BIT>
static inline void cal_gru_reset_gate(float* hout,const float* hin,const BIT* b_r,const float* temp_wx,const float* temp_wh,
                  int emit_word_id_start,int emit_word_id_end,int _aligned_hidden_size,int r_offset,BIT(*gate_act)(const BIT)){

    for (int emit_word_id = emit_word_id_start; emit_word_id < emit_word_id_end; emit_word_id++) {
        int emit_id_offset = emit_word_id - emit_word_id_start;
        BIT* w_x_r = (BIT*)(temp_wx + r_offset * _aligned_hidden_size
                            + emit_word_id * _aligned_hidden_size * 3);
        BIT* w_h_r = (BIT*)(temp_wh + 0 * _aligned_hidden_size
                            + emit_id_offset * _aligned_hidden_size * 2);
        BIT* emit_hout = (BIT*)(hout + emit_id_offset * _aligned_hidden_size);
        const BIT* emit_hin = (BIT*)(hin + emit_id_offset * _aligned_hidden_size);

        for (int frame_id = 0; frame_id < _aligned_hidden_size / (sizeof(BIT) / 4); ++frame_id) {
            BIT r;
            r = w_x_r[frame_id] + w_h_r[frame_id] + b_r[frame_id]; //h_out=gate_r
            r = gate_act(r);
//            LOG(INFO)<<"SABER:"<<emit_word_id<<","<<frame_id<<"::"<<r[0]<<"="<<w_x_r[frame_id][0]<<","<<w_h_r[frame_id][0]<<","<<b_r[frame_id][0];
            emit_hout[frame_id] = r * emit_hin[frame_id];
        }
    }
}

template <typename BIT>
static inline void cal_gru_forgate_output_gate(float* hout,const float* hin,const BIT* b_z,const BIT* b_o,const float* temp_wx,const float* temp_whr,
                   const float* temp_wh,int emit_word_id_start,int emit_word_id_end,int _aligned_hidden_size,int z_offset,int o_offset,BIT(*gate_act)(const BIT),BIT(*hid_act)(const BIT) ){


    for (int emit_word_id = emit_word_id_start; emit_word_id < emit_word_id_end; emit_word_id++) {

        int emit_offset = emit_word_id - emit_word_id_start;
        BIT* w_x_z = (BIT*)(temp_wx + z_offset * _aligned_hidden_size
                            + emit_word_id * _aligned_hidden_size * 3);
        BIT* w_x_o = (BIT*)(temp_wx + o_offset * _aligned_hidden_size
                            + emit_word_id * _aligned_hidden_size * 3);

        BIT* w_h_z = (BIT*)(temp_wh + 1 * _aligned_hidden_size
                            + emit_offset * _aligned_hidden_size * 2);

        BIT* w_h_o = (BIT*)(temp_whr + emit_offset * _aligned_hidden_size);
        BIT* emit_hout = (BIT*)(hout + emit_offset * _aligned_hidden_size) ;
        const BIT* emit_hin = (BIT*)(hin + emit_offset * _aligned_hidden_size) ;

        for (int frame_id = 0; frame_id < _aligned_hidden_size / (sizeof(BIT) / 4); ++frame_id) {
            BIT z;
            BIT _h;
            z = gate_act(w_x_z[frame_id] + w_h_z[frame_id] + b_z[frame_id]);
            _h = w_x_o[frame_id] + w_h_o[frame_id] + b_o[frame_id];
            _h = hid_act(_h);
            emit_hout[frame_id] = (1 - z) * emit_hin[frame_id] + z * _h;
        }
    }

}

template <typename BIT>
static inline void cal_gru_cudnn(float* hout,const float* hin,const BIT* b_o,const BIT* b_r,const BIT* b_z,
                                 const float* temp_wx,const float* temp_wh,
                                 int emit_word_id_start,int emit_word_id_end,int _aligned_hidden_size,
                                 int o_offset,int r_offset,int z_offset){

    for (int emit_word_id = emit_word_id_start; emit_word_id < emit_word_id_end; emit_word_id++) {
        int emit_id_offset = emit_word_id - emit_word_id_start;

        BIT* w_x_r = (BIT*)(temp_wx + r_offset * _aligned_hidden_size
                            + emit_word_id * _aligned_hidden_size * 3);
        BIT* w_h_r = (BIT*)(temp_wh + r_offset * _aligned_hidden_size
                            + emit_id_offset * _aligned_hidden_size * 3);

        BIT* w_x_z = (BIT*)(temp_wx + z_offset * _aligned_hidden_size
                            + emit_word_id * _aligned_hidden_size * 3);
        BIT* w_h_z = (BIT*)(temp_wh + z_offset * _aligned_hidden_size
                            + emit_id_offset * _aligned_hidden_size * 3);

        BIT* w_x_o = (BIT*)(temp_wx + o_offset * _aligned_hidden_size
                            + emit_word_id * _aligned_hidden_size * 3);
        BIT* w_h_o = (BIT*)(temp_wh + o_offset * _aligned_hidden_size
                            + emit_id_offset * _aligned_hidden_size * 3);

        BIT* emit_hout = (BIT*)(hout + emit_id_offset * _aligned_hidden_size) ;
        const BIT* emit_hin = (BIT*)(hin + emit_id_offset * _aligned_hidden_size);

        for (int frame_id = 0; frame_id < _aligned_hidden_size / (sizeof(BIT) / 4); ++frame_id) {
            BIT r = Sigmoid(w_x_r[frame_id] + w_h_r[frame_id] + b_r[frame_id]); //h_out=gate_r
            BIT z = Sigmoid(w_x_z[frame_id] + w_h_z[frame_id] + b_z[frame_id]);
            BIT _h =Tanh( w_x_o[frame_id] + r * w_h_o[frame_id] + b_o[frame_id]);
            emit_hout[frame_id] = z * emit_hin[frame_id] + (1 - z) * _h;
//            LOG(INFO)<<"target "<<emit_word_id<<","<<frame_id<<" : "<<r[0]<<","<<z[0]<<","<<_h[0]<<" : "<<r[1]<<","<<z[1]<<","<<_h[1];
        }
    }
}


template <>
template<typename BIT>
SaberStatus SaberGru<X86, AK_FLOAT>::
batch_s_aligned(const std::vector<OpTensor*>& inputs,
                std::vector<OpTensor*>& outputs,
                GruParam<X86>& param) {
//    CHECK_NE(param.formula, GRU_CUDNN) << "X86 gru not support cudnn formula now";
    int loop_div = sizeof(BIT) / sizeof(float);
    //    LOG(INFO)<<"loop_div "<<loop_div;
    const OpDataType* weight_h = (const OpDataType*)_aligned_weights_h2h.data();
    const OpDataType* weight_w = (const OpDataType*)_aligned_weights_i2h.data();
    const OpDataType* bias = (const OpDataType*)_aligned_weights_bias.data();

    BIT(*gate_act)(const BIT) = Activate_inner<BIT>(param.gate_activity);
    BIT(*hid_act)(const BIT) =Activate_inner<BIT>(param.h_activity);
    std::vector<std::vector<int>> offset_vec_vec = inputs[0]->get_seq_offset();
    CHECK(offset_vec_vec.size()>=1)<<"offset_vec_vec size should >=1";
    std::vector<int>offset_vec=offset_vec_vec[offset_vec_vec.size()-1];
    CHECK(offset_vec.size()>=2)<<"offset_vec size should >=2";
    std::vector<int> length_vec(offset_vec.size() - 1);
    int batch_size = offset_vec.size() - 1;
    int seqsum = 0;
    int max_seq_len = 0;
    bool is_hw2seq = offset_vec.size() > 2;
    int word_sum = is_hw2seq ? offset_vec[offset_vec.size() - 1] : inputs[0]->channel();
    utils::AlignedUtils aligned_utils;
    utils::VectorPrint vector_print;
    const OpDataType* h_init = nullptr;

    const OpDataType* x = (const OpDataType*)inputs[0]->data();
    OpDataType* out = ( OpDataType*)outputs[0]->mutable_data();
    bool is_reverse = param.is_reverse;

    if (inputs.size() > 1) {
        h_init = (const OpDataType*)inputs[1]->data();
        utils::try_expand_tensor(_aligned_init_hidden,batch_size * _aligned_hidden_size);
        aligned_utils.aligned_last_dim(h_init, (OpDataType*)_aligned_init_hidden.mutable_data(),
                                       batch_size * _hidden_size, _hidden_size, _aligned_hidden_size);
        h_init = (const OpDataType*)_aligned_init_hidden.data();
    } else if (param.init_hidden() != nullptr) {
        h_init = (const OpDataType*)param.init_hidden()->data();
        //FIXME:is it correct?
    } else {
        bool need_clear=utils::try_expand_tensor(_aligned_init_hidden,batch_size * _aligned_hidden_size);
        if(need_clear){
            memset(_aligned_init_hidden.mutable_data(),0,_aligned_init_hidden.valid_shape().count()* sizeof(OpDataType));
        }
        h_init = (const OpDataType*)_aligned_init_hidden.data();
    }

    std::vector<int> emit_offset_vec;
    int emit_length = 0;
    utils::SeqSortedseqTranseUtil transe_util(is_reverse);
    bool transform = transe_util.get_sorted_map(offset_vec, emit_offset_vec, emit_length);

    OpDataType* inner_h_out = out;
    OpDataType* inner_x = (OpDataType*)x;
    const OpDataType* inner_h_init = h_init;

    for (int i = 0; i < offset_vec.size() - 1; ++i) {
        int len = offset_vec[i + 1] - offset_vec[i];
        length_vec[i] = len;
        max_seq_len = max_seq_len > len ? max_seq_len : len;
        seqsum += len;
    }

    utils::try_expand_tensor(_temp_wx,seqsum * 3 * _aligned_hidden_size);
    utils::try_expand_tensor(_temp_out,seqsum * _aligned_hidden_size * param.num_direction);
    if(param.formula==GRU_ORIGIN) {
        utils::try_expand_tensor(_temp_wh, batch_size * 2 * _aligned_hidden_size);
        utils::try_expand_tensor(_temp_whr, batch_size * _aligned_hidden_size);
    }else{
        utils::try_expand_tensor(_temp_wh, batch_size * 3 * _aligned_hidden_size);
    }


    if (transform) {
        utils::try_expand_tensor(_temp_x,seqsum * _word_size);
        inner_h_out = (OpDataType*)_temp_out.mutable_data();
        inner_x = (OpDataType*)_temp_x.mutable_data();
        transe_util.seq_2_sorted_seq(x, inner_x, _word_size);

        if (inner_h_init != nullptr) {
            utils::try_expand_tensor(_temp_h_init,batch_size * _aligned_hidden_size);
            transe_util.hidden_2_sorted_hidden(inner_h_init, (OpDataType*)_temp_h_init.mutable_data(), _aligned_hidden_size);
            inner_h_init = (const OpDataType*)_temp_h_init.data();
        }

    } else if (_hidden_size != _aligned_hidden_size) {
        inner_h_out = (OpDataType*)_temp_out.mutable_data();
    }

    OpDataType* temp_wh = (OpDataType*)_temp_wh.mutable_data();
    OpDataType* temp_wx = (OpDataType*)_temp_wx.mutable_data();
    OpDataType* temp_whr = (OpDataType*)_temp_whr.mutable_data();
    /////////////////////////////////////////////////
    //wx

    gemm(false, false, seqsum, 3 * _aligned_hidden_size, _word_size, 1.f, inner_x, weight_w, 0.f,
         temp_wx);


    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;
    const BIT* b_r = (BIT*)(bias + r_offset * _aligned_hidden_size);
    const BIT* b_z = (BIT*)(bias + z_offset * _aligned_hidden_size);
    const BIT* b_o = (BIT*)(bias + o_offset * _aligned_hidden_size);



    int reverse_out_offset = seqsum;


    for (int word_id = 0; word_id < emit_length; word_id++) {
        int real_word_id = word_id;
        int last_word_id = word_id - 1;

        if (param.is_reverse && batch_size == 1) {
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
            hin = inner_h_out + emit_offset_vec[last_word_id] * _aligned_hidden_size;
        }

        float* hout = nullptr;
        hout = emit_offset_vec[real_word_id] * _aligned_hidden_size + inner_h_out;

        if(param.formula==GRU_ORIGIN) {
            //wh
            gemm(false, false, emit_word_length, 2 * _aligned_hidden_size, _aligned_hidden_size, 1.0, hin,
                 weight_h ,
                 0.f, temp_wh);


            //#pragma omp parallel for
            cal_gru_reset_gate<BIT>(hout, hin, b_r, temp_wx, temp_wh, emit_word_id_start, emit_word_id_end,
                                    _aligned_hidden_size, r_offset, gate_act);


            gemm(false, false, emit_word_length, _aligned_hidden_size, _aligned_hidden_size, 1.0, hout,
                 static_cast<const OpDataType*>(_aligned_weights_h2h_o.data()), 0.f, temp_whr);

            cal_gru_forgate_output_gate<BIT>(hout, hin, b_z, b_o, temp_wx, temp_whr, temp_wh, emit_word_id_start,
                                             emit_word_id_end,
                                             _aligned_hidden_size, z_offset, o_offset, gate_act, hid_act);
        }else if(param.formula==GRU_CUDNN){

            gemm(false, false, emit_word_length, 3 * _aligned_hidden_size, _aligned_hidden_size, 1.0, hin,
                 weight_h,
                 0.f, temp_wh);

            cal_gru_cudnn<BIT>(hout,hin,b_o,b_r,b_z,temp_wx,temp_wh,emit_word_id_start,emit_word_id_end,
                               _aligned_hidden_size,o_offset,r_offset,z_offset);
        }

    }

    if (transform) {
        transe_util.sorted_seq_2_seq(inner_h_out, out, _hidden_size, _aligned_hidden_size);
    } else if (_hidden_size != _aligned_hidden_size) {
        aligned_utils.unaligned_last_dim((const OpDataType*)_temp_out.data(), out, seqsum * _hidden_size, _hidden_size,
                                         _aligned_hidden_size);
    }

    return SaberSuccess;
};


template<>
SaberStatus SaberGru<X86, AK_FLOAT>::dispatch(\
        const std::vector<OpTensor*>& inputs,
        std::vector<OpTensor*>& outputs,
        GruParam<X86>& param) {

    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());

    batch_s_aligned<SABER_X86_TYPE >(inputs, outputs, param);

    return SaberSuccess;


};

template class SaberGru<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberGru, GruParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberGru, GruParam, X86, AK_INT8);
}
}
