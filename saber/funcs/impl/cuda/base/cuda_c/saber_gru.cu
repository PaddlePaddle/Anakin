#include "saber/funcs/impl/cuda/saber_gru.h"
#include "saber/core/tensor_op.h"
#include "cuda_inline_activation.h"
namespace anakin {

namespace saber {


template <typename Dtype>
__global__ void cal_reset_kernel(Dtype* w_x_r,Dtype* w_h_r,const Dtype* b_r,int hidden_size, Dtype* output,
                                 const Dtype* hidden_pre,const ActiveType gate_activity) {
    int index = threadIdx.x;
    if (index > hidden_size) {
        return;
    }
    int w_base_index = blockIdx.x * hidden_size * 3 + index;
    int u_base_index = blockIdx.x * hidden_size * 2 + index;
    int h_base_index = blockIdx.x * hidden_size + index;
    Dtype hidden_pre_value = hidden_pre[h_base_index];
    Dtype before_act_r = w_x_r[w_base_index] + w_h_r[u_base_index] + b_r[index];
    Dtype act_r = activate_cuda_float(before_act_r,gate_activity);
    output[h_base_index] = hidden_pre_value * act_r;
};


template <typename Dtype>
__global__ void cal_final_kernel( Dtype* w_x_z, Dtype* w_x_o,Dtype* w_h_z,const Dtype* b_z, const Dtype* b_o,
        int hidden_size, Dtype* output, const Dtype* hidden_pre,const Dtype* w_h_o,
                                  const ActiveType gate_activity,const ActiveType h_activity) {
    int index = threadIdx.x;
    if (index > hidden_size) {
        return;
    }

    int w_base_index = blockIdx.x * hidden_size * 3 + index;
    int u_base_index = blockIdx.x * hidden_size * 2 + index;
    int h_base_index = blockIdx.x * hidden_size + index;
    Dtype hidden_pre_value = hidden_pre[h_base_index];
    Dtype before_act_z = w_x_z[w_base_index] + w_h_z[u_base_index] + b_z[index];
    Dtype act_z =  activate_cuda_float(before_act_z,gate_activity);
    Dtype before_act_h = w_x_o[w_base_index] + w_h_o[h_base_index]
                         + b_o[index];
    Dtype acted = activate_cuda_float(before_act_h,h_activity);

    output[h_base_index] = (static_cast<Dtype>(1.0) - act_z) * hidden_pre_value + act_z * acted;
}



template<>
SaberStatus SaberGru<NV, AK_FLOAT>::dispatch(\
        const std::vector<OpTensor*>& inputs,
        std::vector<OpTensor*>& outputs,
        GruParam <NV>& param) {
    CHECK_GE(param.formula,GRU_ORIGIN)<<"ONLY SUPPORT GRU_ORIGIN NOW";
    OpTensor* x = inputs[0];
    std::vector<std::vector<int>> offset_vec_vec = x->get_seq_offset();
    std::vector<int> offset = offset_vec_vec[offset_vec_vec.size()-1];

    const OpDataType* x_data = static_cast<const OpDataType*>(x->data());
    const OpDataType* h;
    OpTensor* dout = outputs[0];
    OpDataType* dout_data = static_cast<OpDataType*>(dout->mutable_data());

    const OpDataType* weights_i2h=static_cast<const OpDataType*>(param.weight()->data());
    const OpDataType* weights_h2h=weights_i2h+3*_hidden_size*_word_size;
    const OpDataType* weights_bias=static_cast<const OpDataType*>(param.bias()->data());

    int batch_size = offset.size() - 1;
    int seq_sum = x->num();
    bool is_batched = offset.size() > 2;
    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;

    std::vector<int> emit_offset_vec;
    int emit_length = 0;
    utils::try_expand_tensor(_temp_map_dev,seq_sum);
    is_batched = _seq_util.get_sorted_map(offset, emit_offset_vec, emit_length,
                                        _ctx->get_compute_stream());

    if (is_batched) {
        Shape seq_shape({1, 1, seq_sum, _word_size});
        utils::try_expand_tensor(_temp_tensor_in,seq_shape);
        Shape seq_out_shape({1, 1, seq_sum, _hidden_size});
        utils::try_expand_tensor(_temp_tensor_out,seq_out_shape);
        _seq_util.seq_2_sorted_seq(x_data, static_cast<OpDataType*>(_temp_tensor_in.mutable_data()), _word_size,
                                   _ctx->get_compute_stream());
        x_data = static_cast<const OpDataType*>(_temp_tensor_in.data());
        dout_data = static_cast<OpDataType*>(_temp_tensor_out.mutable_data());
    }

    Shape shape_wx({seq_sum, 1, 3, _hidden_size});
    utils::try_expand_tensor(_temp_wx,shape_wx);

    Shape shape_wh({1, batch_size, 2, _hidden_size});
    utils::try_expand_tensor(_temp_wh,shape_wh);

    Shape shape_whr({1, batch_size, 1, _hidden_size});
    utils::try_expand_tensor(_temp_whr,shape_whr);

    _gemm_wx(seq_sum, 3 * _hidden_size, _word_size, 1.f, x_data, 0.f, weights_i2h,
             static_cast<OpDataType*>(_temp_wx.mutable_data()), _ctx->get_compute_stream());

    const OpDataType* b_r = weights_bias + r_offset * _hidden_size;
    const OpDataType* b_z = weights_bias + z_offset * _hidden_size;
    const OpDataType* b_o = weights_bias + o_offset * _hidden_size;

    if (inputs.size() == 1) {
        if (_temp_zero.valid_size() < batch_size * _hidden_size) {
            utils::try_expand_tensor(_temp_zero,batch_size * _hidden_size);
            CUDA_CHECK(cudaMemsetAsync(_temp_zero.mutable_data(), 0,
                                       sizeof(OpDataType)*batch_size * _hidden_size,
                                       _ctx->get_compute_stream()));
        }
        h = static_cast<const OpDataType*>(_temp_zero.data());
    } else {
        h = static_cast<const OpDataType*>(inputs[1]->data());
    }


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

        const OpDataType* hidden_in;
        OpDataType* hidden_out = dout_data + emit_offset_vec[real_word_id] * _hidden_size;

        if (word_id == 0) {
            hidden_in = h;
        } else {
            hidden_in = dout_data + emit_offset_vec[last_word_id] * _hidden_size;
        }

        _gemm_wh_2(emit_word_length, 2 * _hidden_size, _hidden_size, 1.f, hidden_in, 0.f,
                   weights_h2h + _hidden_size * _hidden_size,static_cast<OpDataType*>( _temp_wh.mutable_data()),
                   _ctx->get_compute_stream());

        OpDataType* w_x_r = static_cast<OpDataType*>(_temp_wx.mutable_data()) + r_offset * _hidden_size
                             + emit_word_id_start * _hidden_size * 3;
        OpDataType* w_x_z = static_cast<OpDataType*>(_temp_wx.mutable_data()) + z_offset * _hidden_size
                             + emit_word_id_start * _hidden_size * 3;
        OpDataType* w_x_o = static_cast<OpDataType*>(_temp_wx.mutable_data()) + o_offset * _hidden_size
                             + emit_word_id_start * _hidden_size * 3;

        OpDataType* w_h_r = static_cast<OpDataType*>(_temp_wh.mutable_data()) + 0 * _hidden_size;
        OpDataType* w_h_z = static_cast<OpDataType*>(_temp_wh.mutable_data()) + 1 * _hidden_size;


        const OpDataType* w_o = weights_h2h;
        CHECK_LE(_hidden_size, 1024) << "now not support hidden size > 1024 for paddle formula";
        int frame_per_block = _hidden_size <= 1024 ? _hidden_size : 1024;

        cal_reset_kernel<< < emit_word_length, frame_per_block, 0
                                       , _ctx->get_compute_stream() >> > (
                                           w_x_r, w_h_r
                                           , b_r, _hidden_size, hidden_out, hidden_in,param.gate_activity);

        _gemm_wh_o(emit_word_length, _hidden_size, _hidden_size, 1.f, hidden_out, 0.f, w_o,
                   static_cast<OpDataType*>(_temp_whr.mutable_data()), _ctx->get_compute_stream());

        cal_final_kernel<< < emit_word_length, frame_per_block, 0
                , _ctx->get_compute_stream() >> > (
                    w_x_z, w_x_o, w_h_z, b_z, b_o, _hidden_size, hidden_out, hidden_in, static_cast<const OpDataType*>(_temp_whr.data()),
                            param.gate_activity,param.h_activity);

    }

    if (is_batched) {
        _seq_util.sorted_seq_2_seq(static_cast<const OpDataType*>(_temp_tensor_out.data()), static_cast<OpDataType*>(dout->mutable_data()), _hidden_size,
                                   _ctx->get_compute_stream());
    }

    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    return SaberSuccess;
}
template class SaberGru<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberGru, GruParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberGru, GruParam, NV, AK_INT8);
}
}

