#include "saber/funcs/impl/cuda/saber_gru.h"
#include "saber/core/tensor_op.h"
#include "cuda_inline_activation.h"
namespace anakin {

namespace saber {


static void anakin_NV_gemm(cublasHandle_t handle, const bool TransA,
                           const bool TransB, const int M, const int N, const int K,
                           const float alpha, const float* A, const float* B, const float beta,
                           float* C) {
    // Note that cublas follows fortran order.
    int lda = (!TransA/* == CblasNoTrans*/) ? K : M;
    int ldb = (!TransB/* == CblasNoTrans*/) ? N : K;
    cublasOperation_t cuTransA =
        (!TransA/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        (!TransB/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA,
                             N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <typename Dtype>
__global__ void cal_one_kernel_paddlesigmoid_tanh_cudnn_formula(Dtype* w_x_r, Dtype* w_x_z,
        Dtype* w_x_o,
        Dtype* w_h_r, Dtype* w_h_z, Dtype* w_h_o,
        const Dtype* b_r, const Dtype* b_z, const Dtype* b_o,
        int hidden_size, Dtype* output, const Dtype* hidden_pre) {
    int w_base_index = blockIdx.x * hidden_size * 3;
    int h_base_index = blockIdx.x * hidden_size;
    Dtype* in_w_x_r = w_x_r + w_base_index;
    Dtype* in_w_h_r = w_h_r + w_base_index;
    Dtype* in_w_x_z = w_x_z + w_base_index;
    Dtype* in_w_h_z = w_h_z + w_base_index;
    Dtype* in_w_x_o = w_x_o + w_base_index;
    Dtype* in_w_h_o = w_h_o + w_base_index;
    const Dtype* in_hidden_pre = hidden_pre + h_base_index;
    Dtype* out_output = output + h_base_index;

    for (int index = threadIdx.x; index < hidden_size; index += blockDim.x) {
        const Dtype min = SIGMOID_THRESHOLD_MIN_PADDLE;
        const Dtype max = SIGMOID_THRESHOLD_MAX_PADDLE;

        Dtype before_act_r = in_w_x_r[index] + in_w_h_r[index] + b_r[index];
        before_act_r = (before_act_r < min) ? min : ((before_act_r > max) ? max : before_act_r);
        Dtype act_r = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-before_act_r));

        Dtype before_act_z = in_w_x_z[index] + in_w_h_z[index] + b_z[index];
        before_act_z = (before_act_z < min) ? min : ((before_act_z > max) ? max : before_act_z);
        Dtype act_z = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-before_act_z));

        Dtype before_act_h = in_w_x_o[index] + in_w_h_o[index] * act_r
                             + b_o[index];
        before_act_h = (before_act_h > EXP_MAX_INPUT_PADDLE) ? EXP_MAX_INPUT_PADDLE : before_act_h;
        Dtype acted = tanhf(before_act_h);
        out_output[index] = (1 - act_z) * acted + act_z * in_hidden_pre[index];
    }
}

template <typename Dtype>
__global__ void cal_one_kernel_sigmoid_tanh_modi_cudnn_formula(Dtype* w_x_r, Dtype* w_x_z,
        Dtype* w_x_o,
        Dtype* w_h_r, Dtype* w_h_z, Dtype* w_h_o,
        const Dtype* b_r, const Dtype* b_z, const Dtype* b_o,
        int hidden_size, Dtype* output, const Dtype* hidden_pre) {

    int w_base_index = blockIdx.x * hidden_size * 3 + threadIdx.x;
    int h_base_index = blockIdx.x * hidden_size + threadIdx.x;

    for (int index = threadIdx.x; index < hidden_size;
            index += blockDim.x, w_base_index += blockDim.x, h_base_index += blockDim.x) {
        Dtype before_act_r = w_x_r[w_base_index] + w_h_r[w_base_index] + b_r[index];
        Dtype act_r = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-before_act_r));
        Dtype before_act_z = w_x_z[w_base_index] + w_h_z[w_base_index] + b_z[index];
        Dtype act_z = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-before_act_z));
        Dtype before_act_h = w_x_o[w_base_index] + w_h_o[w_base_index] * act_r
                             + b_o[index];
        Dtype acted = tanh(before_act_h);
        output[h_base_index] = (static_cast<Dtype>(1.0) - act_z) * acted + act_z * hidden_pre[h_base_index];
    }
}


#define CAL_KERNEL_DEFINE(GATACTNAME)\
template <typename Dtype>\
__global__ void cal_reset_kernel##GATACTNAME(Dtype* w_x_r,Dtype* w_h_r,const Dtype* b_r,int hidden_size, Dtype* output, const Dtype* hidden_pre) {\
    int index = threadIdx.x;\
    if (index > hidden_size) {\
        return;\
    }\
    int w_base_index = blockIdx.x * hidden_size * 3 + index;\
    int u_base_index = blockIdx.x * hidden_size * 2 + index;\
    int h_base_index = blockIdx.x * hidden_size + index;\
    Dtype hidden_pre_value = hidden_pre[h_base_index];\
    Dtype before_act_r = w_x_r[w_base_index] + w_h_r[u_base_index] + b_r[index];\
    Dtype act_r = GATACTNAME(before_act_r);\
    output[h_base_index] = hidden_pre_value * act_r;\
};


#define FINAL_KERNEL_DEFINE(GATACTNAME,OUTACTNAME)\
template <typename Dtype>\
__global__ void cal_final_kernel##GATACTNAME##OUTACTNAME( Dtype* w_x_z, Dtype* w_x_o,Dtype* w_h_z,const Dtype* b_z, const Dtype* b_o,\
        int hidden_size, Dtype* output, const Dtype* hidden_pre,const Dtype* w_h_o) {\
    int index = threadIdx.x;\
    if (index > hidden_size) {\
        return;\
    }\
\
    int w_base_index = blockIdx.x * hidden_size * 3 + index;\
    int u_base_index = blockIdx.x * hidden_size * 2 + index;\
    int h_base_index = blockIdx.x * hidden_size + index;\
    Dtype hidden_pre_value = hidden_pre[h_base_index];\
    Dtype before_act_z = w_x_z[w_base_index] + w_h_z[u_base_index] + b_z[index];\
    Dtype act_z =  GATACTNAME(before_act_z);\
    Dtype before_act_h = w_x_o[w_base_index] + w_h_o[h_base_index]\
                         + b_o[index];\
    Dtype acted = OUTACTNAME(before_act_h);\
\
    output[h_base_index] = (static_cast<Dtype>(1.0) - act_z) * hidden_pre_value + act_z * acted;\
}

#define RESET_KERNEL_NAME(GATACTNAME) cal_reset_kernel##GATACTNAME
#define FINAL_KERNEL_NAME(GATACTNAME,OUTACTNAME) cal_final_kernel##GATACTNAME##OUTACTNAME

CAL_KERNEL_DEFINE(sigmoid);

CAL_KERNEL_DEFINE(sigmoid_fluid);

FINAL_KERNEL_DEFINE(sigmoid_fluid,tanh_fluid);

FINAL_KERNEL_DEFINE(sigmoid_fluid,relu);

#if 0
template <>
SaberStatus SaberGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::gru_cudnn(
    const std::vector<DataTensor_in*> inputs,
    std::vector<DataTensor_out*> outputs,
    GruParam<OpTensor>& param) {

    DataTensor_in* x = inputs[0];
    const InDataType* x_data = x->data();
    std::vector<int> offset=x->get_seq_offset();
    const InDataType* h;
    DataTensor_out* dout = outputs[0];
    OutDataType* dout_data = dout->mutable_data();

    //TODO:check shape first
    const OpTensor* b = param.bias();

    int batch_size = offset.size() - 1;; //x->get_seq_offset().size()-1;
    int sequence = x->num();
    int hidden_size = b->valid_size() / 3;
    bool isHW2Seq=offset.size()>2;
    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;


    if (isHW2Seq) {
        x_data = hw2seq(inputs, param, _word_size, hidden_size, sequence);
        batch_size = offset.size() - 1;

        if (x_data != x->data()) {
            dout_data = _temp_tensor_out.mutable_data();
        }
    }

    Shape shape_wx(sequence, batch_size, 3, hidden_size);
    _temp_WX.try_expand_size(shape_wx);

    Shape shape_wh(1, batch_size, 3, hidden_size);
    _temp_WH.try_expand_size(shape_wh);

    anakin_NV_gemm(_cublas_handle, false, false, sequence * batch_size, 3 * hidden_size,
                   _word_size, 1.0, x_data, _weights_i2h.data(), 0.0, _temp_WX.mutable_data());




    const OpDataType* b_r = b->data() + r_offset * hidden_size;
    const OpDataType* b_z = b->data() + z_offset * hidden_size;
    const OpDataType* b_o = b->data() + o_offset * hidden_size;

    if (inputs.size() == 1) {
        CUDA_CHECK(cudaMemsetAsync(dout_data, 0, sizeof(InDataType) * batch_size * hidden_size,
                                   _ctx->get_compute_stream()));
        h = dout_data;
    } else {
        h = inputs[1]->data();
        CHECK_EQ(inputs[1]->valid_size(), batch_size * hidden_size) <<
                "h size should be batch_size * hidden_size";
    }

    for (int seq = 0; seq < sequence; seq++) {
        const InDataType* hidden_in;
        InDataType* hidden_out = dout_data + seq * batch_size * hidden_size;

        if (seq == 0) {
            hidden_in = h;
        } else {
            hidden_in = dout_data + (seq - 1) * batch_size * hidden_size;
        }

        anakin_NV_gemm(_cublas_handle, false, false, batch_size,
                       3 * hidden_size, hidden_size, 1.0, hidden_in,
                       _weights_h2h.data(), 0.0, _temp_WH.mutable_data());

        OpDataType* w_x_r = _temp_WX.mutable_data() + r_offset * hidden_size
                            + seq * batch_size * hidden_size * 3;
        OpDataType* w_x_z = _temp_WX.mutable_data() + z_offset * hidden_size
                            + seq * batch_size * hidden_size * 3;
        OpDataType* w_x_o = _temp_WX.mutable_data() + o_offset * hidden_size
                            + seq * batch_size * hidden_size * 3;

        OpDataType* w_h_r = _temp_WH.mutable_data() + r_offset * hidden_size;
        OpDataType* w_h_z = _temp_WH.mutable_data() + z_offset * hidden_size;
        OpDataType* w_h_o = _temp_WH.mutable_data() + o_offset * hidden_size;

        int frame_per_block = hidden_size <= 1024 ? hidden_size : 1024;

        if (param.gate_activity == Active_sigmoid
                && param.h_activity == Active_tanh) {
            cal_one_kernel_sigmoid_tanh_modi_cudnn_formula
                    << < batch_size, frame_per_block, 0, _ctx->get_compute_stream() >> >
                    (w_x_r, w_x_z, w_x_o, w_h_r, w_h_z, w_h_o
                     , b_r, b_z, b_o, hidden_size, hidden_out, hidden_in);
        } else if (param.gate_activity == Active_sigmoid_fluid
                   && param.h_activity == Active_tanh) {
            cal_one_kernel_paddlesigmoid_tanh_cudnn_formula
                    << < batch_size, frame_per_block, 0, _ctx->get_compute_stream() >> >
                    (w_x_r, w_x_z, w_x_o, w_h_r, w_h_z, w_h_o
                     , b_r, b_z, b_o, hidden_size, hidden_out, hidden_in);
        } else {
            LOG(ERROR) << "not support active  function";
        }

    }

    if (isHW2Seq) {
        seq2hw(outputs, inputs, param, hidden_size, dout_data);
        outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    }
    return SaberSuccess;

}
#endif

template<>
SaberStatus SaberGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
const std::vector<DataTensor_in*>& inputs,
std::vector<DataTensor_out*>& outputs,
GruParam <OpTensor>& param) {
    if (param.formula == GRU_CUDNN) {
//                LOG(ERROR) << "saber cudnn formula not support reverse yet";
//        if (param.is_reverse) {
//                    LOG(ERROR) << "saber cudnn formula not support reverse yet";
//
//        }
//        return gru_cudnn(inputs, outputs, param);
        CHECK(false)<<"not support gru_cudnn now!";
    }

    //    LOG(INFO)<<"gru_paddle";
    DataTensor_in* x = inputs[0];
    std::vector<int> offset=x->get_seq_offset();
    const InDataType* x_data = x->data();
    const InDataType* h;
    DataTensor_out* dout = outputs[0];
    OutDataType* dout_data = dout->mutable_data();

    //TODO:check shape first
    const OpTensor* b = param.bias();

    int batch_size = offset.size() - 1; //x->get_seq_offset().size()-1;
    int seq_sum = x->num();
    int hidden_size = b->valid_size() / 3;
    bool isHW2Seq=offset.size()>2;
    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;

    _temp_map_dev.try_expand_size(seq_sum);
    isHW2Seq = _seq_util.get_sorted_map(offset, _ctx->get_compute_stream());
    auto emit_offset_vec = _seq_util.get_emit_offset_vec();
    auto emit_length = emit_offset_vec.size() - 1;
    if (isHW2Seq) {
        Shape seq_shape(1, 1, seq_sum, _word_size);
        _temp_tensor_in.try_expand_size(seq_shape);
        Shape seq_out_shape(1, 1, seq_sum, _hidden_size);
        _temp_tensor_out.try_expand_size(seq_out_shape);
        _seq_util.seq_2_sorted_seq(x_data,_temp_tensor_in.mutable_data(),_word_size,_ctx->get_compute_stream());
        x_data=_temp_tensor_in.data();
        dout_data = _temp_tensor_out.mutable_data();
    }

    Shape shape_WX(seq_sum, batch_size, 3, hidden_size);
    _temp_WX.try_expand_size(shape_WX);

    Shape shape_WH(1, batch_size, 2, hidden_size);
    _temp_WH.try_expand_size(shape_WH);

    Shape shape_WHR(1, batch_size, 1, hidden_size);
    _temp_WHR.try_expand_size(shape_WHR);

    _gemm_wx(seq_sum, 3 * hidden_size, _word_size,1.0, x_data,0.0, _weights_i2h.data(),_temp_WX.mutable_data(),_ctx->get_compute_stream());

    const OpDataType* b_r = b->data() + r_offset * hidden_size;
    const OpDataType* b_z = b->data() + z_offset * hidden_size;
    const OpDataType* b_o = b->data() + o_offset * hidden_size;

    if (inputs.size() == 1) {
        if(_temp_zero.valid_size()<batch_size * hidden_size){
            _temp_zero.try_expand_size(batch_size * hidden_size);
            CUDA_CHECK(cudaMemsetAsync(_temp_zero.mutable_data(), 0, sizeof(OutDataType)*batch_size * hidden_size,
                                       _ctx->get_compute_stream()));
        }

        h = _temp_zero.data();
    } else {
        h = inputs[1]->data();
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

        const OutDataType* hidden_in;
        OutDataType* hidden_out = dout_data + emit_offset_vec[real_word_id] * hidden_size;

        if (word_id == 0) {
            hidden_in = h;
        } else {
            hidden_in = dout_data + emit_offset_vec[last_word_id] * hidden_size;
        }

        _gemm_wh_2(emit_word_length, 2 * hidden_size, hidden_size,1.0, hidden_in,0.0, _weights_h2h.data() + hidden_size * hidden_size,_temp_WH.mutable_data(),_ctx->get_compute_stream());

        OutDataType* w_x_r = _temp_WX.mutable_data() + r_offset * hidden_size
                             + emit_word_id_start * hidden_size * 3;
        OutDataType* w_x_z = _temp_WX.mutable_data() + z_offset * hidden_size
                             + emit_word_id_start * hidden_size * 3;
        OutDataType* w_x_o = _temp_WX.mutable_data() + o_offset * hidden_size
                             + emit_word_id_start * hidden_size * 3;

        OutDataType* w_h_r = _temp_WH.mutable_data() + 0 * hidden_size;
        OutDataType* w_h_z = _temp_WH.mutable_data() + 1 * hidden_size;



        const OpDataType * w_o = _weights_h2h.data();
                CHECK_LE(hidden_size, 1024) << "now not support hidden size > 1024 for paddle formula";
        int frame_per_block = hidden_size <= 1024 ? hidden_size : 1024;
        if(param.gate_activity == Active_sigmoid) {
            RESET_KERNEL_NAME(sigmoid) << < emit_word_length, frame_per_block, 0
                    , _ctx->get_compute_stream() >> > (
                    w_x_r, w_h_r
                            , b_r, hidden_size, hidden_out, hidden_in);
        }else if(param.gate_activity == Active_sigmoid_fluid){
            RESET_KERNEL_NAME(sigmoid_fluid) << < emit_word_length, frame_per_block, 0
                    , _ctx->get_compute_stream() >> > (
                    w_x_r, w_h_r
                            , b_r, hidden_size, hidden_out, hidden_in);
        }else{
            CHECK_EQ(0,1) << "not support gate active  function "<<param.gate_activity;
        }

        _gemm_wh_o(emit_word_length, hidden_size, hidden_size,1.0, hidden_out,0.0,w_o,_temp_WHR.mutable_data(),_ctx->get_compute_stream());

        if(param.gate_activity == Active_sigmoid_fluid&&param.h_activity == Active_tanh_fluid) {
            FINAL_KERNEL_NAME(sigmoid_fluid,tanh_fluid)<< < emit_word_length, frame_per_block, 0
                    , _ctx->get_compute_stream() >> > (
                    w_x_z, w_x_o, w_h_z, b_z, b_o, hidden_size, hidden_out, hidden_in, _temp_WHR.data());
        }else if(param.gate_activity == Active_sigmoid_fluid&&param.h_activity == Active_relu){
            FINAL_KERNEL_NAME(sigmoid_fluid,relu)<< < emit_word_length, frame_per_block, 0
                    , _ctx->get_compute_stream() >> > (
                    w_x_z, w_x_o, w_h_z, b_z, b_o, hidden_size, hidden_out, hidden_in, _temp_WHR.data());
        }else{
            CHECK_EQ(0,1) << "not support active  function "<<param.gate_activity<<","<<param.h_activity;
        }

    }

    if (isHW2Seq) {
        _seq_util.sorted_seq_2_seq(_temp_tensor_out.data(),dout->mutable_data(),_hidden_size,_ctx->get_compute_stream());
    }
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    return SaberSuccess;
}
template class SaberGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
#if 0
template<>
SaberStatus SaberGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        GruParam <OpTensor>& param) {
    if (param.formula == GRU_CUDNN) {
        LOG(ERROR) << "saber cudnn formula not support reverse yet";
        if (param.is_reverse) {
            LOG(ERROR) << "saber cudnn formula not support reverse yet";

        }
        return gru_cudnn(inputs, outputs, param);
    }

    //    LOG(INFO)<<"gru_paddle";
    DataTensor_in* x = inputs[0];
    std::vector<int> offset=x->get_seq_offset();
    const InDataType* x_data = x->data();
    const InDataType* h;
    DataTensor_out* dout = outputs[0];
    OutDataType* dout_data = dout->mutable_data();

    //TODO:check shape first
    const OpTensor* b = param.bias();

    int batch_size = offset.size() - 1; //x->get_seq_offset().size()-1;
    int sequence = x->num();
    int hidden_size = b->valid_size() / 3;
    bool isHW2Seq=offset.size()>2;
    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;

//    CHECK_EQ(w_h2h->height(), hidden_size) << "w_h2h->height()==batch_size";
//    CHECK_EQ(w_h2h->width(), hidden_size * 3) << "w_h2h->width()==hidden_size*3";
//
//    CHECK_EQ(w_i2h->height(), word_size) << "w_i2h->height()==word_size";
//    CHECK_EQ(w_i2h->width(), hidden_size * 3) << "w_i2h->width()==hidden_size*3";

    if (isHW2Seq) {
        x_data = hw2seq(inputs, param, _word_size, hidden_size, sequence);
//        batch_size = inputs[0]->get_seq_offset().size() - 1;

        if (x_data != x->data()) {
            dout_data = _temp_tensor_out.mutable_data();
        }
    }

    Shape shape_WX(sequence, batch_size, 3, hidden_size);
    _temp_WX.try_expand_size(shape_WX);

    Shape shape_WH(1, batch_size, 2, hidden_size);
    _temp_WH.try_expand_size(shape_WH);

//    anakin_NV_gemm(_cublas_handle, false, false, sequence * batch_size, 3 * hidden_size,
//                   _word_size, 1.0, x_data, _weights_i2h.data(), 0.0, _temp_WX.mutable_data());

    _gemm_wx(sequence * batch_size, 3 * hidden_size, _word_size,1.0, x_data,0.0, _weights_i2h.data(),_temp_WX.mutable_data(),_ctx.get_compute_stream());

    const OpDataType* b_r = b->data() + r_offset * hidden_size;
    const OpDataType* b_z = b->data() + z_offset * hidden_size;
    const OpDataType* b_o = b->data() + o_offset * hidden_size;

    if (inputs.size() == 1) {
        CUDA_CHECK(cudaMemsetAsync(dout_data, 0, sizeof(OutDataType)*batch_size * hidden_size,
                                   _ctx->get_compute_stream()));
        h = dout_data;
    } else {
        h = inputs[1]->data();
    }

    for (int seq = 0; seq < sequence; ++seq) {
        int realseq = seq;
        int last_seq = realseq - 1;

        if (param.is_reverse) {
//            DLOG(INFO)<<"reverse gru";
            realseq = sequence - 1 - seq;
            last_seq = realseq + 1;
        }

        const OutDataType* hidden_in;
        OutDataType* hidden_out = dout_data + realseq * batch_size * hidden_size;

        if (seq == 0) {
            hidden_in = h;
        } else {
            hidden_in = dout_data + last_seq * batch_size * hidden_size;
        }

//        anakin_NV_gemm(_cublas_handle, false, false, batch_size,
//                       2 * hidden_size, hidden_size, 1.0, hidden_in,
//                       _weights_h2h.data() + hidden_size * hidden_size, 0.0, _temp_WH.mutable_data());
        _gemm_wh_2(batch_size, 2 * hidden_size, hidden_size,1.0, hidden_in,0.0, _weights_h2h.data() + hidden_size * hidden_size,_temp_WH.mutable_data(),_ctx.get_compute_stream());

        OutDataType* w_x_r = _temp_WX.mutable_data() + r_offset * hidden_size
                             + realseq * batch_size * hidden_size * 3;
        OutDataType* w_x_z = _temp_WX.mutable_data() + z_offset * hidden_size
                             + realseq * batch_size * hidden_size * 3;
        OutDataType* w_x_o = _temp_WX.mutable_data() + o_offset * hidden_size
                             + realseq * batch_size * hidden_size * 3;

        OutDataType* w_h_r = _temp_WH.mutable_data() + 0 * hidden_size;
        OutDataType* w_h_z = _temp_WH.mutable_data() + 1 * hidden_size;
        const OpDataType * w_o = _weights_h2h.data();

        CHECK_LE(hidden_size, 1024) << "now not support hidden size > 1024 for paddle formula";

        int frame_per_block = hidden_size <= 1024 ? hidden_size : 1024;

        //        DLOG(INFO) << "act = " << param._gate_activity << "," << param._h_activity;

        if (param.gate_activity == Active_sigmoid
                && param.h_activity == Active_tanh) {
            cal_one_kernel_sigmoid_tanh_paddle_formula
            <<< batch_size, frame_per_block, sizeof(OutDataType)*hidden_size
            , _ctx->get_compute_stream()>>>(
                w_x_r, w_x_z, w_x_o, w_h_r, w_h_z, w_o
                , b_r, b_z, b_o, hidden_size, hidden_out, hidden_in);

        } else if (param.gate_activity == Active_sigmoid_fluid
                   && param.h_activity == Active_tanh_fluid) {
            cal_one_kernel_sigmoidfluid_tanhfluid_paddle_formula
                    <<< batch_size, frame_per_block, sizeof(OutDataType)*hidden_size
                    , _ctx.get_compute_stream()>>>(
                    w_x_r, w_x_z, w_x_o, w_h_r, w_h_z, w_o
                            , b_r, b_z, b_o, hidden_size, hidden_out, hidden_in);

        }  else if (param.gate_activity == Active_sigmoid_fluid
                    && param.h_activity == Active_relu) {
            cal_one_kernel_paddlesigmoid_relu_paddle_formula
                    << < batch_size, frame_per_block, sizeof(OutDataType)*hidden_size
                    , _ctx->get_compute_stream() >> >
                    (w_x_r, w_x_z, w_x_o, w_h_r, w_h_z, w_o
                     , b_r, b_z, b_o, hidden_size, hidden_out, hidden_in);

        } else {
            LOG(ERROR) << "not support active  function";
        }
    }

    if (isHW2Seq) {
        seq2hw(outputs, inputs, param, hidden_size, dout_data);
    }
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    return SaberSuccess;
}
#endif

}
}

