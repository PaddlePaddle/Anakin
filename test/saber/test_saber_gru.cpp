#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "mkl_cblas.h"

#include "saber/core/context.h"
#include "saber/funcs/gru.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/core/tensor_op.h"
#include "debug.h"

#include "test_saber_func.h"
#include "test_util.h"

using namespace anakin::saber;
using namespace std;

template <typename Dtype>
static Dtype InValidAct(Dtype a) {
    CHECK(false)<<"InValidAct";
}

template <typename Dtype>
static Dtype Sigmoid(const Dtype a) {
    return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-a));
}

template <typename Dtype>
static Dtype Tanh(const Dtype a) {
    Dtype tmp = -2.0 * a;
    return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

template <typename Dtype>
static Dtype Relu(const Dtype a) {
    return a > static_cast<Dtype>(0.0) ? a : static_cast<Dtype>(0.0);
}

template <typename Dtype>
static Dtype Identity(const Dtype a) {
    return a;
}

#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0
#define EXP_MAX_INPUT 40.0

template <typename Dtype>
static Dtype Sigmoid_fluid(const Dtype a) {
    const Dtype min = SIGMOID_THRESHOLD_MIN;
    const Dtype max = SIGMOID_THRESHOLD_MAX;
    Dtype tmp = (a < min) ? min : ((a > max) ? max : a);
    return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-tmp));
}

template <typename Dtype>
static Dtype Tanh_fluid(const Dtype a) {
    Dtype tmp = -2.0 * a;
    tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
    return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

template <typename Dtype>
struct ACTIVATION{
    typedef Dtype(*Act)(const Dtype);
};

template <typename Dtype>
inline typename ACTIVATION<Dtype>::Act Activate(ActiveType type){
    static  typename ACTIVATION<Dtype>::Act vec[9]={&InValidAct<Dtype>, &Sigmoid<Dtype>, &Relu<Dtype>, &Tanh<Dtype>,
                                                    &InValidAct<Dtype>,& InValidAct<Dtype>, &Identity<Dtype>, &Sigmoid_fluid<Dtype>,
                                                    &Tanh_fluid<Dtype>};
    return vec[type];
}

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
};


template <typename Tensor4f,typename TargetType>
void compute_ref_gru_fwd_me(std::vector<Tensor4f*> &inputs, std::vector<Tensor4f*> &outputs, GruParam<TargetType> &param){
    typedef float OpDataType;
    CHECK_NE(param.formula, GRU_CUDNN) << "X86 gru not support cudnn formula now";
    int hidden_size = param.bias()->valid_size() / 3;
    int weights_bias_size = hidden_size * 3;
    int weights_h2h_size = hidden_size * hidden_size * 3;
    int weights_i2h_size = param.weight()->valid_size() - weights_h2h_size;
    int word_size = weights_i2h_size / hidden_size / 3;

    const OpDataType* weight_h = ((const OpDataType*)param.weight()->data())+weights_i2h_size;
    const OpDataType* weight_w = (const OpDataType*)param.weight()->data();
    const OpDataType* bias = (const OpDataType*)param.bias()->data();

    OpDataType(* gat_act)(const OpDataType) = Activate<OpDataType>(param.gate_activity);
    OpDataType(* h_act)(const OpDataType) = Activate<OpDataType>(param.h_activity);

    std::vector<std::vector<int> > offset_vec_vec = inputs[0]->get_seq_offset();
    std::vector<int> offset_vec=offset_vec_vec[offset_vec_vec.size()-1];

    int batch_size = offset_vec.size() - 1;
    int seqsum = inputs[0]->num();

    const OpDataType* h_init = nullptr;

    Shape zero_hidden_shape({1,1,batch_size,hidden_size},Layout_NCHW);
    Tensor4f zero_hidden(zero_hidden_shape);
    if (inputs.size() > 1) {
        h_init = (const OpDataType*)inputs[1]->data();
    } else if (param.init_hidden() != nullptr) {
        h_init = (const OpDataType*)param.init_hidden()->data();
    } else {
        h_init = (const OpDataType*)zero_hidden.data();
    }

    const OpDataType* x = (const OpDataType*)inputs[0]->data();
    OpDataType* out = (OpDataType*)outputs[0]->mutable_data();

    bool is_reverse = param.is_reverse;

    //        Shape wx_shaep(1,seqsum,3,_alignedhidden_size_iter_num,_aligned_size);
    Shape temp_wx_shape({1,1,1,seqsum * 3 * hidden_size});
    Shape temp_wh_shape({1,1,1,2 * hidden_size});
    Shape temp_whr_shape({1,1,1, hidden_size});
    Tensor4f temp_wx_t(temp_wx_shape);
    Tensor4f temp_wh_t(temp_wh_shape);
    Tensor4f temp_whr_t(temp_whr_shape);

    OpDataType* temp_wx = (OpDataType*)temp_wx_t.mutable_data();
    OpDataType* temp_wh = (OpDataType*)temp_wh_t.mutable_data();
    OpDataType* temp_whr =(OpDataType*)temp_whr_t.mutable_data();

    //    LOG(INFO) << "gemm b" << inputs[0]->valid_shape().count() << "," <<
    //              _weights_i2h.valid_shape().count() << "," << _temp_wx.valid_shape().count();
    //wx
    gemm(false, false, seqsum, 3 * hidden_size, word_size, 1.f, x, weight_w, 0.f, temp_wx);

    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;
    const OpDataType* b_r = bias + r_offset * hidden_size;
    const OpDataType* b_z = bias + z_offset * hidden_size;
    const OpDataType* b_o = bias + o_offset * hidden_size;


    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        int batch_offset = offset_vec[batch_id];
        int batch_length = offset_vec[batch_id+1]-batch_offset;

        for (int seq_id_in_batch = 0; seq_id_in_batch < batch_length; ++seq_id_in_batch) {
            int seqid = batch_offset + seq_id_in_batch;
            int last_seq_id = seqid - 1;

            if (is_reverse) {
                seqid = batch_offset + batch_length - 1 - seq_id_in_batch;
                last_seq_id = seqid + 1;
            }

            const OpDataType* hin;
            OpDataType* hout = seqid * hidden_size + out;

            if (seq_id_in_batch == 0) {
                hin = h_init + batch_id * hidden_size;
            } else {
                hin = out + last_seq_id * hidden_size;
            }

            gemm(false, false, 1, 2 * hidden_size, hidden_size, 1.0, hin,
                 weight_h + hidden_size * hidden_size,
                 0.f, temp_wh);

            volatile OpDataType r;
            volatile OpDataType z;
            volatile OpDataType _h;
            OpDataType* w_x_r = temp_wx + r_offset * hidden_size
                                + seqid * hidden_size * 3;
            OpDataType* w_x_z = temp_wx + z_offset * hidden_size
                                + seqid * hidden_size * 3;
            OpDataType* w_x_o = temp_wx + o_offset * hidden_size
                                + seqid * hidden_size * 3;

            OpDataType* w_h_r = temp_wh + 0 * hidden_size;
            OpDataType* w_h_z = temp_wh + 1 * hidden_size;
            const OpDataType* w_o = weight_h;

            //#pragma simd
            for (int frame_id = 0; frame_id < hidden_size; ++frame_id) {
                r = w_x_r[frame_id] + w_h_r[frame_id] + b_r[frame_id]; //h_out=gate_r
                r = gat_act(r);
                hout[frame_id] = r * hin[frame_id];
            }

            gemm(false, false, 1, hidden_size, hidden_size, 1.0, hout, w_o, 0.f, temp_whr);

            //#pragma simd
            for (int frame_id = 0; frame_id < hidden_size; ++frame_id) {
                z = gat_act(w_x_z[frame_id] + w_h_z[frame_id] + b_z[frame_id]);
                _h = w_x_o[frame_id] + temp_whr[frame_id] + b_o[frame_id];
                _h = h_act(_h);
                hout[frame_id] = (1 - z) * hin[frame_id] + z * _h;
            }
        }

    }

}
//#define COMPARE_WITH_OUT
template <typename HOST,typename DEVICE>
void gru_ut(int word_size = 222,
             int hidden_size = 333,
             std::vector<int> offsets = {0, 3,13,22,30,50},
             bool is_reverse = false,
             ActiveType gate_activity=Active_sigmoid,
             ActiveType h_activity_in=Active_tanh,
             int perf_iter=0,ImplEnum test_mode=SABER_IMPL){
    typedef Tensor<HOST> TensorHf4;
    typedef Tensor<DEVICE> TensorDf4;
    Context<DEVICE> ctx_dev(0, 1, 1);

    Shape shape_weight({1, 1, 1,hidden_size*word_size*3+hidden_size*hidden_size*3},Layout_NCHW);
    Shape shape_bias({1,1,1,hidden_size*3},Layout_NCHW);

    Shape shape_x({offsets[offsets.size() - 1], word_size, 1, 1},Layout_NCHW);
    Shape shape_h({offsets[offsets.size() - 1], hidden_size, 1, 1},Layout_NCHW);
    TensorHf4 host_x(shape_x);
    TensorHf4 host_weight(shape_weight);
    TensorHf4 host_bias(shape_bias);
    TensorHf4 host_hidden_out(shape_h);
    TensorDf4 dev_x(shape_x);
    TensorDf4 dev_weight(shape_weight);
    TensorDf4 dev_bias(shape_bias);
    TensorDf4 dev_hidden_out(shape_h);
#ifdef COMPARE_WITH_OUT
    readTensorData(host_weight, "host_w");
    readTensorData(host_x, "host_x");
    readTensorData(host_bias, "host_b");
#else
    fill_tensor_rand(host_weight);
    fill_tensor_rand(host_x);
    fill_tensor_rand(host_bias);
#endif
    dev_weight.copy_from(host_weight);
    dev_x.copy_from(host_x);
    dev_bias.copy_from(host_bias);

    host_x.set_seq_offset({offsets});
    dev_x.set_seq_offset({offsets});
    GruParam<DEVICE> param(&dev_weight, &dev_bias,GRU_ORIGIN,gate_activity,h_activity_in,
                           is_reverse, nullptr,1.f,1,1);

    Gru<DEVICE, AK_FLOAT> gru_op;

    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    inputs.push_back(&dev_x);
    outputs.push_back(&dev_hidden_out);

    SABER_CHECK(gru_op.init(inputs, outputs, param, SPECIFY, test_mode, ctx_dev));
    SABER_CHECK(gru_op.compute_output_shape(inputs, outputs, param));
    outputs[0]->re_alloc(outputs[0]->valid_shape(),outputs[0]->get_dtype());
#ifndef COMPARE_WITH_OUT
    SABER_CHECK(gru_op(inputs, outputs, param, ctx_dev));
    outputs[0]->record_event(ctx_dev.get_compute_stream());
    outputs[0]->sync();

    if(perf_iter>0) {
        SaberTimer<DEVICE> t1;
        t1.start(ctx_dev);
        for (int i = 0; i < perf_iter; ++i) {
            SABER_CHECK(gru_op(inputs, outputs, param, ctx_dev));
            outputs[0]->record_event(ctx_dev.get_compute_stream());
            outputs[0]->sync();
        }
        t1.end(ctx_dev);
        LOG(INFO) << "!!saber care: iter = " << perf_iter << " , total time: " << t1.get_average_ms() <<
                  "avg time : " << t1.get_average_ms() / perf_iter << " args [" << offsets[offsets.size() - 1]
                  << "," << offsets.size() - 1 << ","<< word_size << "," << hidden_size << "]";
    }
    host_hidden_out.copy_from(dev_hidden_out);
#endif

    TensorHf4 compare_g(shape_h);

//    readTensorData(compare_g, "host_correct");
//    write_tensorfile(host_hidden_out, "host_g.txt");
//    write_tensorfile(compare_g, "host_correct.txt");

    std::vector<TensorHf4*> inputs_ref;
    std::vector<TensorHf4*> outputs_ref;
    inputs_ref.push_back(&host_x);
    outputs_ref.push_back(&compare_g);
    GruParam<HOST> param_ref(&host_weight, &host_bias,GRU_ORIGIN,gate_activity,h_activity_in,
                             is_reverse, nullptr,1.f,1,1);
    compute_ref_gru_fwd_me(inputs_ref,outputs_ref,param_ref);
#ifdef COMPARE_WITH_OUT
    host_hidden_out.copy_from(compare_g);
    write_tensorfile(host_hidden_out, "host_g.txt");
    readTensorData(compare_g, "host_correct");
#endif
    double maxdiff = 0;
    double maxratio = 0;
    tensor_cmp_host((const float*)host_hidden_out.data(), (const float*)compare_g.data(), host_hidden_out.valid_size(), maxratio, maxdiff);
    if (abs(maxratio) <= 0.001||abs(maxdiff)<0.001) {
        LOG(INFO) << "passed  " << maxratio<<","<<maxdiff<<",?="<<abs(maxratio);
    } else {
        CHECK(false) << "failed : ratio " << maxratio<<","<<maxdiff;
    }
}

#ifdef USE_X86_PLACE


TEST(TestSaberFunc, test_func_gru_x86) {
    Env<X86>::env_init();
    gru_ut<X86,X86>(222,333,{0,2,5,12,30}, true,Active_sigmoid,Active_tanh,100);
    gru_ut<X86,X86>(222,333,{0,2,5,12,30}, false,Active_sigmoid,Active_tanh,100);
    gru_ut<X86,X86>(222,333,{0,2,5,12,30}, true,Active_sigmoid,Active_relu,100);
    gru_ut<X86,X86>(222,333,{0,2,5,12,30}, false,Active_sigmoid,Active_relu,100);
    gru_ut<X86,X86>(222,333,{0,30},        true,Active_sigmoid,Active_tanh,100);
    gru_ut<X86,X86>(222,333,{0,30},        false,Active_sigmoid,Active_tanh,100);

    gru_ut<X86,X86>(222,333,{0,2,5,12,30}, true,Active_sigmoid,Active_tanh,100,VENDER_IMPL);
    gru_ut<X86,X86>(222,333,{0,2,5,12,30}, false,Active_sigmoid,Active_tanh,100,VENDER_IMPL);
    gru_ut<X86,X86>(222,333,{0,2,5,12,30}, true,Active_sigmoid,Active_relu,100,VENDER_IMPL);
    gru_ut<X86,X86>(222,333,{0,2,5,12,30}, false,Active_sigmoid,Active_relu,100,VENDER_IMPL);
    gru_ut<X86,X86>(222,333,{0,30},        true,Active_sigmoid,Active_tanh,100,VENDER_IMPL);
    gru_ut<X86,X86>(222,333,{0,30},        false,Active_sigmoid,Active_tanh,100,VENDER_IMPL);

}

#endif

#ifdef NVIDIA_GPU

TEST(TestSaberFunc, test_func_gru_nv) {
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    gru_ut<NVHX86,NV>(222,333,{0,2,5,12,30}, true,Active_sigmoid,Active_tanh,100);
    gru_ut<NVHX86,NV>(222,333,{0,2,5,12,30}, false,Active_sigmoid,Active_tanh,100);
    gru_ut<NVHX86,NV>(222,333,{0,2,5,12,30}, true,Active_sigmoid,Active_relu,100);
    gru_ut<NVHX86,NV>(222,333,{0,2,5,12,30}, false,Active_sigmoid,Active_relu,100);
    gru_ut<NVHX86,NV>(222,333,{0,30},        true,Active_sigmoid,Active_tanh,100);
    gru_ut<NVHX86,NV>(222,333,{0,30},        false,Active_sigmoid,Active_tanh,100);

}

#endif
int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}