#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "mkl_cblas.h"
#include "mkl_vml_functions.h"

#include "saber/core/context.h"
#include "saber/funcs/lstm.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/core/tensor_op.h"
#include "debug.h"

#include "test_saber_func.h"
#include "test_util.h"

using namespace anakin::saber;
using namespace std;

template <typename T>
bool compare_tensor(T& data, T& ref_data, float eps = 1e-4) {
    typedef float data_t;

    if (data.size() != ref_data.size()) {
                LOG(ERROR)<<"data.size() != ref_data.size()";
        return false;
    }

    data_t absdiff = 0.f;
    data_t absref = 0.f;
    for (int i = 0; i < data.size(); i++) {
        absdiff = std::fabs(((const data_t*)data.data())[i] - ((const data_t*)ref_data.data())[i]);
        absref = std::fabs(((const float*)(ref_data.data()))[i]);
        float e = absdiff > eps ? absdiff / absref : absdiff;
        if (e <= eps) {
            return true;
        } else {
            LOG(ERROR)<<"i = "<<i;
            LOG(ERROR) << "out = " << ((data_t*)data.data())[i];
            LOG(ERROR) << "out_ref = " << ((data_t*)ref_data.data())[i];
            return false;
        }
    }
            LOG(ERROR)<<"data.size() = "<<data.size();
    return false;
}

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
typename ACTIVATION<Dtype>::Act Activate(ActiveType type){
    static  typename ACTIVATION<Dtype>::Act vec[9]={&InValidAct<Dtype>, &Sigmoid<Dtype>, &Relu<Dtype>, &Tanh<Dtype>,
                                                    &InValidAct<Dtype>,& InValidAct<Dtype>, &Identity<Dtype>, &Sigmoid_fluid<Dtype>,
                                                    &Tanh_fluid<Dtype>};
    return vec[type];
}

//template <typename Dtype>
//class Activate{
//public:
//    static  typename ACTIVATION<Dtype>::Act vec[9]={&InValidAct<Dtype>, &Sigmoid<Dtype>, &Relu<Dtype>, &Tanh<Dtype>,
//                                            &InValidAct<Dtype>,& InValidAct<Dtype>, &Identity<Dtype>, &Sigmoid_fluid<Dtype>,
//                                            &Tanh_fluid<Dtype>};
//    static typename ACTIVATION<Dtype>::Act get(ActiveType type){
//        return vec[type];
//    }
//    Activate(){
//
//    }
//};

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

template <typename Dtype>
void compute_ref_lstm_one_word(Dtype* wx_i,Dtype* wx_f,Dtype* wx_c,Dtype* wx_o,Dtype* h_old,Dtype* h_new,Dtype* cell_old,Dtype* cell_new,
                               Dtype* bias_i,Dtype* bias_f,Dtype* bias_c,Dtype* bias_o,Dtype* w_c_i,
                               Dtype* w_c_f,Dtype* w_c_o,int hidden_size,
                               ActiveType gate_activity,ActiveType cell_activity,ActiveType candidate_activity){
    for(int i=0;i<hidden_size;i++){
        Dtype gate_i=Activate<Dtype >(gate_activity)(wx_i[i]+w_c_i[i]*cell_old[i]))
    }
}

template <typename Tensor4f>
void compute_ref_lstm_fwd_me(std::vector<Tensor4f*> &src, std::vector<Tensor4f*> &dst, LstmParam<X86> &param){
    typedef float Dtype;
    SaberStatus status = SaberSuccess;

    Tensor4f *input_tensor = src[0];
    Tensor4f *output_tensor = dst[0];
    const Dtype *x = (const Dtype*)input_tensor->data();
    int word_size=input_tensor->channel();
    int hidden_size=output_tensor->channel();
    int seq_sum=input_tensor->num();

    const Dtype *weights = (const Dtype *)param.weight()->data();
    const Dtype *weights_x=weights;
    const Dtype *weights_h=weights+4*word_size*hidden_size;
    const Dtype *bias = (const Dtype *)param.bias()->data();
    const Dtype *weights_peephole=bias+4*hidden_size;
    const Dtype *init_hidden = (const Dtype *)param.init_hidden()->data();

    Dtype *h = (Dtype*)dst[0]->mutable_data();
    Dtype *c = new Dtype[seq_sum*hidden_size];
    Dtype *wx= new Dtype[seq_sum*4*hidden_size];
    Dtype *wh= new Dtype[4*hidden_size];

    std::vector<int> seq_offset = input_tensor->get_seq_offset()[input_tensor->get_seq_offset().size()-1];


    for(int seq_id=0;seq_id<seq_offset.size()-1;seq_id++){
        int seq_start=seq_offset[seq_id];
        int seq_end=seq_offset[seq_id+1];
        if(param.is_reverse){

        }
        for(int word_id=seq_offset[seq_id];word_id<seq_offset[seq_id+1];word_id++){
            Activate<Dtype>(Active_sigmoid)(1.f);
        }
    }



}



typedef Tensor<X86> TensorHf4;
void py_lstm(int word_size = 222,
             int hidden_size = 333){
    Context<X86> ctx_dev(0, 1, 1);
    std::vector<int> offsets = {0, 3,12,19,20};
    ImplEnum test_mode=SABER_IMPL;
//    ImplEnum test_mode=VENDER_IMPL;
    bool is_reverse = false;
    bool with_peephole=false;
    Shape shape_weight({1, 1, 1,hidden_size*hidden_size*4+hidden_size*word_size*4},Layout_NCHW);
    Shape shape_bias;
    if(with_peephole){
        shape_bias=Shape({1,1,1,hidden_size*7},Layout_NCHW);
    }else{
        shape_bias=Shape({1,1,1,hidden_size*4},Layout_NCHW);
    }
    Shape shape_x({offsets[offsets.size() - 1], word_size, 1, 1},Layout_NCHW);
    Shape shape_h({offsets[offsets.size() - 1], hidden_size, 1, 1},Layout_NCHW);
    TensorHf4 host_x(shape_x);
    TensorHf4 host_weight(shape_weight);
    TensorHf4 host_bias(shape_bias);
    TensorHf4 host_hidden_out(shape_h);
    readTensorData(host_weight, "host_w");
    readTensorData(host_x, "host_x");
    readTensorData(host_bias, "host_b");

    host_x.set_seq_offset({offsets});
    LstmParam<X86> param(&host_weight, &host_bias,nullptr,Active_unknow,Active_sigmoid,Active_tanh,Active_tanh,
                               with_peephole,false,is_reverse);
    Lstm<X86, AK_FLOAT> lstm_op;

    std::vector<TensorHf4*> inputs;
    std::vector<TensorHf4*> outputs;
    inputs.push_back(&host_x);
    outputs.push_back(&host_hidden_out);

    SABER_CHECK(lstm_op.init(inputs, outputs, param, SPECIFY, test_mode, ctx_dev));
    SABER_CHECK(lstm_op.compute_output_shape(inputs, outputs, param));
    outputs[0]->re_alloc(outputs[0]->valid_shape(),outputs[0]->get_dtype());
    SABER_CHECK(lstm_op(inputs, outputs, param, ctx_dev));

    TensorHf4 compare_g(shape_h);
    readTensorData(compare_g, "host_correct");
    write_tensorfile(host_hidden_out, "host_g.txt");
    write_tensorfile(compare_g, "host_correct.txt");
    double maxdiff = 0;
    double maxratio = 0;
    tensor_cmp_host((const float*)host_hidden_out.data(), (const float*)compare_g.data(), host_hidden_out.valid_size(), maxratio, maxdiff);
    if (abs(maxratio) <= 0.001) {
                LOG(INFO) << "passed  " << maxratio<<","<<maxdiff<<",?="<<abs(maxratio);
    } else {
                LOG(INFO) << "failed : ratio " << maxratio<<","<<maxdiff;
    }

}

TEST(TestSaberFunc, test_func_lstm) {
    Env<X86>::env_init();

    py_lstm();
}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    printf("%f",Activate<float >(Active_sigmoid)(1.f));
    return 0;
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}