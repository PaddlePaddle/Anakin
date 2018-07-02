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
#include "saber/saber_types.h"
#include "x86_test_common.h"
#include "test_saber_func_x86.h"
#include "debug.h"

using namespace anakin::saber;
using namespace std;

typedef struct _test_lstm_params {
    int mb;
    int input_size;
    int layer_size;
    ActiveType input_activation;
    ActiveType gate_activation;
    ActiveType candidate_activation;
    ActiveType cell_activation;
    bool with_peephole;
    bool with_init_hidden;
    bool skip_input;
} test_lstm_params;

typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4f;

inline void sigmoid(int len, float *x, float *y) {
    for (int i = 0; i < len; i++) {
        y[i] = 1. / (1. + exp(-x[i]));
    }
}

inline void relu(int len, float *x, float *y) {
    for (int i = 0; i < len; i++) {
        y[i] = x[i] < 0 ? 0 : x[i];
    }
}

inline void tanh(int len, float *x, float *y) {
    for (int i = 0; i < len; i++) {
        float e_x = exp(x[i]);
        float e_minus_x = 1 / e_x;
        y[i] = (e_x - e_minus_x) / (e_x + e_minus_x);
    }
}

inline void stanh(int len, float *x, float *y) {
    for (int i = 0; i < len; i++) {
        float e_x = exp(2 * x[i] / 3);
        float e_minus_x = 1 / e_x;
        y[i] = 1.7159 * (e_x - e_minus_x) / (e_x + e_minus_x);
    }
}

void compute_ref_lstm_fwd(std::vector<Tensor4f*> &src, std::vector<Tensor4f*> &dst, LstmParam<Tensor4f> &param) {
    SaberStatus status = SaberSuccess;

    const Tensor4f *weights = param.weight();
    const Tensor4f *bias = param.bias();
    const Tensor4f *init_hidden = param.init_hidden();

    Tensor4f *input = src[0];
    float *h = dst[0]->mutable_data();
    float *c = nullptr;

    // get Wx = [Wix, Wfx, Wcx, Woc] while they are all input_size * layer_size matrices
    const float *x = input->data();
    int N = input->num();
    int input_size = input->channel();
    int layer_size = dst[0]->channel();

    if (dst.size() >= 2) {
        c = dst[1]->mutable_data();
    } else {
        c = (float*)zmalloc(N * layer_size * sizeof(float), 4096);
    }

    float *xx = nullptr;
    if (param.skip_input) {
        // the input is x * Wx
        xx = const_cast<float *>(x);
    } else {
        // get xx = x * Wx
        const float *Wx = weights->data();
        xx = (float*)zmalloc(N * 4 * layer_size * sizeof(float), 4096);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, 4 * layer_size, input_size, 1, x, input_size, Wx, 4 * layer_size, 0, xx, 4 * layer_size);
        if (param.input_activity != Active_unknow) {
            if (param.input_activity == Active_stanh) {
                stanh(N * 4 * layer_size, xx, xx);
            } else if (param.input_activity == Active_tanh) {
                tanh(N * 4 * layer_size, xx, xx);
            } else {
                LOG(ERROR) << "unsupported gate activation now";
            }
        }
    }

    float* ihcot = (float*)zmalloc(4 * layer_size * sizeof(float), 4096);
    float* act = (float*)zmalloc(layer_size * sizeof(float), 4096);
    float* p = (float*)zmalloc(layer_size * sizeof(float), 4096);

    std::vector<int> seq_offset = input->get_seq_offset();
    int seq_num = seq_offset.size() - 1;

    const float *Wh = nullptr;
    if (param.skip_input) {
        Wh = weights->data();
    } else {
        Wh = weights->data() + 4 * input_size * layer_size;
    }
    const float *b = bias->data();
    const float *peephole = nullptr;
    if (param.with_peephole) {
        peephole = bias->data() + 4 * layer_size;
    }

    float* init_h = (float*)zmalloc(layer_size * sizeof(float), 4096);
    float* init_c = (float*)zmalloc(layer_size * sizeof(float), 4096);
    memset(init_h, 0, layer_size * sizeof(float));
    memset(init_c, 0, layer_size * sizeof(float));

    for (int i = 0; i < seq_num; i++) {
        if (param.init_hidden()) {
            const float *init_state = param.init_hidden()->data();
            memcpy(init_h, init_state + i * layer_size, layer_size * sizeof(float));
            memcpy(init_c, init_state + (i + seq_num)* layer_size, layer_size * sizeof(float));
        }
        // do LSTM per sequence
        int seq_len = seq_offset[i + 1] - seq_offset[i];
        for (int j = 0; j < seq_len; j++) {
            float *ht = h + (seq_offset[i] + j) * layer_size;
            float *ct = c + (seq_offset[i] + j) * layer_size;
            float *xxt = xx + (seq_offset[i] + j) * 4 * layer_size;
            float *ht_1 = nullptr;
            float *ct_1 = nullptr;
            cblas_saxpby (4 * layer_size, 1, xxt, 1, 0, ihcot, 1);

            if (j == 0) {
                ht_1 = init_h;
                ct_1 = init_c;
            } else {
                ht_1 = h + (seq_offset[i] + (j - 1)) * layer_size;
                ct_1 = c + (seq_offset[i] + (j - 1)) * layer_size;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 4 * layer_size, layer_size, 1, ht_1, layer_size, Wh,
                        4 * layer_size, 1, ihcot, 4 * layer_size);

            if (peephole) {
                // peephole for it
                vsMul(layer_size, ct_1, peephole, p);
                cblas_saxpby(layer_size, 1, p, 1, 1, ihcot, 1);
                // peephole for ft
                vsMul(layer_size, ct_1, peephole + layer_size, p);
                cblas_saxpby(layer_size, 1, p, 1, 1, ihcot + layer_size, 1);
            }
            // add bias
            cblas_saxpby(4 * layer_size, 1, b, 1, 1, ihcot, 1);

            // gate activity for it and ft, candidate activity for cct
            if (param.gate_activity == Active_sigmoid) {
                sigmoid(layer_size, ihcot, ihcot);
                sigmoid(layer_size, ihcot + layer_size, ihcot + layer_size);
            } else {
                LOG(ERROR) << "unsupported gate activation now";
            }

            if (param.candidate_activity == Active_relu) {
                relu(layer_size, ihcot + 2 * layer_size, ihcot + 2 * layer_size);
            } else {
                LOG(ERROR) << "unsupported candidate activation now";
            }

            // calc ct
            vsMul(layer_size, ihcot, ihcot + 2 * layer_size, p);
            cblas_saxpby(layer_size, 1, p, 1, 0, ct, 1);
            vsMul(layer_size, ihcot + layer_size, ct_1, p);
            cblas_saxpby(layer_size, 1, p, 1, 1, ct, 1);

            // peephole for ot
            if (peephole) {
                vsMul(layer_size, ct, peephole + 2 * layer_size, p);
                cblas_saxpby(layer_size, 1, p, 1, 1, ihcot + 3 * layer_size, 1);
            }
            if (param.gate_activity == Active_sigmoid) {
                sigmoid(layer_size, ihcot + 3 * layer_size, ihcot + 3 * layer_size);
            }

            // calc ht
            if (param.cell_activity == Active_sigmoid) {
                sigmoid(layer_size, ct, act);
            }
            vsMul(layer_size, ihcot + 3 * layer_size, act, ht);
        }
    }

    if (!param.skip_input && xx) {
        zfree(xx);
        xx = nullptr;
    }
    if (ihcot) {
        zfree(ihcot);
        ihcot = nullptr;
    }
    if (act) {
        zfree(act);
        act = nullptr;
    }
    if (p) {
        zfree(p);
        p = nullptr;
    }
    if (init_h) {
        zfree(init_h);
        init_h = nullptr;
    }
    if (init_c) {
        zfree(init_c);
        init_c = nullptr;
    }
    if (dst.size() < 2 && c != nullptr) {
        zfree(c);
        c = nullptr;
    }

    return;
}

bool lstm_test(test_lstm_params &param) {
    std::vector<Tensor4f*> inputs;

    std::vector<int> seq_offsets;
    int total_seq_len = 0;
    int offset = 0;
    for (int i = 0; i < param.mb; i++) {
        int seq_len = rand()%50 + 50;
        total_seq_len += seq_len;
        seq_offsets.push_back(offset);
        offset += seq_len;
    }
    seq_offsets.push_back(offset);

    Shape inputShape(total_seq_len, param.input_size, 1, 1);
    if (param.skip_input) {
        inputShape[1] = 4 * param.layer_size;
    }
    Tensor4f *i = new Tensor4f(inputShape);
    i->set_seq_offset(seq_offsets);
    inputs.push_back(i);
    fill_tensor_host_rand<Tensor4f>(*(inputs[0]));

    // weight's layout:
    // [ Wih Wfh Wch Woh ]
    Shape weightShape(param.layer_size, 4 * param.layer_size, 1, 1);
    if (!param.skip_input) {
        // weight's layout:
        // [ Wix Wfx Wcx Wox ]
        // [ Wih Wfh Wch Woh ]
        // It's a (input_size + layer_size) * (4 * layer_size) matrix
        weightShape[0] = param.input_size + param.layer_size;
    }
    Tensor4f saberWeight(weightShape);
    fill_tensor_host_rand(saberWeight);

    // bias's layout:
    // while with peephole: [bi, bf, bc, bo, wic, wfc, woc]
    // while not: [bi, bf, bc, bo]
    // It's a 1 * (7 * layer_size or 4 * layer_size) vector
    int bias_num = 4;
    if (param.with_peephole) {
        bias_num = 7;
    }
    Shape biasShape(1, bias_num * param.layer_size, 1, 1);
    Tensor4f saberBias(biasShape);
    fill_tensor_host_rand(saberBias);

    Shape hiddenShape(param.mb * 2, param.layer_size, 1, 1);
    Tensor4f saberHidden(hiddenShape);
    fill_tensor_host_rand(saberHidden);

    LstmParam<Tensor4f> lstm_param(&saberWeight, &saberBias, param.with_init_hidden ? &saberHidden : nullptr,
                                   param.input_activation, param.gate_activation, param.cell_activation,
                                   param.candidate_activation, param.with_peephole, param.skip_input);

    Shape outputShape(total_seq_len, param.layer_size, 1, 1);
    std::vector<Tensor4f*> saber_outputs;
    Tensor4f saberOutputh(outputShape);
    Tensor4f saberOutputc(outputShape);
    saber_outputs.push_back(&saberOutputh);
//    saber_outputs.push_back(&saberOutputc);

    std::vector<Tensor4f*> ref_outputs;
    Tensor4f refOutputh(outputShape);
    Tensor4f refOutputc(outputShape);
    ref_outputs.push_back(&refOutputh);
    ref_outputs.push_back(&refOutputc);

    // compute reference result
    compute_ref_lstm_fwd(inputs, ref_outputs, lstm_param);

    // compute saber result
    Lstm<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> saberLstm;
    Context<X86> ctx_host;
    saberLstm.init(inputs, saber_outputs, lstm_param, SPECIFY, SABER_IMPL, ctx_host);
    saberLstm(inputs, saber_outputs, lstm_param, ctx_host);

    bool flag = compare_tensor(*saber_outputs[0], *ref_outputs[0], 1e-4);
//    flag &= compare_tensor(*saber_outputs[1], *ref_outputs[1], 1e-4);
    return flag;
}

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
void py_lstm(int word_size = 222,
             int hidden_size = 333){
    Context<X86> ctx_dev(0, 1, 1);
    std::vector<int> offsets = {0, 3,12,19,20};
    ImplEnum test_mode=SABER_IMPL;
//    ImplEnum test_mode=VENDER_IMPL;
    bool is_reverse = false;
    bool with_peephole=false;
    Shape shape_weight(1, 1, 1,hidden_size*hidden_size*4+hidden_size*word_size*4);
    Shape shape_bias;
    if(with_peephole){
        shape_bias=Shape(1,1,1,hidden_size*7);
    }else{
        shape_bias=Shape(1,1,1,hidden_size*4);
    }
    Shape shape_x(offsets[offsets.size() - 1], word_size, 1, 1);
    Shape shape_h(offsets[offsets.size() - 1], hidden_size, 1, 1);
    TensorHf4 host_x(shape_x);
    TensorHf4 host_weight(shape_weight);
    TensorHf4 host_bias(shape_bias);
    TensorHf4 host_hidden_out(shape_h);
    readTensorData(host_weight, "host_w");
    readTensorData(host_x, "host_x");
    readTensorData(host_bias, "host_b");

    host_x.set_seq_offset(offsets);
    LstmParam<TensorHf4> param(&host_weight, &host_bias,nullptr,Active_unknow,Active_sigmoid,Active_tanh,Active_tanh,
                               with_peephole,false,is_reverse);
    Lstm<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> lstm_op;

    std::vector<TensorHf4*> inputs;
    std::vector<TensorHf4*> outputs;
    inputs.push_back(&host_x);
    outputs.push_back(&host_hidden_out);

    SABER_CHECK(lstm_op.init(inputs, outputs, param, SPECIFY, test_mode, ctx_dev));
    SABER_CHECK(lstm_op.compute_output_shape(inputs, outputs, param));
    outputs[0]->re_alloc(outputs[0]->valid_shape());
    SABER_CHECK(lstm_op(inputs, outputs, param, ctx_dev));

    TensorHf4 compare_g(shape_h);
    readTensorData(compare_g, "host_correct");
    write_tensorfile(host_hidden_out, "host_g.txt");
    write_tensorfile(compare_g, "host_correct.txt");
    double maxdiff = 0;
    double maxratio = 0;
    tensor_cmp_host(host_hidden_out.data(), compare_g.data(), host_hidden_out.valid_size(), maxratio, maxdiff);
    if (abs(maxratio) <= 0.001) {
        LOG(INFO) << "passed  " << maxratio<<","<<maxdiff<<",?="<<abs(maxratio);
    } else {
        LOG(INFO) << "failed : ratio  " << maxratio<<","<<maxdiff;
    }

}

TEST(TestSaberFuncX86, test_tensor_lstm) {
    Env<X86>::env_init();

    test_lstm_params test_param[] = {
        // batch_size, input_size, layer_size, input_activation, gate_activation, candidate_activation, cell_activation, with_peephole, with_init_hidden, skip_input
        test_lstm_params{6, 55, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, false, false, false},
        test_lstm_params{6, 55, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, true, false, false},
//        test_lstm_params{6, 55, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, false, true, false},
//        test_lstm_params{6, 55, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, true, true, false},
//        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, false, false, false},
//        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, true, false, false},
//        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, true, true, false},
//        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, false, true, false},
//        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, false, false, false},
//        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, true, false, false},
//        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, false, true, false},
//        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, true, true, false},
//        test_lstm_params{6, 55, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, false, false, true},
//        test_lstm_params{6, 55, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, true, false, true},
//        test_lstm_params{6, 55, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, false, true, true},
//        test_lstm_params{6, 55, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, true, true, true},
        /*test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, false, false, true},
        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, true, false, true},
        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, true, true, true},
        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, false, true, true},
        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, false, false, true},
        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, true, false, true},
        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, false, true, true},
        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, true, true, true},*/
    };

    for (size_t i = 0; i < ARRAY_SIZE(test_param); i++) {
        LOG(INFO) << "case " << i;
        bool ret = lstm_test(test_param[i]);
        if (ret) {
            LOG(INFO) << "Test Passed";
        }
        else {
            LOG(ERROR) << "Test Failed";
        }
    }

//    py_lstm();
}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
