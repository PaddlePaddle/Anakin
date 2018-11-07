#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "saber/core/context.h"
#include "saber/funcs/lstm.h"
#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/x86_utils.h"
#endif
#include "saber/core/tensor_op.h"
#include "debug.h"

#include "test_saber_func.h"
#include <cmath>

using namespace anakin::saber;
using namespace std;

template <typename Dtype>
static Dtype InValidAct(Dtype a) {
    CHECK(false) << "InValidAct";
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

template <typename Dtype>
struct ACTIVATION {
    typedef Dtype(*Act)(const Dtype);
};

template <typename Dtype>
inline typename ACTIVATION<Dtype>::Act Activate(ActiveType type) {
    static  typename ACTIVATION<Dtype>::Act vec[7] = {&InValidAct<Dtype>, &Sigmoid<Dtype>, &Relu<Dtype>, &Tanh<Dtype>,
                                                      &InValidAct<Dtype>, & InValidAct<Dtype>, &Identity<Dtype>
                                                     };
    return vec[type];
}


template<typename Dtype>
static void gemm_naive(int m, int n, int k, const float alpha, const Dtype* a, const Dtype* b ,
                       const float beta, Dtype* c) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Dtype acc = 0;

            for (int inner = 0; inner < k; inner++) {
                acc += alpha * a[i * k + inner] * b[inner * n + j];
            }

            c[i * n + j] = acc + beta * c[i * n + j];
        }
    }
}

template <typename Dtype>
void compute_ref_lstm_one_word(const Dtype* wx_i, const Dtype* wx_f, const Dtype* wx_c,
                               const Dtype* wx_o, Dtype* h_new, const Dtype* cell_old, Dtype* cell_new,
                               const Dtype* bias_i, const Dtype* bias_f, const Dtype* bias_c, const Dtype* bias_o,
                               const Dtype* w_c_i,
                               const Dtype* w_c_f, const Dtype* w_c_o, int hidden_size,
                               ActiveType gate_activity, ActiveType cell_activity, ActiveType candidate_activity,
                               bool with_peephole) {

    typename ACTIVATION<Dtype>::Act gate_func = Activate<Dtype >(gate_activity);
    typename ACTIVATION<Dtype>::Act cell_func = Activate<Dtype >(cell_activity);
    typename ACTIVATION<Dtype>::Act candi_func = Activate<Dtype >(candidate_activity);

    if (with_peephole) {
        for (int i = 0; i < hidden_size; i++) {
            Dtype gate_i = gate_func(wx_i[i] + w_c_i[i] * cell_old[i] + bias_i[i]);
            Dtype gate_f = gate_func(wx_f[i] + w_c_f[i] * cell_old[i] + bias_f[i]);
            Dtype gate_c_t = cell_func(wx_c[i] + bias_c[i]);
            Dtype gate_c = gate_f * cell_old[i] + gate_i * gate_c_t;
            Dtype gate_o = gate_func(wx_o[i] + w_c_o[i] * gate_c + bias_o[i]);
            h_new[i] = gate_o * candi_func(gate_c);
            cell_new[i] = gate_c;
            //        DLOG(INFO)<<"gate_i = "<<gate_i<<","<<wx_i[i]<<","<<w_c_i[i]<<","<<cell_old[i]<<","<<bias_i[i]<<",befor "<<wx_o[i]+w_c_o[i]*gate_c+bias_o[i]<<",h = "<<h_new[i]<<",c = "<<cell_new[i];
        }
    } else {
        for (int i = 0; i < hidden_size; i++) {
            Dtype gate_i = gate_func(wx_i[i]  + bias_i[i]);
            Dtype gate_f = gate_func(wx_f[i]  + bias_f[i]);
            Dtype gate_c_t = cell_func(wx_c[i] + bias_c[i]);
            Dtype gate_c = gate_f * cell_old[i] + gate_i * gate_c_t;
            Dtype gate_o = gate_func(wx_o[i]  + bias_o[i]);
            h_new[i] = gate_o * candi_func(gate_c);
            cell_new[i] = gate_c;
            //        DLOG(INFO)<<"gate_i = "<<gate_i<<","<<wx_i[i]<<","<<w_c_i[i]<<","<<cell_old[i]<<","<<bias_i[i]<<",befor "<<wx_o[i]+w_c_o[i]*gate_c+bias_o[i]<<",h = "<<h_new[i]<<",c = "<<cell_new[i];
        }
    }
}

template <typename Tensor4f, typename TargetType>
void compute_ref_lstm_fwd_me(std::vector<Tensor4f*>& src, std::vector<Tensor4f*>& dst,
                             LstmParam<TargetType>& param) {
    typedef float Dtype;
    SaberStatus status = SaberSuccess;

    Tensor4f* input_tensor = src[0];
    Tensor4f* output_tensor = dst[0];
    const Dtype* x = (const Dtype*)input_tensor->data();
    int word_size = input_tensor->channel();
    int hidden_size = output_tensor->channel();
    int seq_sum = input_tensor->num();

    const Dtype* weights = (const Dtype*)param.weight()->data();
    const Dtype* weights_x = weights;
    const Dtype* weights_h = weights + 4 * word_size * hidden_size;
    const Dtype* bias = (const Dtype*)param.bias()->data();
    const Dtype* weights_peephole = bias + 4 * hidden_size;
    const Dtype* init_hidden = nullptr;
    vector<Dtype> vec_init_hidden(hidden_size, 0);

    if (param.init_hidden() != nullptr) {
        init_hidden = (const Dtype*)param.init_hidden()->data();
    } else {
        init_hidden = vec_init_hidden.data();
    }

    const Dtype* b_i = bias + 0 * hidden_size;
    const Dtype* b_f = bias + 1 * hidden_size;
    const Dtype* b_c = bias + 2 * hidden_size;
    const Dtype* b_o = bias + 3 * hidden_size;

    const Dtype* wc_i = weights_peephole + 0 * hidden_size;
    const Dtype* wc_f = weights_peephole + 1 * hidden_size;
    const Dtype* wc_o = weights_peephole + 2 * hidden_size;

    Dtype* h = (Dtype*)dst[0]->mutable_data();
    vector<Dtype> vec_c(seq_sum * hidden_size, 0);
    vector<Dtype> vec_wx(seq_sum * 4 * hidden_size, 0);
    Dtype* c = vec_c.data();
    Dtype* wx = vec_wx.data();
    std::vector<int> seq_offset =
        input_tensor->get_seq_offset()[input_tensor->get_seq_offset().size() - 1];

    gemm_naive(seq_sum, 4 * hidden_size, word_size, 1, x, weights, 0, wx);

    for (int seq_id = 0; seq_id < seq_offset.size() - 1; seq_id++) {
        int seq_start = seq_offset[seq_id];
        int seq_end = seq_offset[seq_id + 1];

        if (param.is_reverse) {
            for (int word_id = seq_end - 1; word_id >= seq_start; word_id--) {

                Dtype* cell_old = nullptr;

                if (word_id == seq_end - 1) {
                    cell_old = c + word_id * hidden_size;
                    //                    LOG(INFO) << "word = " << word_id << ",seq sum = " << seq_sum<<",cell[]="<<word_id * hidden_size<<","<<seq_sum*hidden_size<<","<<c[4];
                    gemm_naive(1, 4 * hidden_size, hidden_size, 1, init_hidden, weights_h, 1,
                               wx + word_id * 4 * hidden_size);
                } else {
                    cell_old = c + (word_id + 1) * hidden_size;
                    gemm_naive(1, 4 * hidden_size, hidden_size, 1, h + (word_id + 1) * hidden_size, weights_h,
                               1, wx + word_id * 4 * hidden_size);
                }

                const Dtype* wx_i = wx + word_id * 4 * hidden_size + 0 * hidden_size;
                const Dtype* wx_f = wx + word_id * 4 * hidden_size + 1 * hidden_size;
                const Dtype* wx_c = wx + word_id * 4 * hidden_size + 2 * hidden_size;
                const Dtype* wx_o = wx + word_id * 4 * hidden_size + 3 * hidden_size;

                Dtype* h_new = h + word_id * hidden_size;
                Dtype* cell_new = c + word_id * hidden_size;

                compute_ref_lstm_one_word(wx_i, wx_f, wx_c, wx_o, h_new, cell_old, cell_new, b_i, b_f, b_c, b_o,
                                          wc_i,
                                          wc_f, wc_o,
                                          hidden_size, param.gate_activity, param.cell_activity,
                                          param.candidate_activity, param.with_peephole);
            }

        } else {
            for (int word_id = seq_start; word_id < seq_end; word_id++) {

                Dtype* cell_old = nullptr;

                if (word_id == seq_start) {
                    cell_old = c + word_id * hidden_size;
                    gemm_naive(1, 4 * hidden_size, hidden_size, 1, init_hidden, weights_h, 1,
                               wx + word_id * 4 * hidden_size);
                } else {
                    cell_old = c + (word_id - 1) * hidden_size;
                    gemm_naive(1, 4 * hidden_size, hidden_size, 1, h + (word_id - 1) * hidden_size, weights_h,
                               1, wx + word_id * 4 * hidden_size);
                }

                const Dtype* wx_i = wx + word_id * 4 * hidden_size + 0 * hidden_size;
                const Dtype* wx_f = wx + word_id * 4 * hidden_size + 1 * hidden_size;
                const Dtype* wx_c = wx + word_id * 4 * hidden_size + 2 * hidden_size;
                const Dtype* wx_o = wx + word_id * 4 * hidden_size + 3 * hidden_size;

                Dtype* h_new = h + word_id * hidden_size;
                Dtype* cell_new = c + word_id * hidden_size;

                compute_ref_lstm_one_word(wx_i, wx_f, wx_c, wx_o, h_new, cell_old, cell_new, b_i, b_f, b_c, b_o,
                                          wc_i,
                                          wc_f, wc_o,
                                          hidden_size, param.gate_activity, param.cell_activity,
                                          param.candidate_activity, param.with_peephole);
            }
        }
    }

}
//#define COMPARE_FILE
template <typename HOST, typename DEVICE>
void lstm_ut(int word_size = 222,
             int hidden_size = 333,
             std::vector<int> offsets = {0, 3, 13, 22, 30, 50},
             bool is_reverse = true,
             bool with_peephole = true,
             ActiveType gate_activity = Active_sigmoid,
             ActiveType cell_activity = Active_tanh,
             ActiveType candi_activity = Active_tanh,
             int perf_iter = 0, ImplEnum test_mode = SABER_IMPL) {
    typedef Tensor<HOST> TensorHf4;
    typedef Tensor<DEVICE> TensorDf4;
    Context<DEVICE> ctx_dev(0, 1, 1);

    Shape shape_weight({1, 1, 1, hidden_size* hidden_size * 4 + hidden_size* word_size * 4}, Layout_NCHW);
    Shape shape_bias;

    if (with_peephole) {
        shape_bias = Shape({1, 1, 1, hidden_size * 7}, Layout_NCHW);
    } else {
        shape_bias = Shape({1, 1, 1, hidden_size * 4}, Layout_NCHW);
    }

    Shape shape_x({offsets[offsets.size() - 1], word_size, 1, 1}, Layout_NCHW);
    Shape shape_h({offsets[offsets.size() - 1], hidden_size, 1, 1}, Layout_NCHW);
    TensorHf4 host_x(shape_x);
    TensorHf4 host_weight(shape_weight);
    TensorHf4 host_bias(shape_bias);
    TensorHf4 host_hidden_out(shape_h);
    TensorDf4 dev_x(shape_x);
    TensorDf4 dev_weight(shape_weight);
    TensorDf4 dev_bias(shape_bias);
    TensorDf4 dev_hidden_out(shape_h);
#ifdef COMPARE_FILE
    readTensorData(host_weight, "host_w");
    readTensorData(host_x, "host_x");
    readTensorData(host_bias, "host_b");
#else
    fill_tensor_rand(host_weight, -1, 1);
    fill_tensor_rand(host_x, -1, 1);
    //    fill_tensor_const(host_weight,0.f);
    //    fill_tensor_const(host_x,0.f);
    fill_tensor_rand(host_bias, -1, 1);
#endif
    dev_weight.copy_from(host_weight);
    dev_x.copy_from(host_x);
    dev_bias.copy_from(host_bias);

    host_x.set_seq_offset({offsets});
    dev_x.set_seq_offset({offsets});
    LstmParam<DEVICE> param(&dev_weight, &dev_bias, nullptr, Active_unknow, gate_activity,
                            cell_activity, candi_activity,
                            with_peephole, false, is_reverse);
    Lstm<DEVICE, AK_FLOAT> lstm_op;

    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    inputs.push_back(&dev_x);
    outputs.push_back(&dev_hidden_out);

    SABER_CHECK(lstm_op.init(inputs, outputs, param, SPECIFY, test_mode, ctx_dev));
    SABER_CHECK(lstm_op.compute_output_shape(inputs, outputs, param));
    outputs[0]->re_alloc(outputs[0]->valid_shape(), outputs[0]->get_dtype());
    SABER_CHECK(lstm_op(inputs, outputs, param, ctx_dev));
    outputs[0]->record_event(ctx_dev.get_compute_stream());
    outputs[0]->sync();

    if (perf_iter > 0) {
        SaberTimer<DEVICE> t1;
        t1.start(ctx_dev);

        for (int i = 0; i < perf_iter; ++i) {
            SABER_CHECK(lstm_op(inputs, outputs, param, ctx_dev));
            outputs[0]->record_event(ctx_dev.get_compute_stream());
            outputs[0]->sync();
        }

        t1.end(ctx_dev);
        LOG(INFO) << "!!saber care: iter = " << perf_iter << " , total time: " << t1.get_average_ms() <<
                  "avg time : " << t1.get_average_ms() / perf_iter << " args [" << offsets[offsets.size() - 1]
                  << "," << offsets.size() - 1 << "," << word_size << "," << hidden_size << "]";
    }

    host_hidden_out.copy_from(dev_hidden_out);
    TensorHf4 compare_g(shape_h);
#ifdef COMPARE_FILE
    readTensorData(compare_g, "host_correct");
    write_tensorfile(host_hidden_out, "host_g.txt");
    write_tensorfile(compare_g, "host_correct.txt");
#else
    std::vector<TensorHf4*> inputs_ref;
    std::vector<TensorHf4*> outputs_ref;
    outputs_ref.push_back(&compare_g);
    inputs_ref.push_back(&host_x);
    LstmParam<HOST> param_ref(&host_weight, &host_bias, nullptr, Active_unknow, gate_activity,
                              cell_activity, candi_activity,
                              with_peephole, false, is_reverse);
    compute_ref_lstm_fwd_me(inputs_ref, outputs_ref, param_ref);
#endif
    double maxdiff = 0;
    double maxratio = 0;
    tensor_cmp_host((const float*)host_hidden_out.data(), (const float*)compare_g.data(),
                    host_hidden_out.valid_size(), maxratio, maxdiff);

    if (abs(maxratio) <= 0.005 || abs(maxdiff) < 0.005) {
        LOG(INFO) << "passed  " << maxratio << "," << maxdiff << ",?=" << abs(maxratio);
    } else {
        write_tensorfile(host_hidden_out, "host_g.txt");
        write_tensorfile(compare_g, "host_correct.txt");

        for (int i : offsets) {
            LOG(INFO) << "offset = " << i;
        }

        LOG(INFO) << "param = " << word_size << "," << hidden_size << "," << ",reverse = " << is_reverse <<
                  ",with_peephole = " << with_peephole;
        LOG(INFO) << "gate_activity = " << gate_activity << ",cell_activity = " << cell_activity <<
                  ",candi_activity = " << candi_activity;
        LOG(INFO) << "impl = " << test_mode;
        CHECK(false) << "failed : ratio " << maxratio << "," << maxdiff;
    }

}

#ifdef USE_X86_PLACE

TEST(TestSaberFunc, test_func_lstm_x86) {
    Env<X86>::env_init();
#ifdef COMPARE_FILE
    lstm_ut<X86, X86>(15, 333, {0, 5}, true, true, Active_tanh, Active_tanh, Active_tanh, 0,
                      SABER_IMPL);
#else

    for (int word_size : {
                15, 222
            })

        for (int hidden_size : {
                    15, 333
                })

            for (bool reverse : {
                        true, false
                    })

                for (bool with_peephole : {
                            true, false
                        })

                    for (ActiveType gate_act : {
                                Active_sigmoid, Active_tanh
                            })

                        for (ActiveType cell_act : {
                                    Active_sigmoid, Active_tanh
                                })

                            for (ActiveType candi_act : {
                                        Active_sigmoid, Active_tanh
                                    })

                                for (ImplEnum impl : {
                                            SABER_IMPL
                                        }) {
                                    lstm_ut<X86, X86>(word_size, hidden_size, {0, 3, 7, 12, 13}, reverse, with_peephole, gate_act,
                                                      cell_act, candi_act, 0, impl);
                                    lstm_ut<X86, X86>(word_size, hidden_size, {0, 5}, reverse, with_peephole, gate_act, cell_act,
                                                      candi_act, 0, impl);
                                }
#endif
}
#endif

#ifdef NVIDIA_GPU
TEST(TestSaberFunc, test_func_lstm_nv) {
    Env<NV>::env_init();

    for (int word_size : {
                15, 222
            })

        for (int hidden_size : {
                    15, 333
                })

            for (bool reverse : {
                        true, false
                    })

                for (bool with_peephole : {
                            true, false
                        })

                    for (ActiveType gate_act : {
                                Active_sigmoid, Active_tanh
                            })

                        for (ActiveType cell_act : {
                                    Active_sigmoid, Active_tanh
                                })

                            for (ActiveType candi_act : {
                                        Active_sigmoid, Active_tanh
                                    })

                                for (ImplEnum impl : {
                                            SABER_IMPL
                                        }) {
                                    lstm_ut<NVHX86, NV>(word_size, hidden_size, {0, 3, 7, 12, 13}, reverse, with_peephole, gate_act,
                                                        cell_act, candi_act, 0, impl);
                                    lstm_ut<NVHX86, NV>(word_size, hidden_size, {0, 5}, reverse, with_peephole, gate_act, cell_act,
                                                        candi_act, 0, impl);
                                }

    for (int word_size : {
                15, 222
            })

        for (int hidden_size : {
                    15, 333
                })

            for (ActiveType gate_act : {
                        Active_sigmoid
                    })

                for (ActiveType cell_act : {
                            Active_tanh
                        })

                    for (ActiveType candi_act : {
                                Active_tanh
                            })

                        for (ImplEnum impl : {
                                    VENDER_IMPL
                                }) {
                            lstm_ut<NVHX86, NV>(word_size, hidden_size, {0, 3, 7, 12, 13}, false, false, gate_act, cell_act,
                                                candi_act, 0, impl);
                            lstm_ut<NVHX86, NV>(word_size, hidden_size, {0, 5}, false, false, gate_act, cell_act, candi_act, 0,
                                                impl);
                        }
}
#endif

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
