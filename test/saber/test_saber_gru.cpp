#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "saber/core/context.h"
#include "saber/funcs/gru.h"
#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/x86_utils.h"
#endif
#include "saber/core/tensor_op.h"
#include "saber/funcs/debug.h"
#include "saber/funcs/saber_util.h"

#include "test_saber_func.h"

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
    Dtype tmp = static_cast<Dtype>(-2.0) * a;
    return (static_cast<Dtype>(2.0) / (static_cast<Dtype>(1.0) + exp(tmp))) - static_cast<Dtype>(1.0);
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

template <typename Tensor4f, typename TargetType>
void compute_ref_gru_fwd_me(std::vector<Tensor4f*>& inputs, std::vector<Tensor4f*>& outputs,
                            GruParam<TargetType>& param) {
    typedef float OpDataType;
    //    CHECK_NE(param.formula, GRU_CUDNN) << "X86 gru not support cudnn formula now";
    int hidden_size = param.bias()->valid_size() / 3;
    int weights_bias_size = hidden_size * 3;
    int weights_h2h_size = hidden_size * hidden_size * 3;
    int weights_i2h_size = param.weight()->valid_size() - weights_h2h_size;
    int word_size = weights_i2h_size / hidden_size / 3;
    Tensor4f temp_tensor;
    const OpDataType* weight_h = ((const OpDataType*)param.weight()->data()) + weights_i2h_size;

    if (param.formula == GRU_CUDNN) {

        utils::try_expand_tensor(temp_tensor, weights_h2h_size);
        Tensor4f temp_tensor_origin;
        utils::try_expand_tensor(temp_tensor_origin, weights_h2h_size);

        float* temp_tensor_ptr = static_cast<float*>(temp_tensor_origin.mutable_data());
        memcpy(temp_tensor_ptr, static_cast<const OpDataType*>(param.weight()->data()) + weights_i2h_size,
               sizeof(OpDataType) * hidden_size * hidden_size);

        float* rz_temp_tensor_ptr = temp_tensor_ptr + hidden_size * hidden_size;
        const float* rz_weights_tensor_ptr = static_cast<const OpDataType*>(param.weight()->data()) +
                                             weights_i2h_size + hidden_size * hidden_size;

        for (int row = 0; row < hidden_size; row++) {
            for (int block = 0; block < 2; block++) {
                int block_offset = block * hidden_size;

                for (int cow = 0; cow < hidden_size; cow++) {
                    rz_temp_tensor_ptr[block * hidden_size * hidden_size + row * hidden_size + cow] =
                        rz_weights_tensor_ptr[row * (2 * hidden_size) + cow + block_offset];
                }
            }
        }

        float* orz_temp_tensor_ptr = temp_tensor_ptr;
        float* orz_weights_tensor_ptr = static_cast<float*>(temp_tensor.mutable_data());

        for (int row = 0; row < hidden_size; row++) {
            for (int block = 0; block < 3; block++) {
                int block_offset = block * hidden_size;

                for (int cow = 0; cow < hidden_size; cow++) {
                    orz_weights_tensor_ptr[row * (3 * hidden_size) + cow + block_offset] =
                        orz_temp_tensor_ptr[block * hidden_size * hidden_size + row * hidden_size + cow];
                }
            }
        }

        weight_h = static_cast<const OpDataType*>(temp_tensor.data());
    }

    const OpDataType* weight_w = (const OpDataType*)param.weight()->data();
    const OpDataType* bias = (const OpDataType*)param.bias()->data();

    OpDataType(* gat_act)(const OpDataType) = Activate<OpDataType>(param.gate_activity);
    OpDataType(* h_act)(const OpDataType) = Activate<OpDataType>(param.h_activity);

    std::vector<std::vector<int> > offset_vec_vec = inputs[0]->get_seq_offset();
    std::vector<int> offset_vec = offset_vec_vec[offset_vec_vec.size() - 1];

    int batch_size = offset_vec.size() - 1;
    int seqsum = inputs[0]->num();

    const OpDataType* h_init = nullptr;

    Shape zero_hidden_shape({1, 1, batch_size, hidden_size}, Layout_NCHW);
    Tensor4f zero_hidden(zero_hidden_shape);
    memset(zero_hidden.mutable_data(), 0, batch_size * hidden_size * sizeof(float));

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

    int wh_channel = 2;

    if (param.formula == GRU_CUDNN) {
        wh_channel = 3;
    }

    Shape temp_wx_shape({1, 1, 1, seqsum * 3 * hidden_size});
    Shape temp_wh_shape({1, 1, 1, wh_channel * hidden_size});
    Shape temp_whr_shape({1, 1, 1, hidden_size});
    Tensor4f temp_wx_t(temp_wx_shape);
    Tensor4f temp_wh_t(temp_wh_shape);
    Tensor4f temp_whr_t(temp_whr_shape);

    OpDataType* temp_wx = (OpDataType*)temp_wx_t.mutable_data();
    OpDataType* temp_wh = (OpDataType*)temp_wh_t.mutable_data();
    OpDataType* temp_whr = (OpDataType*)temp_whr_t.mutable_data();


    //wx
    gemm_naive(seqsum, 3 * hidden_size, word_size, 1.f, x, weight_w, 0.f, temp_wx);

    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;
    const OpDataType* b_r = bias + r_offset * hidden_size;
    const OpDataType* b_z = bias + z_offset * hidden_size;
    const OpDataType* b_o = bias + o_offset * hidden_size;

    if (param.formula == GRU_ORIGIN) {
        for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
            int batch_offset = offset_vec[batch_id];
            int batch_length = offset_vec[batch_id + 1] - batch_offset;

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

                gemm_naive(1, 2 * hidden_size, hidden_size, 1.0, hin,
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

                gemm_naive(1, hidden_size, hidden_size, 1.0, hout, w_o, 0.f, temp_whr);

                //#pragma simd
                for (int frame_id = 0; frame_id < hidden_size; ++frame_id) {
                    z = gat_act(w_x_z[frame_id] + w_h_z[frame_id] + b_z[frame_id]);
                    _h = w_x_o[frame_id] + temp_whr[frame_id] + b_o[frame_id];
                    _h = h_act(_h);
                    hout[frame_id] = (1 - z) * hin[frame_id] + z * _h;
                }
            }

        }
    } else if (param.formula == GRU_CUDNN) {
        Shape h2h_shape({1, 1, hidden_size, 3 * hidden_size});
        Tensor4f temp_tensor(h2h_shape);
        Tensor4f temp_tensor_origin(h2h_shape);
        float* temp_tensor_ptr = static_cast<float*>(temp_tensor_origin.mutable_data());
        memcpy(temp_tensor_ptr, static_cast<const OpDataType*>(param.weight()->data()) + weights_i2h_size,
               sizeof(OpDataType) * hidden_size * hidden_size);

        float* rz_temp_tensor_ptr = temp_tensor_ptr + hidden_size * hidden_size;
        const float* rz_weights_tensor_ptr = static_cast<const OpDataType*>(param.weight()->data()) +
                                             weights_i2h_size + hidden_size * hidden_size;

        for (int row = 0; row < hidden_size; row++) {
            for (int block = 0; block < 2; block++) {
                int block_offset = block * hidden_size;

                for (int cow = 0; cow < hidden_size; cow++) {
                    rz_temp_tensor_ptr[block * hidden_size * hidden_size + row * hidden_size + cow] =
                        rz_weights_tensor_ptr[row * (2 * hidden_size) + cow + block_offset];
                }
            }
        }

        float* orz_temp_tensor_ptr = temp_tensor_ptr;
        float* orz_weights_tensor_ptr = static_cast<float*>(temp_tensor.mutable_data());

        for (int row = 0; row < hidden_size; row++) {
            for (int block = 0; block < 3; block++) {
                int block_offset = block * hidden_size;

                for (int cow = 0; cow < hidden_size; cow++) {
                    orz_weights_tensor_ptr[row * (3 * hidden_size) + cow + block_offset] =
                        orz_temp_tensor_ptr[block * hidden_size * hidden_size + row * hidden_size + cow];
                }
            }
        }

        for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
            int batch_offset = offset_vec[batch_id];
            int batch_length = offset_vec[batch_id + 1] - batch_offset;

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


                gemm_naive(1, 3 * hidden_size, hidden_size, 1.0, hin,
                           orz_weights_tensor_ptr,
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

                OpDataType* w_h_r = temp_wh + r_offset * hidden_size;
                OpDataType* w_h_z = temp_wh + z_offset * hidden_size;
                OpDataType* w_h_o = temp_wh + o_offset * hidden_size;

                //#pragma simd
                for (int frame_id = 0; frame_id < hidden_size; ++frame_id) {
                    r = Sigmoid(w_x_r[frame_id] + w_h_r[frame_id] + b_r[frame_id]); //h_out=gate_r
                    z = Sigmoid(w_x_z[frame_id] + w_h_z[frame_id] + b_z[frame_id]);
                    _h = Tanh(w_x_o[frame_id] + r * w_h_o[frame_id] + b_o[frame_id]);
                    hout[frame_id] = (1 - z) * _h  + z * hin[frame_id];
                }
            }
        }
    } else {
        LOG(FATAL) << "not support formula id = " << param.formula ;
    };

}
//#define COMPARE_WITH_OUT
template <typename HOST, typename DEVICE>
void gru_ut(int word_size = 222,
            int hidden_size = 333,
            std::vector<int> offsets = {0, 3, 13, 22, 30, 50},
            bool is_reverse = false,
            ActiveType gate_activity = Active_sigmoid,
            ActiveType h_activity_in = Active_tanh,
            int perf_iter = 0, ImplEnum test_mode = SABER_IMPL, GruFormula formula = GRU_ORIGIN) {
    typedef Tensor<HOST> TensorHf4;
    typedef Tensor<DEVICE> TensorDf4;
    Context<DEVICE> ctx_dev(0, 1, 1);

    Shape shape_weight({1, 1, 1, hidden_size* word_size * 3 + hidden_size* hidden_size * 3}, Layout_NCHW);
    Shape shape_bias({1, 1, 1, hidden_size * 3}, Layout_NCHW);

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
#ifdef COMPARE_WITH_OUT
    readTensorData(host_weight, "host_w");
    readTensorData(host_x, "host_x");
    readTensorData(host_bias, "host_b");
#else
    fill_tensor_rand(host_weight, -1.f, 1.f);
    fill_tensor_rand(host_x, -1.f, 1.f);
    fill_tensor_rand(host_bias, -1.f, 1.f);

    //    print_tensor(host_weight);
#endif
    dev_weight.copy_from(host_weight);
    dev_x.copy_from(host_x);
    dev_bias.copy_from(host_bias);

    host_x.set_seq_offset({offsets});
    dev_x.set_seq_offset({offsets});
    GruParam<DEVICE> param(&dev_weight, &dev_bias, formula, gate_activity, h_activity_in,
                           is_reverse, nullptr, 1.f, 1, 1);

    Gru<DEVICE, AK_FLOAT> gru_op;

    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    inputs.push_back(&dev_x);
    outputs.push_back(&dev_hidden_out);

    SABER_CHECK(gru_op.init(inputs, outputs, param, SPECIFY, test_mode, ctx_dev));
    SABER_CHECK(gru_op.compute_output_shape(inputs, outputs, param));
    outputs[0]->re_alloc(outputs[0]->valid_shape(), outputs[0]->get_dtype());
#ifndef COMPARE_WITH_OUT
    SABER_CHECK(gru_op(inputs, outputs, param, ctx_dev));
    outputs[0]->record_event(ctx_dev.get_compute_stream());
    outputs[0]->sync();

    if (perf_iter > 0) {
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
                  << "," << offsets.size() - 1 << "," << word_size << "," << hidden_size << "]";
    }

#ifdef NVIDIA_GPU
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    host_hidden_out.copy_from(dev_hidden_out);
#endif

    TensorHf4 compare_g(shape_h);

    std::vector<TensorHf4*> inputs_ref;
    std::vector<TensorHf4*> outputs_ref;
    inputs_ref.push_back(&host_x);
    outputs_ref.push_back(&compare_g);
    GruParam<HOST> param_ref(&host_weight, &host_bias, formula, gate_activity, h_activity_in,
                             is_reverse, nullptr, 1.f, 1, 1);
    compute_ref_gru_fwd_me(inputs_ref, outputs_ref, param_ref);
#ifdef COMPARE_WITH_OUT
    host_hidden_out.copy_from(compare_g);
    write_tensorfile(host_hidden_out, "host_g.txt");
    readTensorData(compare_g, "host_correct");
    write_tensorfile(compare_g, "host_correct.txt");
#else

#endif

    double maxdiff = 0;
    double maxratio = 0;

    tensor_cmp_host((const float*)host_hidden_out.data(), (const float*)compare_g.data(),
                    host_hidden_out.valid_size(), maxratio, maxdiff);

    if (abs(maxratio) <= 0.01 || abs(maxdiff) < 0.01) {
        LOG(INFO) << "passed  " << maxratio << "," << maxdiff << ",?=" << abs(
                      maxratio) << "::" << word_size << "," << hidden_size << "," << is_reverse << "," << offsets.size()
                  << "::" << gate_activity << "," << h_activity_in;
    } else {
        write_tensorfile(host_hidden_out, "host_g.txt");
        write_tensorfile(compare_g, "host_correct.txt");
        CHECK(false) << "failed : ratio " << maxratio << "," << maxdiff << "::" << word_size << "," <<
                     hidden_size << "," << is_reverse << "," << offsets.size() << "::" << gate_activity << "," <<
                     h_activity_in;
    }
}

#ifdef USE_X86_PLACE


TEST(TestSaberFunc, test_func_gru_x86) {
    Env<X86>::env_init();
    srand(12345678);

    for (int word_size : {
                15, 222
            }) {
        for (int hidden_size : {
                    15, 333
                }) {
            for (bool reverse : {
                        false
                    }) {
                for (ActiveType gate_act : {
                            Active_sigmoid
                        }) {
                    for (ActiveType cell_act : {
                                Active_tanh
                            }) {
                        for (ImplEnum impl : {
                                    SABER_IMPL
                                }) {
                            for (GruFormula formula : {
                                        GRU_ORIGIN, GRU_CUDNN
                                    }) {
                                gru_ut<X86, X86>(word_size, hidden_size, {0, 5}, reverse, gate_act, cell_act, 0, impl, formula);
                                gru_ut<X86, X86>(word_size, hidden_size, {0, 3, 7, 12, 13}, reverse, gate_act, cell_act, 0, impl,
                                                 formula);
                            }
                        }
                    }
                }
            }
        }
    }

}

#endif

#ifdef NVIDIA_GPU

TEST(TestSaberFunc, test_func_gru_nv) {
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    srand(12345678);

    for (int word_size : {
                15, 222
            }) {
        for (int hidden_size : {
                    15, 333
                }) {
            for (bool reverse : {
                        true, false
                    }) {
                for (ActiveType gate_act : {
                            Active_sigmoid
                        }) {
                    for (ActiveType cell_act : {
                                Active_tanh
                            }) {
                        for (ImplEnum impl : {
                                    SABER_IMPL
                                }) {
                            for (GruFormula formula : {
                                        GRU_ORIGIN, GRU_CUDNN
                                    }) {
                                gru_ut<NVHX86, NV>(word_size, hidden_size, {0, 3, 7, 12, 13}, reverse, gate_act, cell_act, 0, impl,
                                                   formula);
                                gru_ut<NVHX86, NV>(word_size, hidden_size, {0, 5}, reverse, gate_act, cell_act, 0, impl, formula);
                            }
                        }
                    }
                }
            }
        }
    }

    for (int word_size : {
                15, 222
            }) {
        for (int hidden_size : {
                    15, 333
                }) {
            for (bool reverse : {
                        false
                    }) {
                for (ActiveType gate_act : {
                            Active_sigmoid
                        }) {
                    for (ActiveType cell_act : {
                                Active_tanh
                            }) {
                        for (ImplEnum impl : {
                                    VENDER_IMPL
                                }) {
                            for (GruFormula formula : {
                                        GRU_CUDNN
                                    }) {
                                gru_ut<NVHX86, NV>(word_size, hidden_size, {0, 3, 7, 12, 13}, reverse, gate_act, cell_act, 0, impl,
                                                   formula);
                                gru_ut<NVHX86, NV>(word_size, hidden_size, {0, 5}, reverse, gate_act, cell_act, 0, impl, formula);
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif
int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
