#include "saber/funcs/fc.h"
#include "saber/saber_types.h"
#include "saber/core/context.h"
#include "saber/core/tensor_op.h"

#include "test_saber_func.h"
#include "test_saber_base.h"
#if defined(USE_X86_PLACE)
#include "saber/funcs/impl/x86/kernel/jit_generator.h"
#endif

using namespace anakin::saber;

template <typename dtype>
int count_diff(const void* input1, const void* input2, int size,
               double max_ratio, bool with_print = false) {
    auto src1 = static_cast<const dtype*>(input1);
    auto src2 = static_cast<const dtype*>(input2);

    if (max_ratio <= 0) {
        max_ratio = 1e-2;
    }

    int count = 0;

    for (int i = 0; i < size; ++i) {
        double ratio = fabs(src1[i] - src2[i]) /
                       fabs(src1[i] + src2[i] + 1e-12);

        if (ratio > max_ratio) {
            if (with_print) {
                LOG(ERROR) << "out = " << (float)src1[i]
                           << "\nout_ref = " << (float)src2[i];
            }

            ++count;
        }
    }

    return count;
}

template <typename src_dtype,
          typename op_dtype,
          typename dst_dtype,
          typename bias_dtype,
          typename TargetType>
void fc_cpu_common(const std::vector<Tensor<TargetType>* >& src,
                   std::vector<Tensor<TargetType>* >& dst,
                   FcParam<TargetType>& param) {
    int output_channel = dst[0]->count_valid(1, dst[0]->dims());
    int batch_size = src[0]->num();

    Shape OutShape({batch_size, output_channel, 1, 1}, Layout_NCHW);
    Tensor<X86> dst_tmp;
    dst_tmp.re_alloc(OutShape, AK_INT32);

    auto dst_tmp_data = static_cast<int32_t*>(dst_tmp.mutable_data());
    auto dst_data = static_cast<dst_dtype*>(dst[0]->mutable_data());
    auto weights_data = static_cast<const op_dtype*>(param.weights->data());
    auto bias_data = param.bias ?
                     static_cast<const bias_dtype*>(param.bias->data()) :
                     nullptr;

    for (int i = 0; i < src.size(); i++) {
        int IC = src[i]->count_valid(1, src[i]->dims());
        auto src_data = static_cast<const src_dtype*>(src[i]->data());

        #pragma omp parallel for collapse(2) schedule(static)

        for (int mb = 0; mb < batch_size; mb++) {
            for (int oc = 0; oc < output_channel; oc++) {
                int oidx = mb * output_channel + oc;

                if (i == 0) {
                    if (src[0]->get_dtype() == AK_UINT8) {
                        dst_tmp_data[oidx] = bias_data ? bias_data[oc] : dst_dtype{0};
                    } else {
                        dst_data[oidx] = bias_data ? bias_data[oc] : dst_dtype{0};
                    }
                }

                for (int ic = 0; ic < IC; ic++) {
                    int iidx = mb * IC + ic;
                    int widx = oc * IC + ic;

                    if (src[0]->get_dtype() == AK_UINT8) {
                        dst_tmp_data[oidx] += src_data[iidx] * weights_data[widx];
                    } else {
                        dst_data[oidx] += src_data[iidx] * weights_data[widx];
                    }
                }
            }
        }

        weights_data += output_channel * IC;
    }

    if (src[0]->get_dtype() == AK_UINT8) {
        for (int mb = 0; mb < batch_size; mb++) {
            for (int oc = 0; oc < output_channel; oc++) {
                int dst_index = mb * output_channel + oc;
                float scale = (src[0]->get_scale()[0] * param.weights->get_scale()[oc]) /
                              dst[0]->get_scale()[0];
                dst_data[dst_index] = scale * dst_tmp_data[dst_index];
            }
        }
    }
}

template <DataType inDtype,
          DataType opDtype,
          DataType outDtype,
          DataType biasDtype>
void test_fc_cpu(int mb,
                 std::vector<int> ic,
                 int oc,
                 bool with_bias = false,
                 std::vector<float>scale = {1.f, 1.f, 1.f},
                 LayoutType layout = Layout_NCHW) {
    Env<X86>::env_init();
    Context<X86> ctx_host;

    std::vector<Tensor<X86> *> inputs, outputs, outputs_ref;
    Tensor<X86> weights, bias;

    int total_ic = 0;

    for (int i = 0; i < ic.size(); i++) {
        total_ic += ic[i];
        Shape InputShape({mb, layout == Layout_NCHW ? ic[i] : 1,
                          1, layout == Layout_NCHW ? 1 : ic[i]}, layout);
        inputs.push_back(new Tensor<X86>);
        inputs[i]->re_alloc(InputShape, inDtype);

        if (inDtype == AK_FLOAT) {
            fill_tensor_rand(*inputs[i], -10.f, 10.f);
        } else {
            fill_tensor_rand(*inputs[i], 0, 255);
            inputs[i]->set_scale({scale[0]});
        }
    }

    Shape WeightShape({oc, layout == Layout_NCHW ? total_ic : 1,
                       1, layout == Layout_NCHW ? 1 : total_ic}, layout);
    Shape BiasShape({layout == Layout_NCHW ? oc : 1, 1,
                     1, layout == Layout_NCHW ? 1 : oc}, layout);
    Shape OutShape({mb, layout == Layout_NCHW ? oc : 1,
                    1, layout == Layout_NCHW ? 1 : oc}, layout);

    outputs.push_back(new Tensor<X86>);
    outputs_ref.push_back(new Tensor<X86>);

    weights.re_alloc(WeightShape, opDtype);
    bias.re_alloc(BiasShape, biasDtype);
    outputs[0]->re_alloc(OutShape, outDtype);
    outputs_ref[0]->re_alloc(OutShape, outDtype);

    fill_tensor_rand(weights, -10, 10);
    fill_tensor_rand(bias, -10, 10);

    std::vector<float> scale_weights;

    for (int i = 0; i < oc; i ++) {
        scale_weights.push_back(scale[1]);
    }

    weights.set_scale(scale_weights);
    outputs[0]->set_scale({scale[2]});
    outputs_ref[0]->set_scale({scale[2]});

    FcParam<X86> param(&weights, with_bias ? &bias : nullptr, oc);
    Fc<X86, opDtype> VenderFc;

    VenderFc.init(inputs, outputs, param, SPECIFY, VENDER_IMPL, ctx_host);
    VenderFc(inputs, outputs, param, ctx_host);

    int flag = 10;

    if (opDtype == AK_FLOAT) {
        fc_cpu_common<float, float, float, float, X86>(inputs, outputs_ref, param);
        flag = count_diff<float>(outputs[0]->data(), outputs_ref[0]->data(),
                                 outputs[0]->valid_size(), 1e-3);
    } else {
        if (outDtype == AK_FLOAT) {
            fc_cpu_common<uint8_t, int8_t, float, int32_t, X86>(inputs, outputs_ref, param);
            flag = count_diff<float>(outputs[0]->data(), outputs_ref[0]->data(),
                                     outputs[0]->valid_size(), 1e-5);
        } else if (outDtype == AK_INT32) {
            fc_cpu_common<uint8_t, int8_t, int32_t, int32_t, X86>(inputs, outputs_ref, param);
            flag = count_diff<int32_t>(outputs[0]->data(), outputs_ref[0]->data(),
                                       outputs[0]->valid_size(), 1e-5);
        } else if (outDtype == AK_INT8) {
            fc_cpu_common<uint8_t, int8_t, int8_t, int32_t, X86>(inputs, outputs_ref, param);
            flag = count_diff<int8_t>(outputs[0]->data(), outputs_ref[0]->data(),
                                      outputs[0]->valid_size(), 1e-5);
        }
    }

    if (flag <= 5) {
        LOG(INFO) << "Test fc x86 passed";
    } else {
        LOG(ERROR) << "Test fc x86 failed";
    }

    return;
}

template <typename dtype, typename TargetType_D, typename TargetType_H>
static void fc_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                        std::vector<Tensor<TargetType_H>* >& output, FcParam<TargetType_D>& param) {

    const dtype* data_in = (const dtype*)input[0]->data();
    const dtype* bias = param.bias ? (const dtype*)param.bias->data() : nullptr;

    Tensor<TargetType_H> weights_h(param.weights->valid_shape());
    weights_h.copy_from(*param.weights);

    const dtype* weights = (const dtype*)weights_h.data();
    dtype* data_out = (dtype*)output[0]->mutable_data();

    //is_trans: flase.
    //output: data_out; inputs: data_in ; weights: weights.
    //data_out = data_in * weights. Get weights' elements continuosly.
    int out_rows = input[0]->num();
    int in_cols = input[0]->valid_size() / out_rows;
    int out_cols = param.weights->valid_size() / in_cols;

    for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
            int index_out = i * out_cols + j;
            data_out[index_out] = bias ? bias[j] : 0;

            for (int k = 0; k < in_cols; k++) {
                //data_out[index_out] += data_in[i * in_cols + k] * weights[k * out_cols + j];
                data_out[index_out] += data_in[i * in_cols + k] * weights[j * in_cols + k];
            }
        }
    }
}

template <typename dtype>
static int count_diff(const dtype* src1, const dtype* src2, int size, double max_ratio) {
    if (max_ratio <= 0) {
        max_ratio = 0.1;
    }

    int count = 0;

    for (int i = 0; i < size; ++i) {
        double ratio = fabs(src1[i] - src2[i]) / fabs(src1[i] + src2[i] + 1e-12);

        if (ratio > max_ratio) {
            ++count;
        }
    }

    return count;
}
template <typename TargetType_D, typename TargetType_H>
static void test_fc_int8(int in_num, int in_channel, int in_height, int in_width, int num_output,
                         bool with_bias) {
    Env<TargetType_D>::env_init();
    Env<TargetType_H>::env_init();
    Shape input_shape({in_num, in_channel, in_height, in_width});
    Shape weights_shape({1, 1, num_output, in_channel* in_height * in_width});
    Shape bias_shape({1, 1, 1, num_output});
    Tensor<TargetType_H> host_input(input_shape);
    Tensor<TargetType_D> dev_input{input_shape};
    Tensor<TargetType_H> host_weights(weights_shape);
    Tensor<TargetType_D> dev_weights{weights_shape};
    Tensor<TargetType_H> host_bias;
    Tensor<TargetType_D> dev_bias;
    Tensor<TargetType_H> host_output;
    Tensor<TargetType_D> dev_output;
    Tensor<TargetType_H> check_output;

    float input_max = 1.f;
    fill_tensor_rand(host_input, -input_max, input_max);
    //    fill_tensor_const(host_input, input_max);
    dev_input.copy_from(host_input);
    dev_input.set_scale({input_max / 127.f});

    fill_tensor_rand(host_weights, -input_max, input_max);
    //    fill_tensor_seq(host_weights);
    //    fill_tensor_const(host_weights, input_max);
    dev_weights.copy_from(host_weights);


    if (with_bias) {
        host_bias.re_alloc(bias_shape);
        dev_bias.re_alloc(bias_shape);
        fill_tensor_const(host_bias, input_max);
        //        fill_tensor_rand(host_bias, -input_max, input_max);
        dev_bias.copy_from(host_bias);
    }

    std::vector<Tensor<TargetType_D>* > input_v;
    std::vector<Tensor<TargetType_D>* > output_v;
    input_v.push_back(&dev_input);
    output_v.push_back(&dev_output);

    Context<TargetType_D> ctx1(0, 1, 1);
    FcParam<TargetType_D> param(&dev_weights, &dev_bias, num_output);
    Fc<TargetType_D, AK_INT8> fc;

    fc.compute_output_shape(input_v, output_v, param);
    dev_output.re_alloc(dev_output.valid_shape());
    dev_output.set_scale({1.f});
    host_output.re_alloc(dev_output.valid_shape());
    check_output.re_alloc(dev_output.valid_shape());

    SABER_CHECK(fc.init(input_v, output_v, param, SPECIFY, VENDER_IMPL, ctx1));
    SABER_CHECK(fc(input_v, output_v, param, ctx1));
    typename Tensor<TargetType_D>::API::stream_t stream = ctx1.get_compute_stream();
    output_v[0]->record_event(stream);
    output_v[0]->sync();

    std::vector<Tensor<TargetType_H>* > input_h;
    std::vector<Tensor<TargetType_H>* > output_h;
    input_h.push_back(&host_input);
    output_h.push_back(&check_output);
    fc_cpu_base<float, TargetType_D, TargetType_H>(input_h, output_h, param);


    host_output.copy_from(dev_output);
    double max_ratio = 0.0;
    double max_diff = 0.0;
    tensor_cmp_host_mlu((const float*)check_output.data(), (const float*)host_output.data(),
                        host_output.valid_size(), max_ratio, max_diff);

    if (max_ratio > 0.1) {
        write_tensorfile(dev_weights, "input_weights");
        write_tensorfile(dev_output, "output_dev");
        write_tensorfile(check_output, "check_host");
        LOG(FATAL) << "ratio " << max_ratio;
    } else {
        //        write_tensorfile(dev_output,"output_dev");
        //        write_tensorfile(check_output,"check_host");
        LOG(ERROR) << "passed " << max_ratio;
    }


};
#ifdef USE_X86_PLACE
void test_int8_perf(int m, int n, int k, int iter = 100) {
    signed char* ptr_a = new signed char[m * k];
    unsigned char* ptr_b = new unsigned char[k * n];
    int* ptr_c = new int[m * n];
    Tensor<X86>a(Shape({1, 1, m, k}), AK_INT8);
    Tensor<X86>b(Shape({1, 1, k, n}), AK_UINT8);
    Tensor<X86>c(Shape({1, 1, 1, m}), AK_INT32);

    for (int i = 0; i < m * k; i++) {
        ptr_a[i] = 127;
    }

    for (int i = 0; i < k * n; i++) {
        ptr_b[i] = 255;
    }

    int c_offset = 0;
    cblas_gemm_s8u8s32(CblasColMajor,                       // Layout
                       CblasTrans,                // a need to transpose or not
                       CblasNoTrans,                        // b need to transpose or not
                       CblasFixOffset,                      // c_offset_layout
                       m,                      // m
                       n,                          // n
                       k,                                  // k
                       1.0,                                 // scale
                       ptr_a,                              // a
                       k,                                  // lda
                       0,                                   // a_offset
                       ptr_b,                                 // b
                       k,                                  // ldb
                       0,                                   // b_offset
                       0.0,                                 // beta
                       ptr_c,             // c
                       m,                      // ldc
                       &c_offset);
    Context<X86> ctx(0, 1, 1);
    SaberTimer<X86> timer;
    timer.start(ctx);

    for (int i = 0; i < iter; i++) {
        cblas_gemm_s8u8s32(CblasColMajor,                       // Layout
                           CblasTrans,                // a need to transpose or not
                           CblasNoTrans,                        // b need to transpose or not
                           CblasFixOffset,                      // c_offset_layout
                           m,                      // m
                           n,                          // n
                           k,                                  // k
                           1.0,                                 // scale
                           ptr_a,                              // a
                           k,                                  // lda
                           0,                                   // a_offset
                           ptr_b,                                 // b
                           k,                                  // ldb
                           0,                                   // b_offset
                           0.0,                                 // beta
                           ptr_c,             // c
                           m,                      // ldc
                           &c_offset);
    }

    timer.end(ctx);
    double work = 2 * m * n * k;
    double use_time = timer.get_average_ms() / iter;
    double speed = work / use_time / 1000 / 1000;
    LOG(INFO) << m << "," << n << "," << k << "::" << "gfloat " << speed;
}
#endif

TEST(TestSaberFunc, test_op_fc) {
#ifdef USE_CUDA
#endif

#ifdef USE_X86_PLACE
    Env<X86>::env_init();

    if (jit::mayiuse(jit::avx512_core_vnni)) {

        for (auto m : {
                    1, 3, 5, 7
                }) {
            for (auto n : {
                        3, 12, 17
                    }) {
                for (auto k : {
                            7, 16, 22
                        }) {
                    for (auto with_bias : {
                                false, true
                            }) {
                        test_fc_int8<X86, X86>(m, 1, 1, k, n, with_bias);
                    }
                }
            }
        }

        int m = 3;
        int n = 5;
        int k = 7;
        test_fc_int8<X86, X86>(m, 1, 1, k, n, true);
    }

#endif
}


int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}