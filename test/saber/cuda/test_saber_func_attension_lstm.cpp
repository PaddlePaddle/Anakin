

#include <vector>
#include "core/context.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "funcs/impl/cuda/cudnn_helper.h"
#include "saber_types.h"
#include "saber/funcs/timer.h"
#include "saber/funcs/impl/cuda/saber_attension_lstm.h"
#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_attension_lstm.h"
#endif
#include "saber/funcs/attension_lstm.h"
#include "stdio.h"

#define TEST_X86
using namespace anakin::saber;

void test_saber_attension_lstm(int sequence_size = 2, int batch_size = 1, int word_size = 30,
                    int hidden_size = 15) {

    Context<NV> ctx_dev(0, 0, 0);
    Context<X86> ctx_x86(0, 0, 0);
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    std::vector<int> fc_num_output = {1, 1};
    
    

    std::vector<int> offsets = {0, 3, 7};
    bool is_reverse = false;
    batch_size = offsets.size() - 1;
    Shape input_shape(offsets[offsets.size() - 1], word_size, 1, 1);
    Shape output_shape(offsets[offsets.size() - 1], hidden_size, 1, 1);

    Shape weight_shape(4, word_size + hidden_size, hidden_size, 1);
    Shape bias_shape(4, hidden_size, 1, 1);

    Shape fc_0_weight_shape(word_size + hidden_size, fc_num_output[0], 1, 1);
    Shape fc_0_bias_shape(1, fc_num_output[0], 1, 1);
    Shape fc_1_weight_shape(fc_num_output[0], fc_num_output[1], 1, 1);
    Shape fc_1_bias_shape(1, fc_num_output[1], 1, 1);

    TensorHf4 h_input;
    TensorHf4 h_output_cudnn;
    TensorHf4 h_output_x86;
    TensorHf4 h_weight;
    TensorHf4 h_bias;
    TensorHf4 h_fc_0_weight;
    TensorHf4 h_fc_0_bias;
    TensorHf4 h_fc_1_weight;
    TensorHf4 h_fc_1_bias;
    
    TensorDf4 d_input;
    TensorDf4 d_output;
    TensorDf4 d_weight;
    TensorDf4 d_bias;
    TensorDf4 d_fc_0_weight;
    TensorDf4 d_fc_0_bias;
    TensorDf4 d_fc_1_weight;
    TensorDf4 d_fc_1_bias;

    h_input.re_alloc(input_shape);
    h_output_cudnn.re_alloc(output_shape);
    h_output_x86.re_alloc(output_shape);
    h_weight.re_alloc(weight_shape);
    h_bias.re_alloc(bias_shape);
    h_fc_0_weight.re_alloc(fc_0_weight_shape);
    h_fc_0_bias.re_alloc(fc_0_bias_shape);
    h_fc_1_weight.re_alloc(fc_1_weight_shape);
    h_fc_1_bias.re_alloc(fc_1_bias_shape);

    d_input.re_alloc(input_shape);
    d_output.re_alloc(output_shape);
    d_weight.re_alloc(weight_shape);
    d_bias.re_alloc(bias_shape);
    d_fc_0_weight.re_alloc(fc_0_weight_shape);
    d_fc_0_bias.re_alloc(fc_0_bias_shape);
    d_fc_1_weight.re_alloc(fc_1_weight_shape);
    d_fc_1_bias.re_alloc(fc_1_bias_shape);

    fill_tensor_host_rand(h_input, -1.f, 1.f);
    fill_tensor_host_rand(h_weight, -1.f, 1.f);
    fill_tensor_host_rand(h_bias, -1.f, 1.f);
    fill_tensor_host_rand(h_fc_0_weight, -1.f, 1.f);
    fill_tensor_host_rand(h_fc_0_bias, -1.f, 1.f);
    fill_tensor_host_rand(h_fc_1_weight, -1.f, 1.f);
    fill_tensor_host_rand(h_fc_1_bias, -1.f, 1.f);
    h_input.set_seq_offset(offsets);
    d_input.set_seq_offset(offsets);

    //    d_ux.copy_from(h_ux);
    d_input.copy_from(h_input);
    d_weight.copy_from(h_weight);
    d_bias.copy_from(h_bias);
    d_fc_0_weight.copy_from(h_fc_0_weight);
    d_fc_0_bias.copy_from(h_fc_0_bias);
    d_fc_1_weight.copy_from(h_fc_1_weight);
    d_fc_1_bias.copy_from(h_fc_1_bias);

    std::vector<TensorDf4*> d_input_vec;
    std::vector<TensorDf4*> d_output_vec;
    d_input_vec.push_back(&d_input);
    d_output_vec.push_back(&d_output);

    std::vector<TensorHf4*> h_input_vec;
    std::vector<TensorHf4*> h_output_vec;
    h_input_vec.push_back(&h_input);
    h_output_vec.push_back(&h_output_x86);

    LstmParam<TensorDf4> d_lstm_param(&d_weight, 
                                    &d_bias, 
                                    nullptr,
                                    Active_unknow,
                                    Active_sigmoid,
                                    Active_tanh,
                                    Active_tanh,
                                    false,
                                    false,
                                    false,
                                    0.f,
                                    1,
                                    1);
    FcParam<TensorDf4> fc_0_param(&d_fc_0_weight, &d_fc_0_bias, fc_num_output[0], 1);
    FcParam<TensorDf4> fc_1_param(&d_fc_1_weight, &d_fc_1_bias, fc_num_output[1], 1);
    std::vector<FcParam<TensorDf4>> fc_vec;
    fc_vec.push_back(fc_0_param);
    fc_vec.push_back(fc_1_param);
    AttensionParam<TensorDf4> attn_param(fc_vec);
    AttensionLstmParam<TensorDf4> d_attn_lstm_param(attn_param, d_lstm_param);

    AttensionLstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> attn_lstm;
    attn_lstm.compute_output_shape(d_input_vec, d_output_vec, d_attn_lstm_param);
    d_output_vec[0]->reshape(d_output_vec[0]->valid_shape());


    SABER_CHECK(attn_lstm.init(d_input_vec, d_output_vec, d_attn_lstm_param,
                                   SPECIFY, SABER_IMPL, ctx_dev));
    attn_lstm(d_input_vec, d_output_vec, d_attn_lstm_param, ctx_dev);
    d_output_vec[0]->record_event(ctx_dev.get_compute_stream());
    d_output_vec[0]->sync();
    h_output_cudnn.reshape(d_output_vec[0]->valid_shape());
    h_output_cudnn.copy_from(*d_output_vec[0]);
    

    //int test_iter = 100;
    //SaberTimer<NV> t1;
    //t1.start(ctx_dev);
    //for (int i = 0; i < test_iter; ++i) {
    //    attn_lstm(d_input_vec, d_output_vec, d_attn_lstm_param, ctx_dev);
    //    d_output_vec[0]->record_event(ctx_dev.get_compute_stream());
    //    d_output_vec[0]->sync();
    //}
    //t1.end(ctx_dev);
    //LOG(INFO) << "!!cudnn lstm :" << test_iter << " cudnn test, total time: "
    //         << t1.get_average_ms()/test_iter;
#if defined(TEST_X86)&&defined(USE_X86_PLACE)
    LstmParam<TensorHf4> h_lstm_param(&h_weight,
                                    &h_bias,
                                    nullptr,
                                    Active_unknow,
                                    Active_sigmoid,
                                    Active_tanh,
                                    Active_tanh,
                                    false,
                                    false,
                                    false,
                                    0.f,
                                    1,
                                    1);
    FcParam<TensorHf4> h_fc_0_param(&h_fc_0_weight, &h_fc_0_bias, fc_num_output[0], 1);
    FcParam<TensorHf4> h_fc_1_param(&h_fc_1_weight, &h_fc_1_bias, fc_num_output[1], 1);
    std::vector<FcParam<TensorHf4>> h_fc_vec;
    h_fc_vec.push_back(h_fc_0_param);
    h_fc_vec.push_back(h_fc_1_param);
    AttensionParam<TensorHf4> h_attn_param(h_fc_vec);
    AttensionLstmParam<TensorHf4> h_attn_lstm_param(h_attn_param, h_lstm_param);

    AttensionLstm<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> h_attn_lstm;
    h_attn_lstm.compute_output_shape(h_input_vec, h_output_vec, h_attn_lstm_param);
    h_output_vec[0]->reshape(h_output_vec[0]->valid_shape());


    SABER_CHECK(h_attn_lstm.init(h_input_vec, h_output_vec, h_attn_lstm_param,
                                   SPECIFY, SABER_IMPL, ctx_x86));
    h_attn_lstm(h_input_vec, h_output_vec, h_attn_lstm_param, ctx_x86);
    //print_tensor_host(*h_output_vec[0]);
    //print_tensor_host(h_output_cudnn);
    double maxratio, maxdiff;
    tensor_cmp_host(h_output_vec[0]->data(), h_output_cudnn.data(), h_output_cudnn.valid_size(), maxratio, maxdiff);
    if (maxdiff < 1e-5) {
        LOG(INFO)<<"lstm test passed";
    } else {
        LOG(INFO)<<"radio:" << maxratio << " diff:" << maxdiff;
    }
    
    //SaberTimer<X86> t2;
    //t2.start(ctx_x86);
    //for (int i = 0; i < test_iter; ++i) {
    //    h_attn_lstm(h_input_vec, h_output_vec, h_attn_lstm_param, ctx_x86);
    //}
    //t2.end(ctx_x86);
    //LOG(INFO) << "!!x86 lstm :" << test_iter << "x86 test, total time: "
    //         << t2.get_average_ms()/ test_iter;
#endif

   return;

}

TEST(TestSaberFuncNV, test_func_saber_lstm) {

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef Tensor<X86, AK_INT8, NCHW> TensorHINT8;
    typedef Tensor<NV, AK_INT8, NCHW> TensorDINT8;

    test_saber_attension_lstm();

}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
#if defined(TEST_X86)&&defined(USE_X86_PLACE)
    Env<X86>::env_init();
#endif
//#else
    Env<NV>::env_init();
//#endif

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
