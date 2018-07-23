

#include <vector>
#include "core/context.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "funcs/impl/cuda/cudnn_helper.h"
#include "saber_types.h"
#include "saber/funcs/timer.h"
#include "saber/funcs/impl/cuda/saber_attension_lstm.h"
#include "saber/funcs/attension_lstm.h"
#include "stdio.h"

#define TEST_X86
using namespace anakin::saber;

void test_saber_attension_lstm(int sequence_size = 2, int batch_size = 1, int word_size = 30,
                    int hidden_size = 10) {

    Context<NV> ctx_dev(0, 0, 0);
    Context<X86> ctx_x86(0, 0, 0);
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    std::vector<int> fc_num_output = {10, 1};
    
    

    std::vector<int> offsets = {0, 20,40, 65, 82, 101};
    bool is_reverse = true;
    batch_size = offsets.size() - 1;
    Shape input_shape(offsets[offsets.size() - 1], word_size, 1, 1);
    Shape output_shape(offsets[offsets.size() - 1], hidden_size, 1, 1);

    Shape weight_shape(4, word_size + hidden_size, hidden_size, 1);
    Shape bias_shape(4, hidden_size, 1, 1);

    Shape fc_0_weight_shape(word_size + hidden_size, fc_num_output[0], 1, 1);
    Shape fc_0_bias_shape(1, fc_num_output[0], 1, 1);
    Shape fc_1_weight_shape(fc_num_output[0], fc_num_output[1], 1, 1);
    Shape fc_1_bias_shape(1, fc_num_output[1], 1, 1);

    TensorHf4 host_input;
    TensorHf4 host_output_cudnn;
    TensorHf4 host_output_x86;
    TensorHf4 host_weight;
    TensorHf4 host_bias;
    TensorHf4 host_fc_0_weight;
    TensorHf4 host_fc_0_bias;
    TensorHf4 host_fc_1_weight;
    TensorHf4 host_fc_1_bias;
    
    TensorDf4 dev_input;
    TensorDf4 dev_output;
    TensorDf4 dev_weight;
    TensorDf4 dev_bias;
    TensorDf4 dev_fc_0_weight;
    TensorDf4 dev_fc_0_bias;
    TensorDf4 dev_fc_1_weight;
    TensorDf4 dev_fc_1_bias;

    host_input.re_alloc(input_shape);
    host_output_cudnn.re_alloc(output_shape);
    host_output_x86.re_alloc(output_shape);
    host_weight.re_alloc(weight_shape);
    host_bias.re_alloc(bias_shape);
    host_fc_0_weight.re_alloc(fc_0_weight_shape);
    host_fc_0_bias.re_alloc(fc_0_bias_shape);
    host_fc_1_weight.re_alloc(fc_1_weight_shape);
    host_fc_1_bias.re_alloc(fc_1_bias_shape);

    dev_input.re_alloc(input_shape);
    dev_output.re_alloc(output_shape);
    dev_weight.re_alloc(weight_shape);
    dev_bias.re_alloc(bias_shape);
    dev_fc_0_weight.re_alloc(fc_0_weight_shape);
    dev_fc_0_bias.re_alloc(fc_0_bias_shape);
    dev_fc_1_weight.re_alloc(fc_1_weight_shape);
    dev_fc_1_bias.re_alloc(fc_1_bias_shape);

    fill_tensor_host_rand(host_input);
    fill_tensor_host_rand(host_weight);
    fill_tensor_host_rand(host_bias);
    fill_tensor_host_rand(host_fc_0_weight);
    fill_tensor_host_rand(host_fc_0_bias);
    fill_tensor_host_rand(host_fc_1_weight);
    fill_tensor_host_rand(host_fc_1_bias);
    host_input.set_seq_offset(offsets);
    dev_input.set_seq_offset(offsets);

    //    dev_ux.copy_from(host_ux);
    dev_input.copy_from(host_input);
    dev_weight.copy_from(host_weight);
    dev_bias.copy_from(host_bias);
    dev_fc_0_weight.copy_from(host_fc_0_weight);
    dev_fc_0_bias.copy_from(host_fc_0_bias);
    dev_fc_1_weight.copy_from(host_fc_1_weight);
    dev_fc_1_bias.copy_from(host_fc_1_bias);

    std::vector<TensorDf4*> dev_input_vec;
    std::vector<TensorDf4*> dev_output_vec;
    dev_input_vec.push_back(&dev_input);
    dev_output_vec.push_back(&dev_output);

    std::vector<TensorHf4*> h_input_vec;
    std::vector<TensorHf4*> h_output_vec;
    h_input_vec.push_back(&host_input);
    h_output_vec.push_back(&host_output_x86);

    LstmParam<TensorDf4> dev_lstm_param(&dev_weight, 
                                    &dev_bias, 
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
    FcParam<TensorDf4> fc_0_param(&dev_fc_0_weight, &dev_fc_0_bias, fc_num_output[0], 1);
    FcParam<TensorDf4> fc_1_param(&dev_fc_1_weight, &dev_fc_1_bias, fc_num_output[1], 1);
    std::vector<FcParam<TensorDf4>*> fc_vec;
    fc_vec.push_back(&fc_0_param);
    fc_vec.push_back(&fc_1_param);
    AttensionParam<TensorDf4> attn_param(fc_vec);
    AttensionLstmParam<TensorDf4> dev_attn_lstm_param(attn_param, dev_lstm_param);

    AttensionLstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> attn_lstm;
    attn_lstm.compute_output_shape(dev_input_vec, dev_output_vec, dev_attn_lstm_param);
    dev_output_vec[0]->reshape(dev_output_vec[0]->valid_shape());


    SABER_CHECK(attn_lstm.init(dev_input_vec, dev_output_vec, dev_attn_lstm_param,
                                   SPECIFY, SABER_IMPL, ctx_dev));
    attn_lstm(dev_input_vec, dev_output_vec, dev_attn_lstm_param, ctx_dev);
    dev_output_vec[0]->record_event(ctx_dev.get_compute_stream());
    dev_output_vec[0]->sync();
    host_output_cudnn.reshape(dev_output_vec[0]->valid_shape());
    host_output_cudnn.copy_from(*dev_output_vec[0]);
    

    int test_iter = 100;
    SaberTimer<NV> t1;
    t1.start(ctx_dev);
    for (int i = 0; i < test_iter; ++i) {
        attn_lstm(dev_input_vec, dev_output_vec, dev_attn_lstm_param, ctx_dev);
        dev_output_vec[0]->record_event(ctx_dev.get_compute_stream());
        dev_output_vec[0]->sync();
    }
    t1.end(ctx_dev);
    LOG(INFO) << "!!cudnn lstm :" << test_iter << " cudnn test, total time: "
             << t1.get_average_ms();
#ifdef TEST_X86
    //tensor_cmp_host(h_output_vec[0]->data(), host_output_cudnn.data(), host_output_cudnn.valid_size(), maxratio, maxdiff);
    //if (maxdiff < 1e-5) {
    //    LOG(INFO)<<"lstm test passed";
    //} else {
    //    LOG(INFO)<<"radio:" << maxratio << " diff:" << maxdiff;
    //}
    
    //SaberTimer<X86> t2;
    //t2.start(ctx_x86);
    //for (int i = 0; i < test_iter; ++i) {
    //    x86_lstm(h_input_vec, h_output_vec, h_lstm_param, ctx_x86);
    //    h_output_vec[0]->record_event(ctx_x86.get_compute_stream());
    //    h_output_vec[0]->sync();
    //}
    //t2.end(ctx_x86);
    //LOG(INFO) << "!!x86 lstm :" << test_iter << "cudnn test, total time: "
    //         << t2.get_average_ms();
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
//#ifdef TEST_X86
    Env<X86>::env_init();
//#else
    Env<NV>::env_init();
//#endif

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
