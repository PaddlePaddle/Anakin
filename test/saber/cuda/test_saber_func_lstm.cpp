

#include <vector>
#include "core/context.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "funcs/impl/cuda/cudnn_helper.h"
#include "funcs/lstm.h"
#include "saber_types.h"
#include "saber/funcs/timer.h"
#include "saber/funcs/impl/cuda/saber_eltwise.h"
#include "saber/funcs/impl/cuda/saber_activation.h"
#include "saber/funcs/impl/cuda/saber_lstm.h"
#include "stdio.h"

#define TEST_X86
using namespace anakin::saber;
cublasHandle_t  cublas_handle;

void test_saber_lstm(int sequence_size = 2, int batch_size = 1, int word_size = 4,
                    int hidden_size = 4) {

    Context<NV> ctx_dev(0, 0, 0);
    Context<X86> ctx_x86(0, 0, 0);
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;

    //std::vector<int> offsets = {0, 20,40, 65, 82, 101};
    std::vector<int> offsets = {0, 3};
    bool is_reverse = false;
    batch_size = offsets.size() - 1;
    Shape input_shape(offsets[offsets.size() - 1], word_size, 1, 1);
    Shape output_shape(offsets[offsets.size() - 1], hidden_size, 1, 1);
    Shape weight_shape(4, word_size + hidden_size, hidden_size, 1);
    Shape bias_shape(4, hidden_size, 1, 1);

    TensorHf4 host_input;
    TensorHf4 host_output_cudnn;
    TensorHf4 host_output_x86;
    TensorHf4 host_weight;
    TensorHf4 host_bias;
    TensorDf4 dev_input;
    TensorDf4 dev_output;
    TensorDf4 dev_weight;
    TensorDf4 dev_bias;
    host_input.re_alloc(input_shape);
    host_output_cudnn.re_alloc(output_shape);
    host_output_x86.re_alloc(output_shape);
    host_weight.re_alloc(weight_shape);
    host_bias.re_alloc(bias_shape);
    dev_input.re_alloc(input_shape);
    dev_output.re_alloc(output_shape);
    dev_weight.re_alloc(weight_shape);
    dev_bias.re_alloc(bias_shape);

    //fill_tensor_host_rand(host_input, -1, 1);
    fill_tensor_host_rand(host_weight, -1, 1);
    //fill_tensor_host_rand(host_bias, -1, 1);
    fill_tensor_host_const(host_input, 1);
    //fill_tensor_host_const(host_weight, 1);
    fill_tensor_host_const(host_bias, 0.5);
    host_input.set_seq_offset(offsets);
    dev_input.set_seq_offset(offsets);

    //    dev_ux.copy_from(host_ux);
    dev_input.copy_from(host_input);
    dev_weight.copy_from(host_weight);
    dev_bias.copy_from(host_bias);

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
                                    is_reverse,
                                    0.f,
                                    1,
                                    1);
    Lstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> cudnn_lstm;
    cudnn_lstm.compute_output_shape(dev_input_vec, dev_output_vec, dev_lstm_param);
    dev_output_vec[0]->reshape(dev_output_vec[0]->valid_shape());

    SABER_CHECK(cudnn_lstm.init(dev_input_vec, dev_output_vec, dev_lstm_param,
                                   SPECIFY, VENDER_IMPL, ctx_dev));
    cudnn_lstm(dev_input_vec, dev_output_vec, dev_lstm_param, ctx_dev);
    dev_output_vec[0]->record_event(ctx_dev.get_compute_stream());
    dev_output_vec[0]->sync();
    host_output_cudnn.reshape(dev_output_vec[0]->valid_shape());
    host_output_cudnn.copy_from(*dev_output_vec[0]);
    

    int test_iter = 0;
    SaberTimer<NV> t1;
    t1.start(ctx_dev);
    for (int i = 0; i < test_iter; ++i) {
        cudnn_lstm(dev_input_vec, dev_output_vec, dev_lstm_param, ctx_dev);
        dev_output_vec[0]->record_event(ctx_dev.get_compute_stream());
        dev_output_vec[0]->sync();
    }
    t1.end(ctx_dev);
    LOG(INFO) << "!!cudnn lstm :" << test_iter << " cudnn test, total time: "
             << t1.get_average_ms();
#if defined(TEST_X86) &&defined(USE_X86_PLACE)
    Lstm<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> x86_lstm;
    LstmParam<TensorHf4> h_lstm_param(&host_weight, 
                                    &host_bias, 
                                    nullptr,
                                    Active_unknow,
                                    Active_sigmoid,
                                    Active_tanh,
                                    Active_tanh,
                                    false,
                                    false,
                                    is_reverse,
                                    0.f,
                                    1,
                                    1);
    x86_lstm.compute_output_shape(h_input_vec, h_output_vec, h_lstm_param);
    h_output_vec[0]->reshape(h_output_vec[0]->valid_shape());

    SABER_CHECK(x86_lstm.init(h_input_vec, h_output_vec, h_lstm_param,
                                   SPECIFY, SABER_IMPL, ctx_x86));
    x86_lstm(h_input_vec, h_output_vec, h_lstm_param, ctx_x86);
    double maxdiff = 0;
    double maxratio = 0;
    tensor_cmp_host(h_output_vec[0]->data(), host_output_cudnn.data(), host_output_cudnn.valid_size(), maxratio, maxdiff);
    if (maxdiff < 1e-5) {
        LOG(INFO)<<"lstm test passed";
    } else {
        LOG(INFO)<<"radio:" << maxratio << " diff:" << maxdiff;
    }
    
    SaberTimer<X86> t2;
    t2.start(ctx_x86);
    for (int i = 0; i < test_iter; ++i) {
        x86_lstm(h_input_vec, h_output_vec, h_lstm_param, ctx_x86);
        h_output_vec[0]->record_event(ctx_x86.get_compute_stream());
        h_output_vec[0]->sync();
    }
    t2.end(ctx_x86);
    LOG(INFO) << "!!x86 lstm :" << test_iter << "x86 lstm test, total time: "
             << t2.get_average_ms();
#endif

   return;

}

TEST(TestSaberFuncNV, test_func_saber_lstm) {

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef Tensor<X86, AK_INT8, NCHW> TensorHINT8;
    typedef Tensor<NV, AK_INT8, NCHW> TensorDINT8;
    //
    //    for(int seq_size:{1,3,5,10,20,30,50,100})
    //        for(int batch_size:{1,3,5,10,20,30,50,100})
    //            for(int word_size:{10,20,30,64,128,256})
    //                for(int hidden_size:{64,128,256,512,1024})
    //                    test_saber_lstm(seq_size,batch_size,word_size,hidden_size);

    test_saber_lstm();

}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
#if defined(TEST_X86) &&defined(USE_X86_PLACE)
    Env<X86>::env_init();
#endif
    Env<NV>::env_init();
//#endif

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
