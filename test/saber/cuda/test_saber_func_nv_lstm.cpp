#include "test_saber_func_NV.h"
#include "saber/core/context.h"
#include "saber/funcs/lstm.h"
#include "debug.h"
using namespace anakin::saber;
typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
void py_lstm(int word_size = 38,
             int hidden_size = 15){
    Context<NV> ctx_dev(0, 1, 1);
    std::vector<int> offsets = {0, 3};
    ImplEnum test_mode=SABER_IMPL;
//    ImplEnum test_mode=VENDER_IMPL;
    bool is_reverse = true;
    bool with_peephole= false;
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
    TensorDf4 dev_x(shape_x);
    TensorDf4 dev_weight(shape_weight);
    TensorDf4 dev_bias(shape_bias);
    TensorDf4 dev_hidden_out(shape_h);
    readTensorData(host_weight, "host_w");
    readTensorData(host_x, "host_x");
    readTensorData(host_bias, "host_b");
    dev_x.copy_from(host_x);
    dev_weight.copy_from(host_weight);
    dev_bias.copy_from(host_bias);

    host_x.set_seq_offset(offsets);
    dev_x.set_seq_offset(offsets);
    LstmParam<TensorDf4> param(&dev_weight, &dev_bias,nullptr,Active_unknow,Active_sigmoid,Active_tanh,Active_tanh,
                               with_peephole,false,is_reverse);
    Lstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> lstm_op;

    std::vector<TensorDf4*> inputs;
    std::vector<TensorDf4*> outputs;
    inputs.push_back(&dev_x);
    outputs.push_back(&dev_hidden_out);

    SABER_CHECK(lstm_op.init(inputs, outputs, param, SPECIFY, test_mode, ctx_dev));
    SABER_CHECK(lstm_op.compute_output_shape(inputs, outputs, param));
    outputs[0]->re_alloc(outputs[0]->valid_shape());
    SABER_CHECK(lstm_op(inputs, outputs, param, ctx_dev));
    outputs[0]->record_event(ctx_dev.get_compute_stream());
    outputs[0]->sync();

    SaberTimer<NV> t1;
    t1.start(ctx_dev);
    int test_iter = 1000;
    for (int i = 0; i < test_iter; ++i) {
        SABER_CHECK(lstm_op(inputs, outputs, param, ctx_dev));
        outputs[0]->record_event(ctx_dev.get_compute_stream());
        outputs[0]->sync();
    }
    t1.end(ctx_dev);
    LOG(INFO) << "!!saber care:" << test_iter << "test, total time: " << t1.get_average_ms() <<
              "avg time : " << t1.get_average_ms() / test_iter << " args [" << word_size << "," << hidden_size << "]";

    host_hidden_out.copy_from(dev_hidden_out);
    TensorHf4 compare_g(shape_h);
    readTensorData(compare_g, "host_correct");
//    write_tensorfile(host_hidden_out, "host_g.txt");
//    write_tensorfile(compare_g, "host_correct.txt");
    double maxdiff = 0;
    double maxratio = 0;
    tensor_cmp_host(host_hidden_out.data(), compare_g.data(), host_hidden_out.valid_size(), maxratio, maxdiff);
    if (abs(maxratio) <= 0.001) {
        LOG(INFO) << "passed  " << maxratio<<","<<maxdiff<<",?="<<abs(maxratio);
    } else {
        LOG(INFO) << "failed : ratio " << maxratio<<","<<maxdiff;
    }

}
TEST(TestSaberFuncNV, test_lstm){
    py_lstm();
}
int main(int argc, const char** argv) {
    logger::init(argv[0]);
    Env<NV>::env_init();
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}