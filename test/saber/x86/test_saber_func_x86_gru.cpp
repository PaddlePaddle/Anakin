
#include <vector>
#include "core/context.h"
#include "funcs/debug.h"
#include "tensor_op.h"
#include "funcs/gru.h"
#include "saber_types.h"
#include "saber/funcs/timer.h"
#include "stdio.h"
#include "x86_test_common.h"
#include "test_saber_func_x86.h"
#include "saber/funcs/impl/x86/saber_gru.h"

#include <mkl_service.h>
#ifdef USE_OPENMP
#include "omp.h"
#endif

//#include "cublas.h"

using namespace anakin::saber;

#define printShape(tensor)\
do{\
LOG(INFO)<<"("<<tensor[0]<<","<<tensor[1]<<","<<tensor[2]<<","<<tensor[3]<<")";\
}while(0)

//#define FAKEINPUT
#define GRUOFFSET
//#define CUDNNGRU
#define TEST_X86
//#define INNER_TEST

typedef Tensor<X86, AK_FLOAT, NCHW> TensorDf4;
typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
#ifdef INNER_TEST
void compute_compare_correct(TensorDf4 dev_x,TensorDf4 last_dev_out,GruParam<TensorDf4> param){
    Context<X86> ctx_dev(0, 1, 1);
    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    TensorDf4 new_dev_out;
    new_dev_out.re_alloc(last_dev_out.valid_shape());

    input_dev_4d.push_back(&dev_x);
    output_dev_4d.push_back(&new_dev_out);

    SaberGru<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> gru;
    SABER_CHECK(gru.init(input_dev_4d, output_dev_4d, param, ctx_dev));

    int test_iter = 1;
    SaberTimer<X86> t1;
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        gru.naiv_gru(input_dev_4d, output_dev_4d, param);
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
    }

    t1.end(ctx_dev);
    LOG(INFO) << "!!saber care: naive" << test_iter << "test, total time: " << t1.get_average_ms();

    double maxdiff = 0;
    double maxratio = 0;
    tensor_cmp_host(last_dev_out.data(), new_dev_out.data(), last_dev_out.valid_size(), maxratio, maxdiff);

    if (abs(maxratio) <= 0.001) {
                LOG(INFO) << "passed  " << maxratio<<" : "<<maxdiff;
    } else {
                LOG(INFO) << "failed : ratio " << maxratio<<" : "<<maxdiff;
                exit(-1);
    }

}
#endif
void test_saber_gru_x86(int sequence_size = 2, int batch_size = 1, int word_size = 52,
                    int hidden_size = 36) {

#ifdef USE_OPENMP
      omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    mkl_set_num_threads(1);
    Context<X86> ctx_dev(0, 1, 1);
    std::vector<int> offsets = {0,39};
    ImplEnum test_mode=SABER_IMPL;
//    ImplEnum test_mode=VENDER_IMPL;
    bool is_reverse = false;
    batch_size = offsets.size() - 1;
    Shape shape_ux(1, 1, offsets[offsets.size() - 1], hidden_size * 3);
    Shape shape_x(offsets[offsets.size() - 1], word_size, 1, 1);
    Shape shape_out(1, 1, offsets[offsets.size() - 1], hidden_size);


    Shape shape_ux_k(1, 1, 1, hidden_size);
    Shape shape_u(1, 1, word_size, 3 * hidden_size);

    Shape shape_b(1, 1, 3, hidden_size);
    Shape shape_w(1, 1, hidden_size, 3 * hidden_size);
    Shape shape_wu(1, 1,1, (hidden_size* 3 * hidden_size+word_size* 3 * hidden_size));
    Shape shape_h(1, 1, batch_size, hidden_size);


    TensorHf4 host_ux;//z,r,o
    TensorHf4 host_x;//z,r,o
    TensorHf4 host_b;//z,r,o
    TensorHf4 host_wu;//z,r,o
    TensorHf4 host_h;
    TensorHf4 host_out;
    TensorHf4 host_out_bak;

    TensorDf4 dev_ux;
    TensorDf4 dev_x;
    TensorDf4 dev_b;
    TensorDf4 dev_wu;
    TensorDf4 dev_h;
    TensorDf4 dev_out;
    TensorDf4 dev_out_bak;

    //    host_ux.re_alloc(shape_ux);
    host_x.re_alloc(shape_x);
    host_b.re_alloc(shape_b);
    host_wu.re_alloc(shape_wu);
    host_h.re_alloc(shape_h);
    host_out.re_alloc(shape_out);
    host_out_bak.re_alloc(shape_out);

    //    dev_ux.re_alloc(shape_ux);
    dev_x.re_alloc(shape_x);
    dev_b.re_alloc(shape_b);
    dev_wu.re_alloc(shape_wu);
    dev_h.re_alloc(shape_h);
    dev_out.re_alloc(shape_out);
    dev_out_bak.re_alloc(shape_out);
#ifdef INNER_TEST
    fill_tensor_host_rand(host_wu,-1,1);
    fill_tensor_host_rand(host_x,-1,1);
    fill_tensor_host_rand(host_b,-1,1);
    fill_tensor_host_rand(host_h,-1,1);
#else
    readTensorData(host_wu, "host_wu");
    readTensorData(host_x, "host_x");
    readTensorData(host_b, "host_b");
    readTensorData(host_h, "host_h");
#endif



    //    dev_ux.copy_from(host_ux);
    dev_x.copy_from(host_x);
    dev_b.copy_from(host_b);
    dev_wu.copy_from(host_wu);
    dev_h.copy_from(host_h);

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;
    input_dev_4d.push_back(&dev_x);
//    input_dev_4d.push_back(&dev_h);
    output_dev_4d.push_back(&dev_out);


    dev_x.set_seq_offset(offsets);
    GruParam<TensorDf4> param(&dev_wu, &dev_b, GRU_ORIGIN,Active_sigmoid,Active_tanh,is_reverse);


    Gru<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> dev_gru;
    SaberTimer<X86> t1;

    dev_gru.compute_output_shape(input_dev_4d, output_dev_4d, param);
            LOG(INFO) << "shape of output =" << dev_out.valid_shape().data()[0] << ","
                      << dev_out.valid_shape().data()[1] << ","
                      << dev_out.valid_shape().data()[2] << ","
                      << dev_out.valid_shape().data()[3];

    output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape());
    SABER_CHECK(dev_gru.init(input_dev_4d, output_dev_4d, param, SPECIFY, test_mode, ctx_dev));
    Shape shape = output_dev_4d[0]->get_stride();
    //    printShape(shape);

    dev_gru(input_dev_4d, output_dev_4d, param, ctx_dev);

    int test_iter = 1000;

    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        dev_gru(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
    }

    t1.end(ctx_dev);
            LOG(INFO) << "!!saber care:" << test_iter << "test, total time: " << t1.get_average_ms() <<
                      "avg time : " << \
              t1.get_average_ms() / test_iter << " args [" \
                << sequence_size << "," << batch_size << ","
                      << word_size << "," << hidden_size << "]";
#ifdef INNER_TEST
    compute_compare_correct(dev_x,dev_out,param);
    return;
#else
    Tensor<X86, AK_FLOAT, NCHW> host_g;
    Tensor<X86, AK_FLOAT, NCHW> compare_g;
    host_g.re_alloc(shape_out);
    compare_g.re_alloc(shape_out);
    host_g.copy_from(dev_out);
    write_tensorfile(host_g, "host_g.txt");
    readTensorData(compare_g, "host_correct");
    write_tensorfile(compare_g, "host_correct.txt");
    double maxdiff = 0;
    double maxratio = 0;
    tensor_cmp_host(host_g.data(), compare_g.data(), host_g.valid_size(), maxratio, maxdiff);

    if (abs(maxratio) <= 0.001) {
                LOG(INFO) << "passed  " << maxratio<<","<<maxdiff<<",?="<<abs(maxratio);
    } else {
                LOG(INFO) << "failed : ratio " << maxratio<<","<<maxdiff;
    }
    return;
#endif
    //    return;
}

TEST(TestSaberFuncX86, test_func_saber_gru_x86) {

    test_saber_gru_x86();

}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
#ifdef TEST_X86
    Env<X86>::env_init();
#else
    Env<NV>::env_init();
#endif

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
