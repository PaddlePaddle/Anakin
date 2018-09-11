

#include <vector>
#include "core/context.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "funcs/impl/cuda/cudnn_helper.h"
#include "funcs/gru.h"
#include "saber_types.h"
#include "saber/funcs/timer.h"
#include "saber/funcs/impl/cuda/saber_eltwise.h"
#include "saber/funcs/impl/cuda/saber_activation.h"
#include "saber/funcs/impl/cuda/saber_gru.h"
#include "stdio.h"
//#include "cublas.h"

using namespace anakin::saber;



//#define printTensorShape(tensor)\
//do{\
//LOG(INFO)<<"("<<tensor.num()<<","<<tensor.channel()<<","<<tensor.height()<<","<<tensor.width()<<")";\
//}while(0)
//
#define printShape(tensor)\
do{\
LOG(INFO)<<"("<<tensor[0]<<","<<tensor[1]<<","<<tensor[2]<<","<<tensor[3]<<")";\
}while(0)
cublasHandle_t  cublas_handle;

void anakin_NV_gemm2(cublasHandle_t handle, const bool TransA,
                     const bool TransB, const int M, const int N, const int K,
                     const float alpha, const float* A, const float* B, const float beta,
                     float* C) {
    // Note that cublas follows fortran order.
    int lda = (!TransA/* == CblasNoTrans*/) ? K : M;
    int ldb = (!TransB/* == CblasNoTrans*/) ? N : K;
    cublasOperation_t cuTransA =
        (!TransA/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        (!TransB/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA,
                             N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

void anakin_NV_gemm(cublasHandle_t handle, const bool TransA,
                    const bool TransB, const int M, const int N, const int K,
                    const float alpha, const float* A, const float* B, const float beta,
                    float* C) {
    // Note that cublas follows fortran order.
    int lda = (!TransA/* == CblasNoTrans*/) ? K : M;
    int ldb = (!TransB/* == CblasNoTrans*/) ? N : K;
    cublasOperation_t cuTransA =
        (!TransA/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        (!TransB/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA,
                             N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

void anakin_NV_gemm_2(cublasHandle_t handle, const int M, const int N, const int K,
                      const float alpha, const float* A, const float* B, const float beta,
                      float* C) {
    // Note that cublas follows fortran order.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, B, N, A, K, &beta, C, N));
}
//#define FAKEINPUT
#define GRUOFFSET
//#define CUDNNGRU
//#define TEST_X86
void test_saber_gru(int sequence_size = 2, int batch_size = 1, int word_size = 222,
                    int hidden_size = 333) {

#ifdef TEST_X86
    Context<X86> ctx_dev(0, 1, 1);
    typedef Tensor<X86, AK_FLOAT, NCHW_C16> TensorDf4;
    typedef Tensor<X86, AK_FLOAT, NCHW_C16> TensorHf4;
#else
    Context<NV> ctx_dev(0, 1, 1);
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
#endif


#ifdef GRUOFFSET
    std::vector<int> offsets = {0,30};
    bool is_reverse = true;
    batch_size = offsets.size() - 1;
    Shape shape_ux(1, 1, offsets[offsets.size() - 1], hidden_size * 3);
    Shape shape_x(offsets[offsets.size() - 1], word_size, 1, 1);
    Shape shape_out(1, 1, offsets[offsets.size() - 1], hidden_size);
#else
    Shape shape_ux(1, sequence_size, batch_size, hidden_size * 3);
    Shape shape_x(1, sequence_size, batch_size, word_size);
    Shape shape_out(1, sequence_size, batch_size, hidden_size);
#endif

    Shape shape_ux_k(1, 1, 1, hidden_size);
    Shape shape_u(1, 1, word_size, 3 * hidden_size);

    Shape shape_b(1, 1, 3, hidden_size);
    Shape shape_w(1, 1, hidden_size, 3 * hidden_size);
    Shape shape_wu(1, 1,1, hidden_size* 3 * hidden_size+word_size* 3 * hidden_size);
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

#ifdef FAKEINPUT
    fill_tensor_host_rand(host_ux);
    fill_tensor_host_rand(host_u);
    fill_tensor_host_rand(host_x);
    fill_tensor_host_rand(host_b);
    fill_tensor_host_rand(host_w);
    fill_tensor_host_rand(host_h);
#else
    //    readTensorData(host_ux, "host_ux");
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

#ifdef CUDNNGRU
    {
        Shape shape_uw(1, 1, 1, hidden_size * hidden_size * 3 + word_size * hidden_size * 3);
        TensorHf4 host_uw;
        TensorDf4 dev_uw;
        host_uw.re_alloc(shape_uw);
        dev_uw.re_alloc(shape_uw);
#ifdef FAKEINPUT
        fill_tensor_host_rand(host_uw);
#else
        readTensorData(host_uw, "host_uw");
#endif

        dev_uw.copy_from(host_uw);
        GruParam<TensorDf4> cudnn_param(&dev_uw, &dev_b, GRU_CUDNN);
        Gru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> dev_gru_cudnn;
        dev_gru_cudnn.compute_output_shape(output_dev_4d, input_dev_4d, cudnn_param);
        LOG(INFO) << "shape of output =" << dev_out.valid_shape().data()[0] << ","
                  << dev_out.valid_shape().data()[1] << ","
                  << dev_out.valid_shape().data()[2] << ","
                  << dev_out.valid_shape().data()[3];

        output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape());
        SABER_CHECK(dev_gru_cudnn.init(input_dev_4d, output_dev_4d, cudnn_param,
                                       SPECIFY, VENDER_IMPL, ctx_dev));
        Shape cudnn_out_shape = output_dev_4d[0]->get_stride();
        //    printShape(shape);

        dev_gru_cudnn(input_dev_4d, output_dev_4d, cudnn_param, ctx_dev);

        int test_iter = 100;
        SaberTimer<NV> t1;
        t1.start(ctx_dev);

        for (int i = 0; i < test_iter; ++i) {
            dev_gru_cudnn(input_dev_4d, output_dev_4d, cudnn_param, ctx_dev);
            output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
            output_dev_4d[0]->sync();
        }

        t1.end(ctx_dev);
        LOG(INFO) << "!!cudnn care:" << test_iter << "cudnn test, total time: "
                 << t1.get_average_ms() << "avg time : "
               << t1.get_average_ms() / test_iter << " args ["
                 << sequence_size << "," << batch_size << ","
                  << word_size << "," << hidden_size << "]";

        dev_out_bak.copy_from(dev_out);

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

        if (maxdiff <= 0.001) {
            LOG(INFO) << "cudnn passed";
        } else {
            LOG(INFO) << "cudnn failed :" << maxdiff;
        }
    };

#endif

#ifdef GRUOFFSET

    dev_x.set_seq_offset(offsets);
    GruParam<TensorDf4> param(&dev_wu, &dev_b, GRU_ORIGIN,Active_sigmoid_fluid,Active_tanh_fluid,is_reverse);
#else
#ifdef CUDNNGRU
    GruParam<TensorDf4> param(&dev_w, &dev_u, &dev_b, false, GRU_CUDNN);
#else
    GruParam<TensorDf4> param(&dev_w, &dev_u, &dev_b, false, GRU_ORIGIN);
    //    GruParam<TensorDf4> param(&dev_w, &dev_u, &dev_b, false,GRU_CUDNN);
#endif
#endif
#ifdef TEST_X86
    Gru<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW_C16, NCHW_C16, NCHW_C16> dev_gru;
    SaberTimer<X86> t1;
#else
    Gru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> dev_gru;
    SaberTimer<NV> t1;
#endif
    dev_gru.compute_output_shape(input_dev_4d, output_dev_4d, param);
    LOG(INFO) << "shape of output =" << dev_out.valid_shape().data()[0] << ","
              << dev_out.valid_shape().data()[1] << ","
              << dev_out.valid_shape().data()[2] << ","
              << dev_out.valid_shape().data()[3];

    output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape());
    SABER_CHECK(dev_gru.init(input_dev_4d, output_dev_4d, param, SPECIFY, SABER_IMPL, ctx_dev));
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
    //        cudaDeviceSynchronize();
    //        record_dev_tensorfile(W_X_Z,"W_X_Z");
#if (!defined(FAKEINPUT)&&!defined(CUDNNGRU))
    Tensor<X86, AK_FLOAT, NCHW> host_g;
    Tensor<X86, AK_FLOAT, NCHW> compare_g;
    host_g.re_alloc(shape_out);
    compare_g.re_alloc(shape_out);
    host_g.copy_from(dev_out);
//    write_tensorfile(host_g, "host_g.txt");
    readTensorData(compare_g, "host_correct");
//    write_tensorfile(compare_g, "host_correct.txt");
    double maxdiff = 0;
    double maxratio = 0;
    tensor_cmp_host(host_g.data(), compare_g.data(), host_g.valid_size(), maxratio, maxdiff);

    if (abs(maxratio) <= 0.001) {
        LOG(INFO) << "passed  " << maxratio<<","<<maxdiff;
    } else {
        LOG(INFO) << "failed : ratio " << maxratio<<","<<maxdiff;
    }

#elif (!defined(FAKEINPUT)&&defined(CUDNNGRU))
    Tensor<X86, AK_FLOAT, NCHW> host_g;
    Tensor<X86, AK_FLOAT, NCHW> compare_g;
    host_g.re_alloc(shape_out);
    compare_g.re_alloc(shape_out);
    host_g.copy_from(dev_out);
    compare_g.copy_from(dev_out_bak);
    write_tensorfile(host_g, "host_g.txt");
    write_tensorfile(compare_g, "host_correct.txt");
    double maxdiff = 0;
    double maxratio = 0;
    int max_index = 0;
    tensor_cmp_host(host_g.data(), compare_g.data(), host_g.valid_size(), maxratio, maxdiff);

    if (maxdiff <= 0.001) {
        LOG(INFO) << "cudnn saber passed";
    } else {
        LOG(INFO) << "cudnn saber failed :" << maxdiff << " , index = " << max_index;
    }

#endif

    return;

    //    return;
}

TEST(TestSaberFuncNV, test_func_saber_gru) {

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef Tensor<X86, AK_INT8, NCHW> TensorHINT8;
    typedef Tensor<NV, AK_INT8, NCHW> TensorDINT8;
    //
    //    for(int seq_size:{1,3,5,10,20,30,50,100})
    //        for(int batch_size:{1,3,5,10,20,30,50,100})
    //            for(int word_size:{10,20,30,64,128,256})
    //                for(int hidden_size:{64,128,256,512,1024})
    //                    test_saber_gru(seq_size,batch_size,word_size,hidden_size);

    test_saber_gru();

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
