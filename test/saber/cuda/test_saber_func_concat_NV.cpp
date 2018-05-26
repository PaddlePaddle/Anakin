#include "core/context.h"
#include "funcs/concat.h"
#include "test_saber_func_concat_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncConcatNV, test_func_concat_NV) {

    Env<NV>::env_init();
    typedef TargetWrapper<NV> API;

    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef TensorDf4::Dtype dtype;

    const int test_iter = 100;

    int concat_axis = 2; // height
    int w_in = 8;
    int sum_hin = 8;
    std::vector<int> h_in = {1, 2, 3, 2};
    int ch_in = 2;
    int num_in = 2;

    Shape shape_out(num_in, ch_in, sum_hin, w_in);

    ConcatParam<TensorDf4> param(concat_axis);

    LOG(INFO) << " input size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in[0] << ", width=" << w_in;
    LOG(INFO) << "input 4 tensor: h = " << h_in[0] << ", " << \
              h_in[1] << ", " << h_in[2] << ", " << h_in[3];
    LOG(INFO) << "concat axis= " << param.axis;

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    TensorDf4 tdev1, tdev2, tdev3, tdev4;
    Tensor<X86, AK_FLOAT, NCHW> th1, th2, th3, th4;
    Shape sh1(num_in, ch_in, h_in[0], w_in);
    Shape sh2(num_in, ch_in, h_in[1], w_in);
    Shape sh3(num_in, ch_in, h_in[2], w_in);
    Shape sh4(num_in, ch_in, h_in[3], w_in);
    th1.re_alloc(sh1);
    th2.re_alloc(sh2);
    th3.re_alloc(sh3);
    th4.re_alloc(sh4);
    tdev1.re_alloc(sh1);
    tdev2.re_alloc(sh2);
    tdev3.re_alloc(sh3);
    tdev4.re_alloc(sh4);

    for (int j = 0; j < th1.size(); ++j) {
        th1.mutable_data()[j] = j;
    }

    for (int j = 0; j < th2.size(); ++j) {
        th2.mutable_data()[j] = j;
    }

    for (int j = 0; j < th3.size(); ++j) {
        th3.mutable_data()[j] = j;
    }

    for (int j = 0; j < th4.size(); ++j) {
        th4.mutable_data()[j] = j;
    }

    tdev1.copy_from(th1);
    tdev2.copy_from(th2);
    tdev3.copy_from(th3);
    tdev4.copy_from(th4);
    input_dev_4d.push_back(&tdev1);
    input_dev_4d.push_back(&tdev2);
    input_dev_4d.push_back(&tdev3);
    input_dev_4d.push_back(&tdev4);

    // start Reshape & doInfer
    Context<NV> ctx_dev(0, 1, 1);

    Concat<NV, AK_FLOAT> concat_dev;

    TensorDf4 tdev_out;

    output_dev_4d.push_back(&tdev_out);

    LOG(INFO) << "concat compute output shape";
    concat_dev.compute_output_shape(input_dev_4d, output_dev_4d, param);
    LOG(INFO) << "output shape: " << tdev_out.valid_shape()[0] << ", " \
              << tdev_out.valid_shape()[1] << ", " << tdev_out.valid_shape()[2] \
              << ", " << tdev_out.valid_shape()[3];
    tdev_out.re_alloc(shape_out);

    LOG(INFO) << "concat initialization";
    concat_dev.init(input_dev_4d, output_dev_4d, param, SPECIFY, SABER_IMPL, ctx_dev);

    LOG(INFO) << "concat compute";
    SaberTimer<NV> t1;
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        concat_dev(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->sync();
    }

    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    printf("total time : %.4f, avg time : %.4f\n", ts, ts / test_iter);
    CUDA_POST_KERNEL_CHECK;
    print_tensor_device(*output_dev_4d[0]);
    cudaDeviceSynchronize();
}

TEST(TestSaberFuncConcatNV, test_NV_concat_with_shared_buffer) {

    Env<NV>::env_init();
    typedef TargetWrapper<NV> API;

    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef std::vector<Shape> Shape_v;
    typedef TensorDf4::Dtype dtype;

    //! runtime context
    Context<NV> ctx_dev(0, 1, 1);

    //! constructor of concat op
    Concat<NV, AK_FLOAT> concat_dev;

    int concat_axis = 3; // channel
    int sum_w_in = 8;
    std::vector<int> w_in = {4, 4};
    int h_in = 4;
    std::vector<int> ch_in = {4, 4};
    int sum_ch_in = 4;
    int num_in = 2;

    Shape shape_out(num_in, sum_ch_in, h_in, sum_w_in);

    ConcatParam<TensorDf4> param(concat_axis);

    LOG(INFO) << " input size, num=" << num_in << ", channel=" << \
              ch_in[0] << ", height=" << h_in << ", width=" << w_in[0];
    LOG(INFO) << "input 4 tensor: width = " << w_in[0] << ", " << w_in[1];
    LOG(INFO) << "concat axis= " << param.axis;

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    //! declare input tensors
    TensorDf4 tdev_in, tdev1, tdev2;
    TensorHf4 th1, th2;
    //! input shape
    Shape sh1(num_in, sum_ch_in, h_in, w_in[0]);
    Shape sh2(num_in, sum_ch_in, h_in, w_in[1]);
    Shape os1(0, 0, 0, 0);
    Shape os2(0, 0, 0, w_in[0]);
    //! alloc memory and data
    tdev_in.re_alloc(shape_out);
    th1.re_alloc(sh1);
    th2.re_alloc(sh2);

    for (int j = 0; j < th1.size(); ++j) {
        th1.mutable_data()[j] = 1;
    }

    for (int j = 0; j < th2.size(); ++j) {
        th2.mutable_data()[j] = 2;
    }

    //! shared from other tensor
    tdev1.share_sub_buffer(tdev_in, sh1, os1);
    tdev2.share_sub_buffer(tdev_in, sh2, os2);
    //! prepare input tensor vector
    input_dev_4d.push_back(&tdev1);
    input_dev_4d.push_back(&tdev2);

    TensorDf4 tdev_out;

    output_dev_4d.push_back(&tdev_out);

    //! init concat op
    Shape_v shape_out_v;
    LOG(INFO) << "concat compute output shape";
    concat_dev.compute_output_shape(input_dev_4d, output_dev_4d, param);
    tdev_out.re_alloc(output_dev_4d[0]->shape());
    //    LOG(INFO) << "shape out 4d: " << shape_out[0] << ", " << shape_out[1] << ", " << \
    //              shape_out[2] << ", " << shape_out[3];
    //    LOG(INFO) << "compute shape out 4d: " << shape_out_v[0][0] << ", " << \
    //            shape_out_v[0][1] << ", " << shape_out_v[0][2] << ", " << shape_out_v[0][3];
    //    CHECK_EQ(shape_out_v[0] == shape_out, true) << "compute output shape error";
    LOG(INFO) << "concat initialization";
    SABER_CHECK(concat_dev.init(input_dev_4d, output_dev_4d, param, RUNTIME, VENDER_IMPL, ctx_dev));
    //! following code returns error code
    //SABER_CHECK(concat_dev.init(input_dev_4d, output_dev_4d, param, SPECIFY, VENDER_IMPL, ctx_dev));
    //! declare output tensor, alloc memory


    //! do async memory copy
    Context<NV> ctx2(0, 2, 2);
    tdev1.async_copy_from(th1, ctx2.get_data_stream());
    tdev2.async_copy_from(th2, ctx2.get_compute_stream());

    //! record event and do sync
    tdev1.record_event(ctx2.get_data_stream());
    tdev2.record_event(ctx2.get_compute_stream());
    //! uncomment the following 2 lines may get wrong result
    tdev1.sync();
    tdev2.sync();

    LOG(INFO) << "concat compute";
    concat_dev(input_dev_4d, output_dev_4d, param, ctx_dev);
    output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
    output_dev_4d[0]->sync();
    print_tensor_device(*output_dev_4d[0]);
    cudaDeviceSynchronize();
}

TEST(TestSaberFuncConcatNV, test_concat_tensor_with_roi) {

    Env<NV>::env_init();
    typedef TargetWrapper<NV> API;

    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef std::vector<Shape> Shape_v;
    typedef TensorDf4::Dtype dtype;

    //! runtime context
    Context<NV> ctx_dev(0, 1, 1);

    //! constructor of concat op
    Concat<NV, AK_FLOAT> concat_dev;

    int w_total = 10;
    int h_total = 6;
    int c_total = 4;
    int n_total = 2;
    Shape shape_out_total{n_total, c_total, h_total, w_total};
    Shape off_1{0, 0, 1, 1};

    TensorDf4 tdev_out(shape_out_total);

    int concat_axis = 3; // channel
    int sum_w_in = 8;
    std::vector<int> w_in = {4, 4};
    int h_in = 4;
    std::vector<int> ch_in = {2, 2};
    int sum_ch_in = 4;
    int num_in = 2;

    Shape shape_out(num_in, sum_ch_in, h_in, sum_w_in);

    ConcatParam<TensorDf4> param(concat_axis);

    LOG(INFO) << " input size, num=" << num_in << ", channel=" << \
              ch_in[0] << ", height=" << h_in << ", width=" << w_in[0];
    LOG(INFO) << "input 4 tensor: width = " << w_in[0] << ", " << w_in[1];
    LOG(INFO) << "concat axis= " << param.axis;

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    //! declare input tensors
    TensorDf4 tdev_in1, tdev_in2, tdev1, tdev2;
    TensorHf4 th1, th2;
    //! input shape
    Shape sh1(num_in, sum_ch_in, h_in, w_in[0]);
    Shape sh2(num_in, sum_ch_in, h_in, w_in[1]);
    Shape os1(0, 0, 0, 0);
    Shape os2(0, 0, 0, w_in[0]);
    //! alloc memory and data
    tdev_in1.re_alloc(shape_out);
    tdev_in2.re_alloc(shape_out);
    th1.re_alloc(sh1);
    th2.re_alloc(sh2);

    for (int j = 0; j < th1.size(); ++j) {
        th1.mutable_data()[j] = 1;
    }

    for (int j = 0; j < th2.size(); ++j) {
        th2.mutable_data()[j] = 2;
    }

    //! shared from other tensor
    tdev1.share_sub_buffer(tdev_in1, sh1, os1);
    tdev2.share_sub_buffer(tdev_in2, sh2, os2);
    //! prepare input tensor vector
    input_dev_4d.push_back(&tdev1);
    input_dev_4d.push_back(&tdev2);

    TensorDf4 tdev_out_roi;
    tdev_out_roi.share_sub_buffer(tdev_out, shape_out, off_1);

    output_dev_4d.push_back(&tdev_out_roi);

    //! init concat op
    Shape shape_out_v;
    LOG(INFO) << "concat compute output shape";
    concat_dev.compute_output_shape(input_dev_4d, output_dev_4d, param);
    //tdev_out_roi.re_alloc(output_dev_4d[0]->shape());
    LOG(INFO) << "shape out 4d: " << shape_out[0] << ", " << shape_out[1] << ", " << \
              shape_out[2] << ", " << shape_out[3];
    shape_out_v = output_dev_4d[0]->valid_shape();
    LOG(INFO) << "compute shape out 4d: " << shape_out_v[0] << ", " << \
              shape_out_v[1] << ", " << shape_out_v[2] << ", " << shape_out_v[3];
    CHECK_EQ(shape_out_v == shape_out, true) << "compute output shape error";
    LOG(INFO) << "concat initialization";
    concat_dev.init(input_dev_4d, output_dev_4d, param, RUNTIME, SABER_IMPL, ctx_dev);
    //! declare output tensor, alloc memory


    //! do async memory copy
    Context<NV> ctx2(0, 2, 2);
    tdev1.async_copy_from(th1, ctx2.get_data_stream());
    tdev2.async_copy_from(th2, ctx2.get_compute_stream());

    //! record event and do sync
    tdev1.record_event(ctx2.get_data_stream());
    tdev2.record_event(ctx2.get_compute_stream());
    //! uncomment the following 2 lines may get wrong result
    tdev1.sync();
    tdev2.sync();

    LOG(INFO) << "concat compute";
    concat_dev(input_dev_4d, output_dev_4d, param, ctx_dev);
    output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
    output_dev_4d[0]->sync();
    print_tensor_device(*output_dev_4d[0]);
    print_tensor_device(tdev_out);
    cudaDeviceSynchronize();
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

