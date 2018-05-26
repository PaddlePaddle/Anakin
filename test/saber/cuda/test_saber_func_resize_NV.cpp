
#include "core/context.h"
#include "funcs/resize.h"
#include "test_saber_func_resize_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

#include <chrono>

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

using namespace anakin::saber;

TEST(TestSaberFuncResizeNV, test_func_resize_NCHW) {

    typedef TargetWrapper<NV> API;
    typedef TargetWrapper<X86> X86_API;

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef TensorDf4::Dtype dtype;

    int w_in = 32;
    int h_in = 32;
    int ch_in = 512;
    int num_in = 1;
    float scale_w = 2;
    float scale_h = 2;

    int test_iter = 100;

    int w_out = int(floor(w_in * scale_w));
    int h_out = int(floor(h_in * scale_h));

    ResizeParam<TensorDf4> param(scale_w, scale_h);

    LOG(INFO) << "Resize param: ";
    LOG(INFO) << " input size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;
    LOG(INFO) << "resize output size = " << num_in << ", " \
              << ch_in << ", " << w_out << ", " << h_out;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out_4d(num_in, ch_in, h_out, w_out);

    TensorHf4 th1;
    TensorDf4 td1, td2;

    th1.re_alloc(shape_in);
    td1.re_alloc(shape_in);

    for (int i = 0; i < th1.size(); ++i) {
        th1.mutable_data()[i] = static_cast<dtype>(i);
    }

    td1.copy_from(th1);

    // start Reshape & doInfer
    Context<NV> ctx_dev(0, 1, 1);
    Context<X86> ctx_host;

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    input_dev_4d.push_back(&td1);

    Resize<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> dev_resize;

    output_dev_4d.push_back(&td2);

    LOG(INFO) << "resize compute output shape";
    dev_resize.compute_output_shape(input_dev_4d, output_dev_4d, param);

    LOG(INFO) << "shape out 4d: " << shape_out_4d[0] << ", " << shape_out_4d[1] << ", " << \
              shape_out_4d[2] << ", " << shape_out_4d[3];

    CHECK_EQ(shape_out_4d == output_dev_4d[0]->valid_shape(), true) << "error output shape";

    output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape());

    // init assume output tensor has been reshpaed by user.
    LOG(INFO) << "resize initialization";
    SABER_CHECK(dev_resize.init(input_dev_4d, output_dev_4d, param, RUNTIME, SABER_IMPL, ctx_dev));
    SABER_CHECK(dev_resize.init(input_dev_4d, output_dev_4d, param, RUNTIME, SABER_IMPL, ctx_dev));
    SaberTimer<NV> t1;
    LOG(INFO) << "resize compute";
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        dev_resize(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
    }

    t1.end(ctx_dev);
    LOG(INFO) << test_iter << "test, total time: " << t1.get_average_ms() << "avg time : " << \
              t1.get_average_ms() / test_iter;
    //print_tensor_host(th1);
    //print_tensor_device(*output_dev_4d[0]);
    //cudaDeviceSynchronize();
}


#if 0
TEST(TestSaberFuncResizeNV, test_func_resize_NCHW_INT8) {

    typedef TargetWrapper<NV> API;
    typedef TargetWrapper<X86> X86_API;

    typedef Tensor<X86, AK_INT8, NCHW> TensorHc4;
    typedef Tensor<NV, AK_INT8, NCHW> TensorDc4;

    typedef TensorDc4::Dtype dtype;

    int w_in = 32;
    int h_in = 32;
    int ch_in = 512;
    int num_in = 1;
    float scale_w = 2;
    float scale_h = 2;

    int test_iter = 100;

    int w_out = int(floor(w_in * scale_w));
    int h_out = int(floor(h_in * scale_h));

    ResizeParam<TensorDc4> param(scale_w, scale_h);

    LOG(INFO) << "Resize param: ";
    LOG(INFO) << " input size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;
    LOG(INFO) << "resize output size = " << num_in << ", " \
              << ch_in << ", " << w_out << ", " << h_out;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out_4d(num_in, ch_in, h_out, w_out);

    TensorHc4 th1;
    TensorDc4 td1, td2;

    th1.re_alloc(shape_in);
    td1.re_alloc(shape_in);

    for (int i = 0; i < th1.size(); ++i) {
        th1.mutable_data()[i] = static_cast<dtype>(i);
    }

    td1.copy_from(th1);

    // start Reshape & doInfer
    Context<NV> ctx_dev(0, 1, 1);
    Context<X86> ctx_host;

    std::vector<TensorDc4*> input_dev_4d;
    std::vector<TensorDc4*> output_dev_4d;

    input_dev_4d.push_back(&td1);

    Resize<NV, AK_INT8, AK_INT8, AK_INT8, NCHW, NCHW, NCHW> dev_resize;

    output_dev_4d.push_back(&td2);

    LOG(INFO) << "resize compute output shape";
    dev_resize.compute_output_shape(input_dev_4d, output_dev_4d, param);

    LOG(INFO) << "shape out 4d: " << shape_out_4d[0] << ", " << shape_out_4d[1] << ", " << \
              shape_out_4d[2] << ", " << shape_out_4d[3];

    CHECK_EQ(shape_out_4d == output_dev_4d[0]->valid_shape(), true) << "error output shape";

    output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape());

    // init assume output tensor has been reshpaed by user.
    LOG(INFO) << "resize initialization";
    dev_resize.init(input_dev_4d, output_dev_4d, param, RUNTIME, SABER_IMPL, ctx_dev);

    SaberTimer<NV> t1;
    LOG(INFO) << "resize compute";
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        dev_resize(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
    }

    t1.end(ctx_dev);
    LOG(INFO) << test_iter << "test, total time: " << t1.get_average_ms() << "avg time : " << \
              t1.get_average_ms() / test_iter;
    //print_tensor_host(th1);
    //print_tensor_device(*output_dev_4d[0]);
    //cudaDeviceSynchronize();
}
#endif
TEST(TestSaberFuncResizeNV, test_func_resize_HW) {

    typedef TargetWrapper<NV> API;
    typedef TargetWrapper<X86> X86_API;

    typedef Tensor<X86, AK_FLOAT, HW> TensorHf2;
    typedef Tensor<NV, AK_FLOAT, HW> TensorDf2;

    typedef TensorDf2::Dtype dtype;

    int w_in = 10;
    int h_in = 10;

    float scale_w = 2;
    float scale_h = 2;

    int test_iter = 100;

    int w_out = int(floor(w_in * scale_w));
    int h_out = int(floor(h_in * scale_h));

    ResizeParam<TensorDf2> param(scale_w, scale_h);

    LOG(INFO) << "Resize param: ";
    LOG(INFO) << " input size, " << " height=" << h_in << ", width=" << w_in;
    LOG(INFO) << "resize output size = " << h_out << ", " << w_out;

    Shape shape_in(h_in, w_in);
    Shape shape_out_2d(h_out, w_out);

    TensorHf2 th1;
    TensorDf2 td1, td2;

    th1.re_alloc(shape_in);
    td1.re_alloc(shape_in);

    for (int i = 0; i < th1.size(); ++i) {
        th1.mutable_data()[i] = static_cast<dtype>(i);
    }

    td1.copy_from(th1);

    // start Reshape & doInfer
    Context<NV> ctx_dev(0, 1, 1);
    Context<X86> ctx_host;

    std::vector<TensorDf2*> input_dev_2d;
    std::vector<TensorDf2*> output_dev_2d;

    input_dev_2d.push_back(&td1);

    Resize<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, HW, HW, HW> dev_resize;

    output_dev_2d.push_back(&td2);

    LOG(INFO) << "resize compute output shape";
    dev_resize.compute_output_shape(input_dev_2d, output_dev_2d, param);

    LOG(INFO) << "shape out 2d: " << shape_out_2d[0] << ", " << shape_out_2d[1];

    CHECK_EQ(shape_out_2d == output_dev_2d[0]->valid_shape(), true) << "error output shape";

    output_dev_2d[0]->re_alloc(output_dev_2d[0]->valid_shape());

    // init assume output tensor has been reshpaed by user.
    LOG(INFO) << "resize initialization";
    dev_resize.init(input_dev_2d, output_dev_2d, param, RUNTIME, SABER_IMPL, ctx_dev);

    SaberTimer<NV> t1;
    LOG(INFO) << "resize compute";
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        dev_resize(input_dev_2d, output_dev_2d, param, ctx_dev);
        output_dev_2d[0]->sync();
    }

    t1.end(ctx_dev);
    LOG(INFO) << test_iter << "test, total time: " << t1.get_average_ms() << "avg time : " << \
              t1.get_average_ms() / test_iter;
    print_tensor_host(th1);
    print_tensor_device(*output_dev_2d[0]);
    cudaDeviceSynchronize();
}

#ifdef USE_OPENCV
TEST(TestSaberFuncResizeNV, test_resize_image_NHWC) {

    using namespace cv;
    typedef TargetWrapper<NV> API;
    typedef TargetWrapper<X86> X86_API;

    typedef Tensor<X86, AK_FLOAT, NHWC> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NHWC> TensorDf4;

    Context<X86> ctx_host;
    Context<NV> ctx_dev;

    typedef TensorDf4::Dtype dtype;

    int test_iter = 100;

    std::string img_path = "../image/cat1.jpg";
    Mat im = imread(img_path);

    if (im.empty()) {
        LOG(WARNING) << "load image: " << img_path << "failed";
        return;
    }

    Mat imf;
    im.convertTo(imf, CV_32FC3);

    int num_in = 1;
    int ch_in = 3;
    int h_in = im.rows;
    int w_in = im.cols;

    float scale_w = 2;
    float scale_h = 2;

    int w_out = int(floor(w_in * scale_w));
    int h_out = int(floor(h_in * scale_h));

    ResizeParam<TensorDf4> param(scale_w, scale_h);

    LOG(INFO) << "Resize param: ";
    LOG(INFO) << " input size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;
    LOG(INFO) << "resize output size = " << num_in << ", " \
              << ch_in << ", " << w_out << ", " << h_out;

    Shape shape_in(num_in, h_in, w_in, ch_in);
    Shape shape_out_4d(num_in, h_out, w_out, ch_in);

    Mat im_r;
    SaberTimer<X86> t1;
    t1.start(ctx_host);

    //auto ts1 = std::chrono::system_clock::now();
    //double tcv = getTickCount();
    for (int i = 0; i < test_iter; ++i) {
        resize(imf, im_r, Size(0, 0), scale_w, scale_h, INTER_LINEAR);
    }

    t1.end(ctx_host);
    //auto ts2 = std::chrono::system_clock::now();
    //tcv = getTickCount() - tcv;
    //tcv = tcv / getTickFrequency();
    LOG(INFO) << "cv resize, " << test_iter << " test, total time: " << t1.get_average_ms() << \
              ", avg time : " << t1.get_average_ms() / test_iter;

    Mat imr_u8;
    im_r.convertTo(imr_u8, CV_8UC3);
    imwrite("cv_resize.jpg", imr_u8);

    TensorDf4 timg((float*)imf.data, X86(), X86_API::get_device_id(), shape_in);
    Resize<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NHWC, NHWC, NHWC> dev_resize;

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    TensorDf4 td2;
    input_dev_4d.push_back(&timg);
    output_dev_4d.push_back(&td2);

    LOG(INFO) << "resize compute output shape";
    dev_resize.compute_output_shape(input_dev_4d, output_dev_4d, param);

    LOG(INFO) << "shape out 4d: " << shape_out_4d[0] << ", " << shape_out_4d[1] << ", " << \
              shape_out_4d[2] << ", " << shape_out_4d[3];

    CHECK_EQ(shape_out_4d == output_dev_4d[0]->valid_shape(), true) << "error output shape";

    output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape());

    // init assume output tensor has been reshpaed by user.
    LOG(INFO) << "resize initialization";
    dev_resize.init(input_dev_4d, output_dev_4d, param, RUNTIME, SABER_IMPL, ctx_dev);

    SaberTimer<NV> t3;
    LOG(INFO) << "resize compute";
    t3.start(ctx_dev);
    double tv3 = getTickCount();

    for (int i = 0; i < test_iter; ++i) {
        dev_resize(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->sync();
    }

    tv3 = getTickCount() - tv3;
    t3.end(ctx_dev);
    LOG(INFO) << "NV resize, " << test_iter << " test, total time: " << t3.get_average_ms() << \
              ", avg time : " << t3.get_average_ms() / test_iter;
    LOG(INFO) << "time: " << 1000 * tv3 / getTickFrequency() << ", avg: " \
              << 1000 * tv3 / getTickFrequency() / test_iter;

    TensorHf4 th(shape_out_4d);
    th.copy_from(td2);
    Mat imf_t(h_out, w_out, CV_32FC3, (void*)th.mutable_data());
    Mat imt;
    imf_t.convertTo(imt, CV_8UC3);
    imwrite("saber_resize.jpg", imt);

}

TEST(TestSaberFuncResizeNV, test_resize_image_NHWC_WITH_ROI) {

    using namespace cv;
    typedef TargetWrapper<NV> API;
    typedef TargetWrapper<X86> X86_API;

    typedef Tensor<X86, AK_FLOAT, NHWC> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NHWC> TensorDf4;

    Context<X86> ctx_host;
    Context<NV> ctx_dev;

    typedef TensorDf4::Dtype dtype;

    int test_iter = 100;

    std::string img_path = "../image/cat1.jpg";
    Mat im = imread(img_path);

    if (im.empty()) {
        LOG(WARNING) << "load image: " << img_path << " failed";
        return;
    }

    Mat imf;
    im.convertTo(imf, CV_32FC3);

    int num_in = 1;
    int ch_in = 3;
    int h_in = im.rows;
    int w_in = im.cols;

    float scale_w = 2;
    float scale_h = 2;

    int w_out = int(floor(w_in * scale_w));
    int h_out = int(floor(h_in * scale_h));

    ResizeParam<TensorDf4> param(scale_w, scale_h);

    LOG(INFO) << "Resize param: ";
    LOG(INFO) << " input size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;
    LOG(INFO) << "resize output size = " << num_in << ", " \
              << ch_in << ", " << w_out << ", " << h_out;

    Shape shape_in(num_in, h_in, w_in, ch_in);

    TensorDf4 timg((float*)imf.data, X86(), X86_API::get_device_id(), shape_in);
    Resize<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NHWC, NHWC, NHWC> dev_resize;

    Shape shape_in_roi(num_in, h_in / 2, w_in / 2, ch_in);
    Shape shape_out_4d(num_in, h_out, w_out, ch_in);
    Shape shape_out_roi(num_in, h_out / 2, w_out / 2, ch_in);
    TensorDf4 timg_roi;
    timg_roi.share_sub_buffer(timg, shape_in_roi, Shape(0, 0, 0, 0));

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    TensorDf4 td1, td2;
    td1.re_alloc(shape_out_4d);
    td2.share_sub_buffer(td1, shape_out_roi, Shape(0, 0, 0, 0));
    input_dev_4d.push_back(&timg_roi);
    output_dev_4d.push_back(&td2);

    LOG(INFO) << "resize compute output shape";
    dev_resize.compute_output_shape(input_dev_4d, output_dev_4d, param);

    LOG(INFO) << "shape out 4d: " << shape_out_roi[0] << ", " << shape_out_roi[1] << ", " << \
              shape_out_roi[2] << ", " << shape_out_roi[3];

    CHECK_EQ(shape_out_roi == output_dev_4d[0]->valid_shape(), true) << "error output shape";

    output_dev_4d[0]->reshape(output_dev_4d[0]->valid_shape());

    // init assume output tensor has been reshpaed by user.
    LOG(INFO) << "resize initialization";
    dev_resize.init(input_dev_4d, output_dev_4d, param, RUNTIME, SABER_IMPL, ctx_dev);

    SaberTimer<NV> t3;
    LOG(INFO) << "resize compute";
    t3.start(ctx_dev);
    double tv3 = getTickCount();

    for (int i = 0; i < test_iter; ++i) {
        dev_resize(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->sync();
    }

    tv3 = getTickCount() - tv3;
    t3.end(ctx_dev);
    LOG(INFO) << "NV resize, " << test_iter << " test, total time: " << t3.get_average_ms() << \
              ", avg time : " << t3.get_average_ms() / test_iter;
    LOG(INFO) << "time: " << 1000 * tv3 / getTickFrequency() << ", avg: " \
              << 1000 * tv3 / getTickFrequency() / test_iter;

    TensorHf4 th(shape_out_roi);
    th.copy_from(td2);
    Mat imf_t(h_out / 2, w_out / 2, CV_32FC3, (void*)th.mutable_data());
    Mat imt;
    imf_t.convertTo(imt, CV_8UC3);
    imwrite("saber_resize_roi.jpg", imt);

}
#endif

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env<NV>::env_init();
    Env<X86>::env_init();

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

