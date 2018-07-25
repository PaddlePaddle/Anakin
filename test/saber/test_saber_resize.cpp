
#include "saber/core/context.h"
#include "saber/funcs/resize.h"
#include "test_saber_func.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>

#include <chrono>


using namespace anakin::saber;
template<typename dtype>
static void test_resize(const int wout, const int hout,
                        const int num,const int channels,
                        const int win, const int hin,
                        const float scale_w, const float scale_h,
                        const dtype* src, dtype* dst){
    int dst_stride_w = 1, dst_stride_h = wout, dst_stride_c = wout * hout, dst_stride_batch = wout * hout * channels;
    int src_stride_w = 1, src_stride_h = win, src_stride_c = win * hin, src_stride_batch = win * hin * channels;
    for(int n = 0; n < num; ++n){
        for(int c = 0; c < channels; ++c){
            int src_index = n * src_stride_batch + c * src_stride_c;
            for(int h = 0; h < hout; ++h){
                for(int w = 0; w < wout; ++w){
                    float fw = w * scale_w;
                    float fh = h * scale_h;
                    int w_start = (int)fw;
                    int w_end = (int)fw + 1;
                    int h_start = (int)fh;
                    int h_end = (int)fh + 1;
                    fw -= w_start;
                    fh -= h_start;
                    const float w00 = (1.0 - fh) * (1.0 - fw);
                    const float w01 = fw * (1.0 - fh);
                    const float w10 = fh * (1.0 - fw);
                    const float w11 = fw * fh;
                    dtype tl = src[src_index + w_start * src_stride_w + h_start * src_stride_h];
                    dtype tr = w_end > win ? 0 : src[src_index + w_end * src_stride_w + h_start * src_stride_h];
                    dtype bl = h_end > hin ? 0 : src[src_index + w_start * src_stride_w + h_end * src_stride_h];
                    dtype br = (w_end > win) || (h_end > hin) ? 0 : src[src_index + w_end * src_stride_w + h_end * src_stride_h];
                    int dst_index = n * dst_stride_batch + c * dst_stride_c + h * dst_stride_h + w * dst_stride_w;
                    dst[dst_index] = static_cast<dtype>(w00 * tl + w01 * tr + w10 * bl + w11 * br);
                }
            }
        }
    }
   
}
template <typename TargetType, typename TargetType_H>
void test_saber_resize_speed_FLOAT(int num_in, int c_in, int h_in, int w_in) {


    typedef Tensor<TargetType_H> TensorHf4;
    typedef Tensor<TargetType> TensorDf4;


    float scale_w = 2;
    float scale_h = 2;
    int test_iter = 100;

    int w_out = int(floor(w_in * scale_w));
    int h_out = int(floor(h_in * scale_h));

    LOG(INFO) << "AK_FLOAT RESIZE SPEED TEST:";
    ResizeParam<TargetType> param(scale_w, scale_h);
    LOG(INFO) << "resize param: scale_w = " << scale_w << ", scale_h = " << scale_h;
    LOG(INFO) << "input size, num =" << num_in << ", channel =" << \
              c_in << ", height =" << h_in << ", width =" << w_in;
    LOG(INFO) << "resize output size, num = " << num_in << ", channel = " \
              << c_in << ", height = " << h_out << ", width = " << w_out;

    Shape shape_in({num_in, c_in, h_in, w_in}, Layout_NCHW);
    Shape shape_out_4d({num_in, c_in, h_out, w_out}, Layout_NCHW);

    TensorHf4 th1;
    TensorDf4 td1, td2;

    th1.re_alloc(shape_in, AK_FLOAT);
    td1.re_alloc(shape_in, AK_FLOAT);

    for (int i = 0; i < th1.size(); ++i) {
        static_cast<float*>(th1.mutable_data())[i] = float(i);
    }

    td1.copy_from(th1);

    // start Reshape & doInfer
    Context<TargetType> ctx_dev(0, 1, 1);

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    input_dev_4d.push_back(&td1);

    Resize<TargetType, AK_FLOAT> dev_resize;

    output_dev_4d.push_back(&td2);
    LOG(INFO) << "resize compute output shape...";
    dev_resize.compute_output_shape(input_dev_4d, output_dev_4d, param);

    LOG(INFO) << "shape out 4d: " << shape_out_4d[0] << ", " << shape_out_4d[1] << ", " << \
              shape_out_4d[2] << ", " << shape_out_4d[3];

    CHECK_EQ(shape_out_4d == output_dev_4d[0]->valid_shape(), true) << "error output shape";

    output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape(), AK_FLOAT);

    // init assume output tensor has been reshpaed by user.
    LOG(INFO) << "resize initialization...";
    SABER_CHECK(dev_resize.init(input_dev_4d, output_dev_4d, param, SPECIFY, SABER_IMPL, ctx_dev));
    SaberTimer<TargetType> t1;
    LOG(INFO) << "resize compute...";
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        dev_resize(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
    }

    t1.end(ctx_dev);
    LOG(INFO) << test_iter << " iters test, total time: " << t1.get_average_ms() << "ms, avg time : " << \
              t1.get_average_ms() / test_iter << "ms";
    LOG(INFO) << "bandwidth: " << (td2.valid_size() + td1.valid_size()) * type_length(td1.get_dtype())  \
    * test_iter / t1.get_average_ms() / 1024 << "MB/s";
    LOG(INFO) << "PASS!!" << std::endl;
}


template <typename TargetType, typename TargetType_H>
void test_saber_resize_speed_INT8(int num_in, int c_in, int h_in, int w_in) {
    
    
    typedef Tensor<TargetType_H> TensorHf4;
    typedef Tensor<TargetType> TensorDf4;
    
    
    float scale_w = 2;
    float scale_h = 2;
    int test_iter = 100;
    
    int w_out = int(floor(w_in * scale_w));
    int h_out = int(floor(h_in * scale_h));
    
    LOG(INFO) << "AK_INT8 RESIZE SPEED TEST:";
    ResizeParam<TargetType> param(scale_w, scale_h);
    LOG(INFO) << "resize param: scale_w = " << scale_w << ", scale_h = " << scale_h;
    LOG(INFO) << "input size, num =" << num_in << ", channel =" << \
    c_in << ", height =" << h_in << ", width =" << w_in;
    LOG(INFO) << "resize output size, num = " << num_in << ", channel = " \
    << c_in << ", height = " << h_out << ", width = " << w_out;
    
    Shape shape_in({num_in, c_in, h_in, w_in}, Layout_NCHW);
    Shape shape_out_4d({num_in, c_in, h_out, w_out}, Layout_NCHW);
    
    TensorHf4 th1;
    TensorDf4 td1, td2;
    
    th1.re_alloc(shape_in, AK_INT8);
    td1.re_alloc(shape_in, AK_INT8);
    
    for (int i = 0; i < th1.size(); ++i) {
        static_cast<char*>(th1.mutable_data())[i] = char(i);
    }
    
    td1.copy_from(th1);
    
    // start Reshape & doInfer
    Context<TargetType> ctx_dev(0, 1, 1);
    
    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;
    
    input_dev_4d.push_back(&td1);
    
    Resize<TargetType, AK_INT8> dev_resize;
    
    output_dev_4d.push_back(&td2);
    
    LOG(INFO) << "resize compute output shape...";
    dev_resize.compute_output_shape(input_dev_4d, output_dev_4d, param);
    
    LOG(INFO) << "shape out 4d: " << shape_out_4d[0] << ", " << shape_out_4d[1] << ", " << \
    shape_out_4d[2] << ", " << shape_out_4d[3];
    
    CHECK_EQ(shape_out_4d == output_dev_4d[0]->valid_shape(), true) << "error output shape";
    output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape(), AK_INT8);
    
    // init assume output tensor has been reshpaed by user.
    LOG(INFO) << "resize initialization...";
    SABER_CHECK(dev_resize.init(input_dev_4d, output_dev_4d, param, SPECIFY, SABER_IMPL, ctx_dev));
    SaberTimer<TargetType> t1;
    LOG(INFO) << "resize compute...";
    t1.start(ctx_dev);
    
    for (int i = 0; i < test_iter; ++i) {
        dev_resize(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
    }
    
    t1.end(ctx_dev);
    LOG(INFO) << test_iter << " iters test, total time: " << t1.get_average_ms() << "ms, avg time : " << \
    t1.get_average_ms() / test_iter << "ms";
    LOG(INFO) << "bandwidth: " << (td2.valid_size() + td1.valid_size()) * type_length(td1.get_dtype())  \
    * test_iter / t1.get_average_ms() / 1024 << "MB/s";
    LOG(INFO) << "PASS!!"<< std::endl;

}

template <typename TargetType, typename TargetType_H>
void test_saber_resize_speed_2d(int h_in, int w_in) {
    
    
    typedef Tensor<TargetType_H> TensorHf2;
    typedef Tensor<TargetType> TensorDf2;
    
    
    float scale_w = 2;
    float scale_h = 2;
    int test_iter = 100;
    
    int w_out = int(floor(w_in * scale_w));
    int h_out = int(floor(h_in * scale_h));
    
    LOG(INFO) << "2D RESIZE SPEED TEST:";
    ResizeParam<TargetType> param(scale_w, scale_h);
    LOG(INFO) << "resize param: scale_w = " << scale_w << ", scale_h = " << scale_h;
    LOG(INFO) << "input size, height =" << h_in << ", width =" << w_in;
    LOG(INFO) << "resize output size, height = " << h_out << ", width = " << w_out;
    
    Shape shape_in({h_in, w_in}, Layout_HW);
    Shape shape_out_2d({h_out, w_out}, Layout_HW);
    
    TensorHf2 th1;
    TensorDf2 td1, td2;
    
    th1.re_alloc(shape_in, AK_FLOAT);
    td1.re_alloc(shape_in, AK_FLOAT);
    
    for (int i = 0; i < th1.size(); ++i) {
        static_cast<float*>(th1.mutable_data())[i] = float(i);
    }
    
    td1.copy_from(th1);
    
    // start Reshape & doInfer
    Context<TargetType> ctx_dev(0, 1, 1);
    
    std::vector<TensorDf2*> input_dev_2d;
    std::vector<TensorDf2*> output_dev_2d;
    
    input_dev_2d.push_back(&td1);
    
    Resize<TargetType, AK_FLOAT> dev_resize;
    
    output_dev_2d.push_back(&td2);
    LOG(INFO) << "resize compute output shape...";
    dev_resize.compute_output_shape(input_dev_2d, output_dev_2d, param);
    
    LOG(INFO) << "shape out 2d: " << shape_out_2d[0] << ", " << shape_out_2d[1];
    
    CHECK_EQ(shape_out_2d == output_dev_2d[0]->valid_shape(), true) << "error output shape";
    
    output_dev_2d[0]->re_alloc(output_dev_2d[0]->valid_shape(), AK_FLOAT);
    
    // init assume output tensor has been reshpaed by user.
    LOG(INFO) << "resize initialization...";
    SABER_CHECK(dev_resize.init(input_dev_2d, output_dev_2d, param, SPECIFY, SABER_IMPL, ctx_dev));
    SaberTimer<TargetType> t1;
    LOG(INFO) << "resize compute...";
    t1.start(ctx_dev);
    
    for (int i = 0; i < test_iter; ++i) {
        dev_resize(input_dev_2d, output_dev_2d, param, ctx_dev);
        output_dev_2d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_2d[0]->sync();
    }
    
    t1.end(ctx_dev);
    LOG(INFO) << test_iter << " iters test, total time: " << t1.get_average_ms() << "ms, avg time : " << \
    t1.get_average_ms() / test_iter << "ms";
    LOG(INFO) << "bandwidth: " << (td2.valid_size() + td1.valid_size()) * type_length(td1.get_dtype())  \
    * test_iter / t1.get_average_ms() / 1024 << "MB/s";
    LOG(INFO) << "PASS!!" << std::endl;
}

template <typename TargetType, typename TargetType_H>
void test_saber_resize_accurancy(int num_in, int c_in, int h_in, int w_in) {
    
    typedef Tensor<TargetType_H> TensorHf4;
    typedef Tensor<TargetType> TensorDf4;
    
    float scale_w = 2;
    float scale_h = 2;
    int test_iter = 100;
    
    int w_out = int(floor(w_in * scale_w));
    int h_out = int(floor(h_in * scale_h));
    
    LOG(INFO) << "RESIZE ACCURANCY TEST:";
    ResizeParam<TargetType> param(scale_w, scale_h);
    LOG(INFO) << "resize param: scale_w = " << scale_w << ", scale_h = " << scale_h;
    LOG(INFO) << "input size: num = " << num_in << ", channel = " << c_in <<", height = " << h_in << ", width =" << w_in;
    LOG(INFO) << "resize output size: num = " << num_in << ", channel = " << c_in <<", height = " << h_out << ", width =" << w_out;
    
    Shape shape_in({num_in, c_in, h_in, w_in}, Layout_NCHW);
    Shape shape_out({num_in, c_in, h_out, w_out}, Layout_NCHW);
    
    TensorHf4 th, th_saber, th_check;
    TensorDf4 td_in, td_out;
    
    th.re_alloc(shape_in, AK_FLOAT);
    th_saber.re_alloc(shape_out, AK_FLOAT);
    th_check.re_alloc(shape_out, AK_FLOAT);
    td_in.re_alloc(shape_in, AK_FLOAT);
    td_out.re_alloc(shape_out, AK_FLOAT);
    
    fill_tensor_rand(th, 0.0, 1.0);
    td_in.copy_from(th);
    Context<TargetType> ctx_dev(0, 1, 1);
    
    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;
    
    input_dev_4d.push_back(&td_in);
    output_dev_4d.push_back(&td_out);

    LOG(INFO) << "resize initialization...";
    Resize<TargetType, AK_FLOAT> dev_resize;
    SABER_CHECK(dev_resize.init(input_dev_4d, output_dev_4d, param, SPECIFY, SABER_IMPL, ctx_dev));

    double max_ratio = 0.0;
    double max_diff = 0.0;
    LOG(INFO) << "100 iters, resize compute...";
    for (int i = 0; i < test_iter; ++i) {
        //run saber resize
        SABER_CHECK(dev_resize(input_dev_4d, output_dev_4d, param, ctx_dev));
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
        th_saber.copy_from(*output_dev_4d[0]);
        // run check resize
        test_resize<float>(w_out, h_out, num_in, c_in, w_in, h_in, 1/scale_w, 1/scale_h, 
                            (const float*)th.data(), (float*)th_check.mutable_data());
        //compare accurancy
        tensor_cmp_host<float>((const float*)th_check.data(), (const float*)th_saber.data(), th_check.valid_size(),
                        max_ratio, max_diff);
        CHECK_EQ(max_diff < 0.01, true) << "FAIL!! check result and saber result are not matched, max_diff = " << max_diff;
    }
    LOG(INFO) << "PASS!!";
}
	


TEST(TestSaberFunc, test_func_resize){
    int num_in = 2;
    int c_in = 4;
    int h_in = 32;
    int w_in = 64;
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    test_saber_resize_speed_FLOAT<NV, NVHX86>(num_in, c_in, h_in, w_in);
    test_saber_resize_speed_INT8<NV, NVHX86>(num_in, c_in, h_in, w_in);
    test_saber_resize_speed_2d<NV, NVHX86>(h_in, w_in);
    test_saber_resize_accurancy<NV, NVHX86>(num_in, c_in, h_in, w_in);

    
#endif
}
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

