#include "saber/core/context.h"
#include "saber/funcs/roi_pooling.h"
#include "test_saber_func.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>
#include <float.h>

using namespace anakin::saber;

void roi_pooling_cpu(const float* input, const float* roi, int num_in, int c_in, int h_in, int w_in,
                     int roi_num, int pool_h, int pool_w, float spatial_scale, float* output){
    int num_out = roi_num;
    int c_out = c_in;
    int h_out = pool_h;
    int w_out = pool_w;
    int in_stride_num = c_in * h_in * w_in;
    int in_stride_c = h_in * w_in;
    int in_stride_h = w_in;
    int in_stride_w = 1;
    int out_stride_num = c_out * h_out * w_out;
    int out_stride_c = h_out * w_out;
    int out_stride_h = w_out;
    int out_stride_w = 1;
    for(int n = 0; n < num_out; ++n){
        int in_index_n = roi[n * 5] * in_stride_num;
        int in_w_start = round(roi[n * 5 + 1] * spatial_scale);
        int in_h_start = round(roi[n * 5 + 2] * spatial_scale);
        int in_w_end = round(roi[n * 5 + 3] * spatial_scale);
        int in_h_end = round(roi[n * 5 + 4] * spatial_scale);
        float roi_rate_w = (float)(in_w_end - in_w_start + 1) / w_out;
        float roi_rate_h = (float)(in_h_end - in_h_start + 1) / h_out;
        for(int c = 0; c < c_out; ++c){
            int in_index = in_index_n + c * in_stride_c;
            for(int h = 0; h < h_out; ++h){
                for(int w = 0; w < w_out; ++w){
                    int w_start = floor(w * roi_rate_w) + in_w_start;
                    int h_start = floor(h * roi_rate_h) + in_h_start;
                    int w_end = ceil((w+1) * roi_rate_w) + in_w_start;
                    int h_end = ceil((h+1) * roi_rate_h) + in_h_start;
                    w_end = w_end > w_in ? w_in : w_end;
                    h_end = h_end > h_in ? h_in : h_end;
                    int out_index = n * out_stride_num + c * out_stride_c + h * out_stride_h + w * out_stride_w;
                    bool is_empty = (h_start >= h_end) || (w_start >= w_end);

                    float max = is_empty ? 0.0f : -FLT_MAX;
                    for(int j = h_start; j < h_end; ++j){
                        for(int i = w_start; i < w_end; ++i){
                            float data_in = input[in_index + i * in_stride_w + j * in_stride_h];
                            if(data_in > max)
                                max = data_in;
                        }
                    }
                    output[out_index] = max;
                }
            }
        }
    }
    
    
}

template<typename TargetType, typename TargetType_H>
void test_saber_roi_pooling_accurancy(int num_in, int c_in, int h_in, int w_in,
                                      int roi_num, int pool_h, int pool_w, int spatial_scale){
    typedef Tensor<TargetType> TensorDf4;
    typedef Tensor<TargetType_H> TensorHf4;
    
    Shape in_shape({num_in, c_in, h_in, w_in}, Layout_NCHW);
    Shape roi_shape({roi_num, 5, 1, 1}, Layout_NCHW);
    Shape out_shape({roi_num, c_in, pool_h, pool_w}, Layout_NCHW);
    
    TensorHf4 th_in, th_roi, th_saber, th_cpu;
    TensorDf4 td_in, td_roi, td_out;
    
    th_in.re_alloc(in_shape, AK_FLOAT);
    th_roi.re_alloc(roi_shape, AK_FLOAT);
    th_saber.re_alloc(out_shape, AK_FLOAT);
    th_cpu.re_alloc(out_shape, AK_FLOAT);
    td_in.re_alloc(in_shape, AK_FLOAT);
    td_roi.re_alloc(roi_shape, AK_FLOAT);
    td_out.re_alloc(out_shape, AK_FLOAT);
    
    LOG(INFO) << "AK_FLOAT ROI POOLING ACCURANCY TEST:";
    LOG(INFO) << "roi pooling param: pool_h = " << pool_h << ", pool_w = " << pool_w << ", spatial_scale = " << spatial_scale;
    LOG(INFO) << "input size, num =" << num_in << ", channel =" << \
    c_in << ", height =" << h_in << ", width =" << w_in;
    LOG(INFO) << "roi pooling output size, num = " << roi_num << ", channel = " \
    << c_in << ", height = " << pool_h << ", width = " << pool_w;
    // prepare host data
    fill_tensor_rand(th_in, 0.0, 1.0);
    // prepare roi data
    float* roi_data = (float*)th_roi.mutable_data();
    srand(time(0));
    for(int i = 0; i < roi_num; ++i){
        roi_data[i * 5] = rand() % num_in;
        roi_data[i * 5 + 1] = floor(rand() % (w_in/2) / spatial_scale);
        roi_data[i * 5 + 2] = floor(rand() % (h_in/2) / spatial_scale);
        roi_data[i * 5 + 3] = floor((rand() % (w_in/2) + w_in/2) / spatial_scale);
        roi_data[i * 5 + 4] = floor((rand() % (h_in/2) + h_in/2) / spatial_scale);
    }
    
    // copy data to device
    td_in.copy_from(th_in);
    td_roi.copy_from(th_roi);

    //construct input vector
    std::vector<TensorDf4*> input_dev;
    std::vector<TensorDf4*> output_dev;
    
    input_dev.push_back(&td_in);
    input_dev.push_back(&td_roi);
    output_dev.push_back(&td_out);
    
    //consturct roipooling Op
    RoiPool<TargetType, AK_FLOAT>dev_roi_pool;
    RoiPoolParam<TargetType> param(pool_h, pool_w, spatial_scale);
    Context<TargetType> ctx_dev(0, 1, 1);
    LOG(INFO) << "roi pooling compute output shape...";
    dev_roi_pool.compute_output_shape(input_dev, output_dev, param);
    CHECK_EQ(output_dev[0]->shape() == out_shape, true) << "error :compute roi pooling output shape";
    LOG(INFO) << "shape out: " << out_shape[0] << ", " << out_shape[1] << ", " << \
    out_shape[2] << ", " << out_shape[3];
    LOG(INFO) << "roi pooling initialization...";
    SABER_CHECK(dev_roi_pool.init(input_dev, output_dev, param, SPECIFY, SABER_IMPL, ctx_dev));
    
    //warm up
    for(int i = 0; i < 10; ++i)
        SABER_CHECK(dev_roi_pool(input_dev, output_dev, param, ctx_dev));
    
    int test_iter = 100;
    double max_ratio = 0.0;
    double max_diff = 0.0;
    LOG(INFO) << test_iter << " iters, roi pooling compute...";
    for(int iter = 0; iter < test_iter; ++iter){
        //run saber roipooling
        SABER_CHECK(dev_roi_pool(input_dev, output_dev, param, ctx_dev));
        output_dev[0]->record_event(ctx_dev.get_compute_stream());
        output_dev[0]->sync();
        th_saber.copy_from(*output_dev[0]);
        //run cpu roipooling
        roi_pooling_cpu((const float*)th_in.data(), (const float*)th_roi.data(),
                        num_in, c_in, h_in, w_in, roi_num,  pool_h, pool_w, spatial_scale,
                        (float*)th_cpu.mutable_data());
        //check result
        tensor_cmp_host<float>((const float*)th_cpu.data(), (const float*)th_saber.data(), th_saber.valid_size(), max_ratio, max_diff);
        CHECK_EQ(max_diff < 0.0001, true) << "FAIL!! cpu result and saber result are not matched, max_diff = " << max_diff;
    }
    LOG(INFO) << "PASS!!"<<std::endl;

}

template<typename TargetType, typename TargetType_H>
void test_saber_roi_pooling_speed(int num_in, int c_in, int h_in, int w_in,
                                      int roi_num, int pool_h, int pool_w, int spatial_scale){
    typedef Tensor<TargetType> TensorDf4;
    typedef Tensor<TargetType_H> TensorHf4;
    
    Shape in_shape({num_in, c_in, h_in, w_in}, Layout_NCHW);
    Shape roi_shape({roi_num, 5, 1, 1}, Layout_NCHW);
    Shape out_shape({roi_num, c_in, pool_h, pool_w}, Layout_NCHW);
    
    TensorHf4 th_in, th_roi, th_saber, th_cpu;
    TensorDf4 td_in, td_roi, td_out;
    
    th_in.re_alloc(in_shape, AK_FLOAT);
    th_roi.re_alloc(roi_shape, AK_FLOAT);
    th_saber.re_alloc(out_shape, AK_FLOAT);
    th_cpu.re_alloc(out_shape, AK_FLOAT);
    td_in.re_alloc(in_shape, AK_FLOAT);
    td_roi.re_alloc(roi_shape, AK_FLOAT);
    td_out.re_alloc(out_shape, AK_FLOAT);
    
    LOG(INFO) << "AK_FLOAT ROI POOLING SPEED TEST:";
    LOG(INFO) << "roi pooling param: pool_h = " << pool_h << ", pool_w = " << pool_w << ", spatial_scale = " << spatial_scale;
    LOG(INFO) << "input size, num =" << num_in << ", channel =" << \
    c_in << ", height =" << h_in << ", width =" << w_in;
    LOG(INFO) << "roi pooling output size, num = " << roi_num << ", channel = " \
    << c_in << ", height = " << pool_h << ", width = " << pool_w;
    // prepare host data
    fill_tensor_rand(th_in, 0.0, 1.0);
    // prepare roi data
    float* roi_data = (float*)th_roi.mutable_data();
    srand(time(0));
    for(int i = 0; i < roi_num; ++i){
        roi_data[i * 5] = rand() % num_in;
        roi_data[i * 5 + 1] = floor(rand() % (w_in/2) / spatial_scale);
        roi_data[i * 5 + 2] = floor(rand() % (h_in/2) / spatial_scale);
        roi_data[i * 5 + 3] = floor((rand() % (w_in/2) + w_in/2) / spatial_scale);
        roi_data[i * 5 + 4] = floor((rand() % (h_in/2) + h_in/2) / spatial_scale);
    }
    
    // copy data to device
    td_in.copy_from(th_in);
    td_roi.copy_from(th_roi);
    
    //construct input vector
    std::vector<TensorDf4*> input_dev;
    std::vector<TensorDf4*> output_dev;
    
    input_dev.push_back(&td_in);
    input_dev.push_back(&td_roi);
    output_dev.push_back(&td_out);
    
    //consturct roipooling Op
    RoiPool<TargetType, AK_FLOAT>dev_roi_pool;
    RoiPoolParam<TargetType> param(pool_h, pool_w, spatial_scale);
    Context<TargetType> ctx_dev(0, 1, 1);
    LOG(INFO) << "roi pooling compute output shape...";
    dev_roi_pool.compute_output_shape(input_dev, output_dev, param);
    CHECK_EQ(output_dev[0]->shape() == out_shape, true) << "error :compute roi pooling output shape";
    LOG(INFO) << "shape out: " << out_shape[0] << ", " << out_shape[1] << ", " << \
    out_shape[2] << ", " << out_shape[3];
    LOG(INFO) << "roi pooling initialization...";
    SABER_CHECK(dev_roi_pool.init(input_dev, output_dev, param, SPECIFY, SABER_IMPL, ctx_dev));
    
    //warm up
    for(int i = 0; i < 10; ++i)
        SABER_CHECK(dev_roi_pool(input_dev, output_dev, param, ctx_dev));
    
    int test_iter = 100;
    double max_ratio = 0.0;
    double max_diff = 0.0;
    SaberTimer<TargetType> t1;
    t1.start(ctx_dev);
    LOG(INFO) << test_iter << " iters, roi pooling compute...";
    for(int iter = 0; iter < test_iter; ++iter){
        //run saber roipooling
        SABER_CHECK(dev_roi_pool(input_dev, output_dev, param, ctx_dev));
        output_dev[0]->record_event(ctx_dev.get_compute_stream());
        output_dev[0]->sync();
    }
    t1.end(ctx_dev);
    LOG(INFO) << test_iter << " iters test, total time: " << t1.get_average_ms() << "ms, avg time : " << \
    t1.get_average_ms() / test_iter << "ms";
    LOG(INFO) << "PASS!!"<<std::endl;
    
}

TEST(TestSaberFunc, test_func_roi_pooling){
    int num_in = 4;
    int c_in = 3;
    int h_in = 32;
    int w_in = 64;
    int roi_num = 3;
    int pool_h = 4;
    int pool_w = 4;
    float spatial_scale = 2;
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    test_saber_roi_pooling_speed<NV, NVHX86>(num_in, c_in, h_in, w_in, roi_num, pool_h, pool_w, spatial_scale);
    test_saber_roi_pooling_accurancy<NV, NVHX86>(num_in, c_in, h_in, w_in, roi_num, pool_h, pool_w, spatial_scale);
    
#endif
}
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

