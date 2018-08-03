#include "saber/core/context.h"
#include "saber/funcs/scale.h"
#include "test_saber_func.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>

using namespace anakin::saber;

template <typename Tensor>
void print_tensor_shape(std::string name, Tensor& t0) {

    LOG(INFO) << name << " valid shape is ["
              << t0.valid_shape()[0] << ", "
              << t0.valid_shape()[1] << ", "
              << t0.valid_shape()[2] << ", "
              << t0.valid_shape()[3] << "].";

    LOG(INFO) << name << " real shape is ["
              << t0.shape()[0] << ", "
              << t0.shape()[1] << ", "
              << t0.shape()[2] << ", "
              << t0.shape()[3] << "].";

    LOG(INFO) << name << " offset is ["
              << t0.offset()[0] << ", "
              << t0.offset()[1] << ", "
              << t0.offset()[2] << ", "
              << t0.offset()[3] << "].";
}

template <typename Dtype>
void fill_vector_rand(std::vector<Dtype>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = rand() *1.0f/RAND_MAX - 0.5;
    }
}

template<typename Dtype>
void print_vector_data(std::vector<Dtype>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        printf("%d, %f\n", i, vec[i]);
    }
}

static int count(const int start, const int end, int n, int c, int w, int h){
    int _layout[4] = {n, c, w ,h};
    int result = 1;
    for(int i = start; i < end; ++i)
        result *= _layout[i];
    return result;
}

template<typename dtype>
static void test_scale(const dtype* src, const int num_in, const int c_in, const int h_in, const int w_in, \
    dtype* dst, std::vector<dtype>& scale_data, bool bias_term, std::vector<dtype>& bias_data, int axis, int num_axes){
    axis = num_axes == 0 ? 0 : axis;
    num_axes  = num_axes >= 0 ? num_axes : 4 - axis;
    int inner_dim = count(axis + num_axes, 4, num_in, c_in, w_in, h_in);
    int scale_dim = count(axis, axis + num_axes, num_in, c_in, w_in, h_in);
    CHECK_EQ(scale_dim, scale_data.size()) << "scale dim not valid";
    if(scale_dim > 1){
        for(int i = 0; i < num_in * c_in * w_in * h_in; ++i){
            int scale_id = (i / inner_dim) % scale_dim;
            dtype scale = scale_data[scale_id];
            if(bias_term){
                dst[i] = scale * src[i] + bias_data[scale_id];
            }else{
                dst[i] = scale * src[i];
            }
        }
    }else{
        dtype scale = scale_data[0];
        for(int i = 0; i < num_in * c_in * w_in * h_in; ++i){
            if(bias_term){
                dtype bias = bias_data[0];
                dst[i] = scale * src[i] + bias;
            }else{
                dst[i] = scale * src[i];
            }
        }
    }
}

template<typename TargetType, typename TargetType_H>
void test_saber_scale_accurancy(int num_in, int c_in, int h_in, int w_in, int axis, int num_axes, bool bias_term, int scale_dim) {

    typedef Tensor<TargetType_H> TensorHf4;
    typedef Tensor<TargetType> TensorDf4;
    int test_iter = 100;
    
    Shape shape_in({num_in, c_in, w_in, h_in}, Layout_NCHW);
    Shape shape_out({num_in, c_in, w_in, h_in}, Layout_NCHW);
    
    TensorHf4 th, th_saber, th_test;
    TensorDf4 td, td_saber;


    th.re_alloc(shape_in, AK_FLOAT);
    td.re_alloc(shape_in, AK_FLOAT);
    th_saber.re_alloc(shape_out, AK_FLOAT);
    th_test.re_alloc(shape_out, AK_FLOAT);
    td_saber.re_alloc(shape_out, AK_FLOAT);
    fill_tensor_rand(th, 0.0, 1.0);

    td.copy_from(th);
    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;
    
    input_dev_4d.push_back(&td);
    output_dev_4d.push_back(&td_saber);

    std::vector<float> scale_w;
    std::vector<float> scale_b;
    scale_w.resize(scale_dim);
    fill_vector_rand(scale_w);
    if(bias_term){
        scale_b.resize(scale_dim);
        fill_vector_rand(scale_b);
    }
    ScaleParam<TargetType> param(scale_w, scale_b, bias_term, axis, num_axes);
    Context<TargetType> ctx_dev(0, 1, 1);
    LOG(INFO) << "scale initialization...";
    Scale<TargetType, AK_FLOAT> dev_scale;
    SABER_CHECK(dev_scale.init(input_dev_4d, output_dev_4d, param, SPECIFY, SABER_IMPL, ctx_dev));
    
    double max_ratio = 0.0;
    double max_diff = 0.0;
    LOG(INFO) << "100 iters, scale compute...";
    for(int i = 0 ; i < test_iter; ++i){
        //run saber scale
        SABER_CHECK(dev_scale(input_dev_4d, output_dev_4d, param, ctx_dev));
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
        th_saber.copy_from(td_saber);
        //run check scale
        test_scale((const float*)th.data(), num_in, c_in, h_in, w_in, (float*)th_test.mutable_data(), \
                    scale_w, bias_term, scale_b, axis, num_axes);
        tensor_cmp_host((const float*)th_test.data(), (const float*)th_saber.data(), th_test.valid_size(),
                               max_ratio, max_diff);
        CHECK_EQ(max_diff < 0.01, true) << "FAIL!! check result and saber result are not matched, max_diff = " << max_diff;
    }
    LOG(INFO) << "PASS!!";

}


TEST(TestSaberFunc, test_func_sale) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    test_saber_scale_accurancy<NV,X86>(2, 2, 4, 4, 1, 1, true, 2);
    test_saber_scale_accurancy<NV,X86>(2, 2, 4, 4, 1, 1, false, 2);
    test_saber_scale_accurancy<NV,X86>(2, 2, 4, 4, 0, -1, true, 64);
    test_saber_scale_accurancy<NV,X86>(2, 2, 4, 4, 0, -1, false, 64);
    test_saber_scale_accurancy<NV,X86>(2, 2, 4, 4, 0, 0, true, 1);
    test_saber_scale_accurancy<NV,X86>(2, 2, 4, 4, 0, 0, false, 1);
#endif
}


int main(int argc, const char** argv) {
    Env<NV>::env_init();
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

