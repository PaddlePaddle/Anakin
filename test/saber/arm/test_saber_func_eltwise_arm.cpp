#include "saber/funcs/eltwise.h"
#include "test_saber_func_test_arm.h"
#include "tensor_op.h"

#define DEFINE_GLOBAL(type, var, value) \
        type (GLB_##var) = (value)
DEFINE_GLOBAL(int, threads, 1);
DEFINE_GLOBAL(int, cluster_id, 0);
DEFINE_GLOBAL(int, operation, 1);
DEFINE_GLOBAL(int, num_coeff, 0);
#define USE_COMPARE

using namespace anakin::saber;

typedef TargetWrapper<ARM> ARM_API;
typedef Tensor<ARM, AK_FLOAT, NCHW> TensorHf4;

void eltwise_basic(const Context<ARM> &ctx, Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, \ 
    std::vector<Tensor<ARM, AK_FLOAT, NCHW>*> &tensor_in,\
    int op_type, std::vector<float> coeffs_ptr, int num_coeff) {
    CHECK_GT(tensor_out.size(), 0) << "output tensor is empty";
    CHECK_GT(tensor_in.size(), 1) << "input tensor is empty";

    int w_in = tensor_in[0]->width();
    int h_in = tensor_in[0]->height();
    int ch_in = tensor_in[0]->channel();
    int num = tensor_in[0]->num();
    int size_in = w_in * h_in;

    float* data_out = tensor_out.mutable_data();
    const float* data_in0 = tensor_in[0]->data();
    const float* data_in1 = tensor_in[1]->data();
    
    if (op_type == 1){ //Operation_PROD
        for (int n = 0; n < num; n++){
            float* data_out_batch = data_out + n * ch_in * size_in;
            const float* data_in0_batch = data_in0 + n * ch_in * size_in;
            const float* data_in1_batch = data_in1 + n * ch_in * size_in;

#pragma omp parallel for
            for (int c = 0; c < ch_in; c++){
                float* data_out_channel = data_out_batch + c * size_in;
                const float* data_in0_channel = data_in0_batch + c * size_in;
                const float* data_in1_channel = data_in1_batch + c * size_in;
                for (int i = 0; i < size_in; i++){
                    data_out_channel[i] = data_in0_channel[i] * data_in1_channel[i];
                }
            }
        }
        for (int b = 2; b <tensor_in.size(); b++){
            const float* data_in = tensor_in[b]->data();
            for (int n = 0; n < num; n++){
                float* data_out_batch = data_out + n * ch_in * size_in;
                const float* data_in_batch = data_in + n * ch_in * size_in;

#pragma omp parallel for
                for (int c = 0; c < ch_in; c++){
                    float* data_out_channel = data_out_batch + c * size_in;
                    const float* data_in_channel = data_in_batch + c * size_in;
                    for (int i = 0; i < size_in; i++){
                        data_out_channel[i] = data_out_channel[i] * data_in_channel[i];
                    }
                }
            }
        }
    }
    if (op_type == 2){ //Operation_SUM
        if (num_coeff == 0){
            for (int n = 0; n < num; n++){
                float* data_out_batch = data_out + n * ch_in * size_in;
                const float* data_in0_batch = data_in0 + n * ch_in * size_in;
                const float* data_in1_batch = data_in1 + n * ch_in * size_in;

#pragma omp parallel for
                for (int c = 0; c < ch_in; c++){
                    float* data_out_channel = data_out_batch + c * size_in;
                    const float* data_in0_channel = data_in0_batch + c * size_in;
                    const float* data_in1_channel = data_in1_batch + c * size_in;
                    for (int i = 0; i < size_in; i++){
                        data_out_channel[i] = data_in0_channel[i] + data_in1_channel[i];
                    }
                }
            }
            for (int b = 2; b <tensor_in.size(); b++){
                const float* data_in = tensor_in[b]->data();
                for (int n = 0; n < num; n++){
                    float* data_out_batch = data_out + n * ch_in * size_in;
                    const float* data_in_batch = data_in + n * ch_in * size_in;

#pragma omp parallel for
                    for (int c = 0; c < ch_in; c++){
                        float* data_out_channel = data_out_batch + c * size_in;
                        const float* data_in_channel = data_in_batch + c * size_in;
                        for (int i = 0; i < size_in; i++){
                            data_out_channel[i] = data_out_channel[i] + data_in_channel[i];
                        }
                    }
                }
            }
        }else{
            for (int n = 0; n < num; n++){
                float* data_out_batch = data_out + n * ch_in * size_in;
                const float* data_in0_batch = data_in0 + n * ch_in * size_in;
                const float* data_in1_batch = data_in1 + n * ch_in * size_in;

#pragma omp parallel for
                for (int c = 0; c < ch_in; c++){
                    float* data_out_channel = data_out_batch + c * size_in;
                    const float* data_in0_channel = data_in0_batch + c * size_in;
                    const float* data_in1_channel = data_in1_batch + c * size_in;
                    for (int i = 0; i < size_in; i++){
                        data_out_channel[i] = data_in0_channel[i]*coeffs_ptr[0] + \ 
                        data_in1_channel[i]*coeffs_ptr[1];
                    }
                }
            }
            for (int b = 2; b <tensor_in.size(); b++){
                const float* data_in = tensor_in[b]->data();
                for (int n = 0; n < num; n++){
                    float* data_out_batch = data_out + n * ch_in * size_in;
                    const float* data_in_batch = data_in + n * ch_in * size_in;

#pragma omp parallel for
                    for (int c = 0; c < ch_in; c++){
                        float* data_out_channel = data_out_batch + c * size_in;
                        const float* data_in_channel = data_in_batch + c * size_in;
                        for (int i = 0; i < size_in; i++){
                            data_out_channel[i] = data_out_channel[i] + \ 
                            data_in_channel[i] * coeffs_ptr[b];
                        }
                    }
                }
            }
        }
    }
    if (op_type == 3){ //Operation_MAX
        for (int n = 0; n < num; n++){
            float* data_out_batch = data_out + n * ch_in * size_in;
            const float* data_in0_batch = data_in0 + n * ch_in * size_in;
            const float* data_in1_batch = data_in1 + n * ch_in * size_in;

#pragma omp parallel for
            for (int c = 0; c < ch_in; c++){
                float* data_out_channel = data_out_batch + c * size_in;
                const float* data_in0_channel = data_in0_batch + c * size_in;
                const float* data_in1_channel = data_in1_batch + c * size_in;
                for (int i = 0; i < size_in; i++){
                    data_out_channel[i] = std::max(data_in0_channel[i], data_in1_channel[i]);
                }
            }
        }
        for (int b = 2; b <tensor_in.size(); b++){
            const float* data_in = tensor_in[b]->data();
            for (int n = 0; n < num; n++){
                float* data_out_batch = data_out + n * ch_in * size_in;
                const float* data_in_batch = data_in + n * ch_in * size_in;

#pragma omp parallel for
                for (int c = 0; c < ch_in; c++){
                    float* data_out_channel = data_out_batch + c * size_in;
                    const float* data_in_channel = data_in_batch + c * size_in;
                    for (int i = 0; i < size_in; i++){
                        data_out_channel[i] = std::max(data_out_channel[i], data_in_channel[i]);
                    }
                }
            }
        }
    }
    
}


void test_arm_eltwise(std::vector<TensorHf4*>& tin, EltwiseType operation, \
     std::vector<float> coeffs_ptr, int num_coeff, int threads, int cluster_id) {

    //printf("operation: %d \n", operation);
    int test_iter = 100;
    double to = 0;
    double min_time = 1000000;
    SaberTimer<ARM> t1;
    SaberTimer<ARM> t2;

    Context<ARM> ctx1;
    Context<ARM> ctx2;
    PowerMode mode = cluster_id == 0? SABER_POWER_HIGH : SABER_POWER_LOW;
    ctx1.set_run_mode(mode, threads);
            LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
                LOG(INFO) << "number of threads: " << thread;
#endif
    }

    TensorHf4 tout_basic;
    TensorHf4 tout_saber;

    //TensorHf4* thin = tin[0];

    std::vector<TensorHf4*> tvout_saber;
    std::vector<TensorHf4*> tvout_basic;

    tvout_saber.push_back(&tout_saber);
    tvout_basic.push_back(&tout_basic);

    int numin = tin[0]->num();
    int chin = tin[0]->channel();
    int hin = tin[0]->height();
    int win = tin[0]->width();
    int pad = 0;

    LOG(INFO) << "eltwise param: ";
    LOG(INFO) << " img_num = " << numin;
    LOG(INFO) << " in_channels = " << chin;
    LOG(INFO) << " img_h = " << hin;
    LOG(INFO) << " img_w = " << win;
    enum { Eltwise_prod = 1, Eltwise_sum = 2, Eltwise_max = 3 };
    if (operation == 1)
        LOG(INFO) << " operation = " << Eltwise_prod;
    if (operation == 2)
        LOG(INFO) << " operation = " << Eltwise_sum;
    if (operation == 3)
        LOG(INFO) << " operation = " << Eltwise_max;

    int input_dim = 1;
    Shape shape_out = tin[0]->valid_shape();
    for (int i = 0; i < 4; i++){
    	shape_out[i] = tin[0]->valid_shape()[i];
    }
   //Shape shape_out{num, ch_out, h_out, w_out}

#ifdef USE_COMPARE

/*
    LOG(INFO) << "initial input tensor data 0:";
    print_tensor_host(*tin[0]);
    LOG(INFO) << "initial input tensor data 1:";
    print_tensor_host(*tin[1]);
*/
    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];

    LOG(INFO) << "run basic eltwise for precision comparation";
    tout_basic.re_alloc(shape_out);
    size_t workspace_size = sizeof(float) * numin * chin * (hin + 2 * pad) * (win + 2 * pad);
    void* work_space_data = fast_malloc(workspace_size);
   
    to = 0;
     for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        eltwise_basic(ctx1, tout_basic, tin, operation, coeffs_ptr, num_coeff);
        
        tvout_basic[0] ->record_event(ctx1.get_compute_stream());
        tvout_basic[0] ->sync();
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    fast_free(work_space_data);
    LOG(INFO) << "basic eltwise running time, ave: " << to / test_iter << ", min time: " << min_time;
   // print_tensor_host(tout_basic);

    
  /*
  LOG(INFO) << "run ncnn eltwise for precision comparation";
    TensorHf4 tout_basic2;
    tout_basic2.re_alloc(shape_out);
     to = 0;
     for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);

        //eltwise_ncnn(ctx1, tout_basic2, tin, operation, coeffs_ptr, num_coeff);

        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "ncnn eltwise running time, ave: " << to/test_iter << ", min time: " << min_time;
    double max_ratio1 = 0;
    double max_diff1 = 0;
    tensor_cmp_host(tout_basic.data(), tout_basic2.data(), tout_basic.valid_size(), max_ratio1, max_diff1);
   // LOG(INFO) << "tout_basic";
   // print_tensor_host(tout_basic);
  // LOG(INFO) << "tout_saber";
   // print_tensor_host(tout_saber);
    LOG(INFO) << "compare result, max diff: " << max_diff1 << ", max ratio: " << max_ratio1;
    //CHECK_EQ(fabsf(max_ratio1) < 1e-5f, true) << "compute result error";
*/
#endif
    
    Eltwise<ARM, AK_FLOAT> eltwise_saber;
    EltwiseParam<TensorHf4> eltwise_param(operation, coeffs_ptr);

    eltwise_saber.compute_output_shape(tin, tvout_saber, eltwise_param);

    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape_1: " << sh_out_saber[0] << ", " << sh_out_saber[1] << ", " \
        << sh_out_saber[2] << ", " << sh_out_saber[3];
    //LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);

    LOG(INFO) << "saber eltwise impl init";
    SABER_CHECK(eltwise_saber.init(tin, tvout_saber, eltwise_param, SPECIFY, SABER_IMPL, ctx2));

    //! compute
    LOG(INFO) << "saber eltwise compute";
    to = 0;
    min_time = 1000000;
    for (int i = 0; i < test_iter; ++i) {
        t2.clear();
        t2.start(ctx2);
       // pooling_saber(tin, tvout_saber, pooling_param, ctx1);
        //eltwise_arm(ctx2, tout_saber, tin, operation, coeffs_ptr, num_coeff);
        eltwise_saber(tin, tvout_saber, eltwise_param, ctx2);
        tvout_saber[0]->record_event(ctx2.get_compute_stream());
        tvout_saber[0]->sync();
        t2.end(ctx2);
        //printf("i: %d \n",i);
        to += t2.get_average_ms();
        if (t2.get_average_ms() < min_time) {
            min_time = t2.get_average_ms();
        }
    }
    LOG(INFO) << "saber eltwise running time, ave: " << to / test_iter << ", min time: " << min_time;
   // print_tensor_host(tout_saber);
    //print_tensor_host(*tvout_saber[0]);

#ifdef USE_COMPARE
    double max_ratio = 0;
    double max_diff = 0;
    //TensorHf4 tdiff(tout_basic.valid_shape());
    //tensor_diff(tout_basic, tout_saber, tdiff);
    //print_tensor_host(tdiff);
    tensor_cmp_host(tout_basic.data(), tout_saber.data(), tout_basic.valid_size(), max_ratio, max_diff);
   // LOG(INFO) << "tout_basic";
   // print_tensor_host(tout_basic);
  // LOG(INFO) << "tout_saber";
   // print_tensor_host(tout_saber);
    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
#endif
}

#if 1
TEST(TestSaberFuncTest, test_func_eltwise_arm) {

    int num = 1;
    int chin = 32;
    int hin = 112;
    int win = 112;

    int pad = 1;
    int stride = 2;
    int kernel = 3;
    //int chout = 3;

   // bool bias_term = false;
   // bool global = true;
   // PoolingType type = 1;

    Shape shape_in(num, chin, hin, win);

    
    //fill_tensor_host_const(tdin, 1.f);

    std::vector<TensorHf4*> tin;
    TensorHf4 tdin;
    tdin.re_alloc(shape_in);
    fill_tensor_host_rand(tdin, -1.f, 1.f);
    TensorHf4 tdin1;
    tdin1.re_alloc(shape_in);
    fill_tensor_host_rand(tdin1, -1.f, 1.f);
    
    tin.push_back(&tdin);
    tin.push_back(&tdin1);
    
    
    std::vector<float> coeffs_ptr;
   
	coeffs_ptr.push_back(1.0f);
	coeffs_ptr.push_back(-1.0f);
    //printf("test_arm_eltwise: GLB_operation: %d \n", GLB_operation);
    test_arm_eltwise(tin, (EltwiseType)GLB_operation, coeffs_ptr, GLB_num_coeff, GLB_threads, GLB_cluster_id);
    //LOG(WARNING) << "pooling not support yet";
}
#endif

int main(int argc, const char** argv){
    anakin::saber::Env<ARM>::env_init();

    // initial logger
    //logger::init(argv[0]);
   // printf("Test0:\n");
     if (argc < 1) {
        LOG(INFO) << "Example of Usage:\n \
        ./output/unit_test/eltwise_test\n \
            threads\n \
            cluster_id\n \
            operation\n \
            num_coeff\n  ";
        exit(0);
    } else if (argc == 3){
        GLB_threads = atoi(argv[1]);
        GLB_cluster_id = atoi(argv[2]);
    }else if (argc == 5){
        GLB_threads = atoi(argv[1]);
        GLB_cluster_id = atoi(argv[2]);
        GLB_operation = atoi(argv[3]);
        GLB_num_coeff = atoi(argv[4]);
    }
    //printf("Test:\n");
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}


