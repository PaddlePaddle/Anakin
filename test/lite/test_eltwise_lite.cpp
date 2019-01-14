#include "test_lite.h"
#include "saber/lite/funcs/saber_eltwise.h"
#include "saber/saber_types.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;
int test_iter = 10;

int num_in = 9;
int ch_in = 9;
int w_in = 9;
int h_in = 9;
int cluster = 0;
int threads = 4;
int elt_type = 2;
DataType Dtype = AK_FLOAT;
typedef Tensor<CPU> TensorHf4;

#define COMPARE_RESULT 1


template<typename dtype>
void eltwise_basic(const Context &ctx, TensorHf4& tensor_out, \
    std::vector<TensorHf4*> &tensor_in, int op_type, std::vector<float> coeffs_ptr, int num_coeff) {
    CHECK_GT(tensor_out.size(), 0) << "output tensor is empty";
    CHECK_GT(tensor_in.size(), 1) << "input tensor is empty";

    int w_in = tensor_in[0]->width();
    int h_in = tensor_in[0]->height();
    int ch_in = tensor_in[0]->channel();
    int num = tensor_in[0]->num();
    int size_in = w_in * h_in;

    dtype* data_out = static_cast<dtype*>(tensor_out.mutable_data());
    const dtype* data_in0 = static_cast<const dtype*>(tensor_in[0]->data());
    const dtype* data_in1 = static_cast<const dtype*>(tensor_in[1]->data());

    if (op_type == 1){ //Operation_PROD
        for (int n = 0; n < num; n++){
            dtype* data_out_batch = data_out + n * ch_in * size_in;
            const dtype* data_in0_batch = data_in0 + n * ch_in * size_in;
            const dtype* data_in1_batch = data_in1 + n * ch_in * size_in;

#pragma omp parallel for
            for (int c = 0; c < ch_in; c++){
                dtype* data_out_channel = data_out_batch + c * size_in;
                const dtype* data_in0_channel = data_in0_batch + c * size_in;
                const dtype* data_in1_channel = data_in1_batch + c * size_in;
                for (int i = 0; i < size_in; i++){
                    data_out_channel[i] = data_in0_channel[i] * data_in1_channel[i];
                }
            }
        }
        for (int b = 2; b <tensor_in.size(); b++){
            const dtype* data_in = static_cast<const dtype*>(tensor_in[b]->data());
            for (int n = 0; n < num; n++){
                dtype* data_out_batch = data_out + n * ch_in * size_in;
                const dtype* data_in_batch = data_in + n * ch_in * size_in;

#pragma omp parallel for
                for (int c = 0; c < ch_in; c++){
                    dtype* data_out_channel = data_out_batch + c * size_in;
                    const dtype* data_in_channel = data_in_batch + c * size_in;
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
                dtype* data_out_batch = data_out + n * ch_in * size_in;
                const dtype* data_in0_batch = data_in0 + n * ch_in * size_in;
                const dtype* data_in1_batch = data_in1 + n * ch_in * size_in;

#pragma omp parallel for
                for (int c = 0; c < ch_in; c++){
                    dtype* data_out_channel = data_out_batch + c * size_in;
                    const dtype* data_in0_channel = data_in0_batch + c * size_in;
                    const dtype* data_in1_channel = data_in1_batch + c * size_in;
                    for (int i = 0; i < size_in; i++){
                        data_out_channel[i] = data_in0_channel[i] + data_in1_channel[i];
                    }
                }
            }
            for (int b = 2; b <tensor_in.size(); b++){
                const dtype* data_in = static_cast<const dtype*>(tensor_in[b]->data());
                for (int n = 0; n < num; n++){
                    dtype* data_out_batch = data_out + n * ch_in * size_in;
                    const dtype* data_in_batch = data_in + n * ch_in * size_in;

#pragma omp parallel for
                    for (int c = 0; c < ch_in; c++){
                        dtype* data_out_channel = data_out_batch + c * size_in;
                        const dtype* data_in_channel = data_in_batch + c * size_in;
                        for (int i = 0; i < size_in; i++){
                            data_out_channel[i] = data_out_channel[i] + data_in_channel[i];
                        }
                    }
                }
            }
        }else{
            for (int n = 0; n < num; n++){
                dtype* data_out_batch = data_out + n * ch_in * size_in;
                const dtype* data_in0_batch = data_in0 + n * ch_in * size_in;
                const dtype* data_in1_batch = data_in1 + n * ch_in * size_in;

#pragma omp parallel for
                for (int c = 0; c < ch_in; c++){
                    dtype* data_out_channel = data_out_batch + c * size_in;
                    const dtype* data_in0_channel = data_in0_batch + c * size_in;
                    const dtype* data_in1_channel = data_in1_batch + c * size_in;
                    for (int i = 0; i < size_in; i++){
                        data_out_channel[i] = data_in0_channel[i] * coeffs_ptr[0] + \
                        data_in1_channel[i] * coeffs_ptr[1];
                    }
                }
            }
            for (int b = 2; b <tensor_in.size(); b++){
                const dtype* data_in = static_cast<const dtype*>(tensor_in[b]->data());
                for (int n = 0; n < num; n++){
                    dtype* data_out_batch = data_out + n * ch_in * size_in;
                    const dtype* data_in_batch = data_in + n * ch_in * size_in;

#pragma omp parallel for
                    for (int c = 0; c < ch_in; c++){
                        dtype* data_out_channel = data_out_batch + c * size_in;
                        const dtype* data_in_channel = data_in_batch + c * size_in;
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
            dtype* data_out_batch = data_out + n * ch_in * size_in;
            const dtype* data_in0_batch = data_in0 + n * ch_in * size_in;
            const dtype* data_in1_batch = data_in1 + n * ch_in * size_in;

#pragma omp parallel for
            for (int c = 0; c < ch_in; c++){
                dtype* data_out_channel = data_out_batch + c * size_in;
                const dtype* data_in0_channel = data_in0_batch + c * size_in;
                const dtype* data_in1_channel = data_in1_batch + c * size_in;
                for (int i = 0; i < size_in; i++){
                    data_out_channel[i] = std::max(data_in0_channel[i], data_in1_channel[i]);
                }
            }
        }
        for (int b = 2; b <tensor_in.size(); b++){
            const dtype* data_in = static_cast<const dtype*>(tensor_in[b]->data());
            for (int n = 0; n < num; n++){
                dtype* data_out_batch = data_out + n * ch_in * size_in;
                const dtype* data_in_batch = data_in + n * ch_in * size_in;

#pragma omp parallel for
                for (int c = 0; c < ch_in; c++){
                    dtype* data_out_channel = data_out_batch + c * size_in;
                    const dtype* data_in_channel = data_in_batch + c * size_in;
                    for (int i = 0; i < size_in; i++){
                        data_out_channel[i] = std::max(data_out_channel[i], data_in_channel[i]);
                    }
                }
            }
        }
    }
}

void test_eltwise(DataType datatype, std::vector<TensorHf4*>& tin, int operation, \
     std::vector<float> coeffs_ptr, int num_coeff, int threads, int cluster_id) {

   // int test_iter = 100;
    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;
    SaberTimer t2;
    // start Reshape & doInfer
    Context ctx1;
    LOG(INFO) << "set runtime context";
    PowerMode mode = cluster == 0? SABER_POWER_HIGH : SABER_POWER_LOW;
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
    if (Dtype == AK_FLOAT){
        tout_basic.set_dtype(AK_FLOAT);
        tout_saber.set_dtype(AK_FLOAT);
    } else if (Dtype == AK_INT8) {
        tout_basic.set_dtype(AK_INT8);
        tout_saber.set_dtype(AK_INT8);
    }
    //TensorHf4* thin = tin[0];

    std::vector<TensorHf4*> tvout_saber;
    std::vector<TensorHf4*> tvout_basic;

    tvout_saber.push_back(&tout_saber);
    tvout_basic.push_back(&tout_basic);

    int numin = tin[0]->num();
    int chin = tin[0]->channel();
    int hin = tin[0]->height();
    int win = tin[0]->width();

    LOG(INFO) << "eltwise param: ";
    LOG(INFO) << " img_num = " << numin;
    LOG(INFO) << " in_channels = " << chin;
    LOG(INFO) << " img_h = " << hin;
    LOG(INFO) << " img_w = " << win;
   // enum { Eltwise_prod = 1, Eltwise_sum = 2, Eltwise_max = 3 };
    // LOG(INFO) << "operation: " << operation;
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

#ifdef COMPARE_RESULT

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

    to = 0;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        if (datatype == AK_FLOAT){
            eltwise_basic<float>(ctx1, tout_basic, tin, operation, coeffs_ptr, num_coeff);
        }
        else if (datatype == AK_INT8){
            eltwise_basic<signed char>(ctx1, tout_basic, tin, operation, coeffs_ptr, num_coeff);
        }
        //tvout_basic[0] ->record_event(ctx1.get_compute_stream());
        //tvout_basic[0] ->sync();
        t1.end();
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "basic eltwise running time, ave: " << to / test_iter << ", min time: " << min_time;
   // print_tensor_host(tout_basic);
#endif

    SaberEltwise eltwise_saber;
    EltwiseParam eltwise_param((EltwiseType)operation, coeffs_ptr);
   // ParamBase* base =new EltwiseActParam(operation, coeffs_ptr, act_type, 0.f, 1.f, false, tslop.data());
    LOG(INFO) << "saber eltwise load param";
    eltwise_saber.load_param(&eltwise_param);
    //LITE_CHECK(eltwise_act_saber.load_param(&eltwise_act_param));
    if (Dtype == AK_FLOAT){
        eltwise_saber.set_op_precision(AK_FLOAT);
    } else if (Dtype == AK_INT8){
        eltwise_saber.set_op_precision(AK_INT8);
    }
    LOG(INFO) << "saber eltwise compute output shape";
    eltwise_saber.compute_output_shape(tin, tvout_saber);

    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape_1: " << sh_out_saber[0] << ", " << sh_out_saber[1] << ", " \
        << sh_out_saber[2] << ", " << sh_out_saber[3];
    //LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);

    LOG(INFO) << "saber eltwise impl init";
    CHECK_EQ(eltwise_saber.init(tin, tvout_saber, ctx1), SaberSuccess) << "init error";
    //SABER_CHECK(eltwise_act_saber.init(tin, tvout_saber, eltwise_act_param, SPECIFY, SABER_IMPL, ctx1));

    //! compute
    LOG(INFO) << "saber eltwise compute";
    to = 0;
    min_time = 1000000;
    for (int i = 0; i < test_iter; ++i) {
        t2.clear();
        t2.start();
        //eltwise_arm(ctx2, tout_saber, tin, operation, coeffs_ptr, num_coeff);
        //eltwise_act_saber(tin, tvout_saber, eltwise_act_param, ctx1);
        eltwise_saber.dispatch(tin, tvout_saber);
       // tvout_saber[0]->record_event(ctx1.get_compute_stream());
       // tvout_saber[0]->sync();
        t2.end();
        //printf("i: %d \n",i);
        to += t2.get_average_ms();
        if (t2.get_average_ms() < min_time) {
            min_time = t2.get_average_ms();
        }
    }
    LOG(INFO) << "saber eltwise running time, ave: " << to / test_iter << ", min time: " << min_time;
   // print_tensor_host(tout_saber);
    //print_tensor_host(*tvout_saber[0]);

#ifdef COMPARE_RESULT
    double max_ratio = 0;
    double max_diff = 0;
    //TensorHf4 tdiff(tout_basic.valid_shape());
    //tensor_diff(tout_basic, tout_saber, tdiff);
    //print_tensor_host(tdiff);
    tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
   // LOG(INFO) << "tout_basic";
   // print_tensor_host(tout_basic);
  // LOG(INFO) << "tout_saber";
   // print_tensor_host(tout_saber);
    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
#endif
}

#if 1
TEST(TestSaberLite, test_func_eltwise_lite) {

    int num = num_in;
    int chin = ch_in;
    int hin = h_in;
    int win = w_in;

   // bool bias_term = false;
   // bool global = true;
   // PoolingType type = 1;

    Shape shape_in(num, chin, hin, win);

    //fill_tensor_host_const(tdin, 1.f);

    std::vector<TensorHf4*> tin;
    TensorHf4 tdin;
    tdin.re_alloc(shape_in, Dtype);
    TensorHf4 tdin1;
    tdin1.re_alloc(shape_in, Dtype);
    if (Dtype == AK_FLOAT){
        tdin.set_dtype(AK_FLOAT);
        fill_tensor_rand(tdin, -1.f, 1.f);
        fill_tensor_rand(tdin1, -1.f, 1.f);
    } else if (Dtype == AK_INT8){
        tdin.set_dtype(AK_INT8);
        for (int i = 0; i < tdin.valid_size(); ++i){
            static_cast<signed char*>(tdin.mutable_data())[i] = i % 126 - 63;
            static_cast<signed char*>(tdin1.mutable_data())[i] = i % 126 - 63;
        }
    }

    tin.push_back(&tdin);
    tin.push_back(&tdin1);

    std::vector<float> coeffs_ptr;

    coeffs_ptr.push_back(1.0f);
    coeffs_ptr.push_back(1.0f);
    //printf("test_arm_eltwise: GLB_operation: %d \n", GLB_operation);
    // LOG(INFO) << "elt_type: " << elt_type;
    test_eltwise(Dtype, tin, elt_type, coeffs_ptr, 0, threads, cluster);
    //LOG(WARNING) << "pooling not support yet";
}
#endif

int main(int argc, const char** argv){

    Env::env_init();

    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }
    if (argc >= 4){
        test_iter = atoi(argv[3]);
    }
    if (argc >= 5 ) {
        elt_type = atoi(argv[4]);
    }
    if (argc >= 6){
        Dtype = (DataType)atoi(argv[5]);
    }
    if (argc >= 7) {
        if (argc < 10) {
            LOG(ERROR) << "usage: ./" << argv[0] << " cluster  threads  test_iter " << \
                " elt_type datatype num ch_in h_in w_in";
            return 0;
        }
        num_in = atoi(argv[6]);
        ch_in = atoi(argv[7]);
        h_in = atoi(argv[8]);
        w_in = atoi(argv[9]);
    }
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
