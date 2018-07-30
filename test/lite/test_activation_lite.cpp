#include "test_lite.h"
#include "saber/lite/funcs/saber_activation.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;
int test_iter = 10;

int w_in = 9;
int h_in = 9;
int ch_in = 9;
int num_in = 9;
int cluster = 0;
int threads = 4;
ActiveType active_type=Active_relu;
typedef Tensor<CPU, AK_FLOAT> TensorHf4;

#define COMPARE_RESULT 1

void activation_basic(TensorHf4& tin,ActiveType active_type,TensorHf4& tout,bool shared =false , float* slopes =nullptr) {
    switch (active_type) {
        case Active_relu:
            for (int i = 0; i < tin.size(); ++i) {
                tout.mutable_data()[i] = (tin.data()[i] >= 0) ? tin.data()[i] : 0;
            }
            break;
        case Active_tanh:
            for (int i = 0; i < tin.size(); ++i) {
                tout.mutable_data()[i] = (exp(tin.data()[i])-exp(-tin.data()[i]))/(exp(tin.data()[i])+exp(-tin.data()[i]));
            }
            break;
        case Active_sigmoid:
            for (int i = 0; i < tin.size(); ++i) {
                tout.mutable_data()[i] = 1/(1+exp(-tin.data()[i]));
            }
            break;
        case Active_prelu:
            for (int i = 0; i < num_in; ++i) {
                for (int j=0; j<ch_in; ++j) {
                    float slope = shared ? slopes[0] : slopes[j];
                    for(int k=0;k<w_in*h_in;++k){
                        int offset=i*ch_in*w_in*h_in+j*w_in*h_in+k;
                        if(tin.data()[offset]<0.f){
                            tout.mutable_data()[offset] = tin.data()[offset]*slope;
                        }else{
                            tout.mutable_data()[offset] = tin.data()[offset];
                        }
                    }
                }
            }
            break;
        default:
            LOG(ERROR)<<"error activation type!";
            break;
    }
}
TEST(TestSaberLite, test_func_activation_arm) {
    // start Reshape & doInfer
    Context ctx1;
    LOG(INFO) << "set runtine context";
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

   

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out = shape_in;

    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
        ch_in << ", height=" << h_in << ", width=" << w_in;

    std::vector<TensorHf4*> vin;
    std::vector<TensorHf4*> vout;

    Tensor<CPU, AK_FLOAT> thin(shape_in);
    fill_tensor_rand(thin, -1.f, 1.f);
    TensorHf4 tout;
    TensorHf4 tout_basic(shape_out);
    vin.push_back(&thin);
    SaberActivation activation_lite;
    float slopes[ch_in];
    for (int i=0; i<ch_in; ++i) {
        slopes[i]=0.1f*i;
    }
#if COMPARE_RESULT
    activation_basic(thin, active_type, tout_basic,true,slopes);
    //print_tensor_host(tout_basic);
#endif
    
    
    ActivationParam param(active_type, 0.f, 1.0f,true,slopes);
    activation_lite.load_param(&param);

    vout.push_back(&tout);
    activation_lite.compute_output_shape(vin, vout);
    CHECK_EQ(shape_out == vout[0]->valid_shape(), true) << "compute shape error";

    LOG(INFO) << "re-alloc tensor buffer";
    vout[0]->re_alloc(vout[0]->valid_shape());

    LOG(INFO) << "activation initialized to saber impl";
    activation_lite.init(vin, vout, ctx1);

    SaberTimer t1;

    LOG(INFO) << "saber activation compute";
    double to = 0;
    double min_time = 100000;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        activation_lite.dispatch(vin, vout);
        t1.end();
        double tdiff = t1.get_average_ms();
        to += tdiff;
        if (tdiff < min_time) {
            min_time = tdiff;
        }
    }

    printf("saber activation total time : %.4f, avg time : %.4f\n", to, to / test_iter, min_time);
#if COMPARE_RESULT
    double max_ratio = 0;
    double max_diff = 0;
  
    tensor_cmp_host(tout_basic.data(), tout.data(), tout_basic.valid_size(), max_ratio, max_diff);
    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
#endif
}

int main(int argc, const char** argv){
    // initial logger
    //logger::init(argv[0]);
    Env::env_init();

    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }
    if (argc ==4 ) {
        LOG(INFO)<<argv[3];
        if(strcmp(argv[3],"relu")==0)
            active_type= Active_relu;
        else if(strcmp(argv[3],"tanh")==0)
            active_type= Active_tanh;
        else if(strcmp(argv[3],"sigmoid")==0)
            active_type= Active_sigmoid;
        else if(strcmp(argv[3],"prelu")==0)
            active_type= Active_prelu;
        else
            active_type=Active_unknow;
    }
    if (argc> 4 || argc < 2){
        LOG(ERROR)<<"please use "<<argv[0]<<"[cluster] [threads] [active_type]";
    }
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

