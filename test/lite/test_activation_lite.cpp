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
typedef Tensor<CPU> TensorHf4;

#define COMPARE_RESULT 1

template <typename dtype>
void activation_basic(const TensorHf4& tin, TensorHf4& tout, ActivationParam& param) {

    int num = tin.num();
    int channel = tin.channel();
    int height = tin.height();
    int width = tin.width();

    dtype* dout = (dtype*)tout.mutable_data();
    const dtype* din = (const dtype*)tin.data();
    int count = tin.valid_size();
    int size = height * width;

    switch (param._act_type) {
        //x > 0 ? x : 0
        case Active_relu:
            for (size_t i = 0; i < count; i++) {
                dout[i] = din[i] > 0 ? din[i] : 0;
            }

            break;

            // sigmoid: 1/(exp(-x) + 1)
        case Active_sigmoid:

            for (size_t i = 0; i < count; i++) {
                dout[i] = 1.0f / (exp(-din[i]) + 1.0f);
            }

            break;

            // tanh : (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        case Active_tanh:
            for (size_t i = 0; i < count; i++) {
                dout[i] =  tanh(din[i]);//(exp(din[i]) - exp(-din[i])) / (exp(din[i]) + exp(-din[i]));
            }

            break;

            // stanh : b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}
        case Active_stanh:
            for (size_t i = 0; i < count; i++) {
                dtype val = din[i] * param._neg_slope;
                dout[i] =  param._coef * tanh(val);
            }

            break;

            // x > 0 ? x : 0;
            // x < threshold ? x : threshold
        case Active_clipped_relu:
            for (size_t i = 0; i < count; i++) {
                const dtype threshold = param._coef;
                dout[i] = din[i] > 0 ? (din[i] < threshold ? din[i] : threshold) : 0;
            }

            break;

            //elu:  x > 0 ? x : coef * (exp(x) - 1)
        case Active_elu:
            for (size_t i = 0; i < count; i++) {
                dout[i] =  din[i] > 0 ? din[i] : param._coef * (exp(din[i]) - 1);
            }

            break;


            //prelu: x > 0 ? x : slope[c] * x
        case Active_prelu:
            for (int n = 0; n < num; n++) {
                const dtype* in_ptr = din + n * channel * size;
                dtype* out_ptr = dout + n * channel * size;

                // const dtype *slope_ptr = (const dtype*)prelu_param.slope->data();
                for (int c = 0; c < channel; c++) {
                    const dtype* in_ch_ptr = in_ptr + c * size;
                    dtype* out_ch_ptr = out_ptr + c * size;
                    float slope = param._prelu_channel_shared? param._prelu_weights[0] : \
                                  param._prelu_weights[c];

                    for (int k = 0; k < size; k++) {
                        out_ch_ptr[k] = in_ch_ptr[k] > 0 ? in_ch_ptr[k] : in_ch_ptr[k] * slope;
                    }
                }
            }
            break;
        default:
            LOG(FATAL) << "unsupported activation type: " << param._act_type;
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

    SaberActivation activation_lite;
    float slopes[ch_in];
    for (int i=0; i<ch_in; ++i) {
        slopes[i]=0.1f*i;
    }
    ActivationParam param(active_type, 0.f, 1.0f, true, slopes);
    activation_lite.load_param(&param);

    std::vector<TensorHf4*> vin;
    std::vector<TensorHf4*> vout;

    Tensor<CPU> thin(shape_in);
    fill_tensor_rand(thin, -1.f, 1.f);
    TensorHf4 tout;
    TensorHf4 tout_basic(shape_out);
    vin.push_back(&thin);

#if COMPARE_RESULT
    activation_basic<float>(thin, tout_basic, param);
    //print_tensor_host(tout_basic);
#endif

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

    tensor_cmp_host(tout_basic, tout, max_ratio, max_diff);
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

