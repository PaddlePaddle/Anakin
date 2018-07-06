#include <vector>
#include <cmath>
#include "test_saber_func_activation_arm.h"
#include "saber/funcs/activation.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/core/context.h"
#include "timer.h"

using namespace anakin::saber;

typedef TargetWrapper<ARM> ARM_API;
typedef Tensor<ARM, AK_FLOAT, NCHW> Tensor4f;
typedef Tensor<ARM, AK_FLOAT, HW> Tensor2f;
typedef Tensor<ARM, AK_FLOAT, W> Tensor1f;
ActiveType active_type=Active_relu;
int threads=1;
int test_iter=10;
std::vector<int> act_ids;
bool channel_shared = false;
Tensor4f slopes;
void test(int n, int c, int h, int w, bool channel_shared, Tensor4f slopes) {
    int num_in = n;
    int ch_in = c;
    int h_in = h;
    int w_in = w;

    // LOG(INFO) << " input num:" << num_in << ", channel:" << ch_in << ", height:" << h_in << ", width:" << w_in;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out(num_in, ch_in, h_in, w_in);
    Tensor4f src_in, dst_saber, dst_ref;
    src_in.re_alloc(shape_in);
    fill_tensor_host_rand(src_in,-0.1f,0.1f);
    dst_ref.re_alloc(shape_out);
    switch (active_type) {
        case Active_relu:
            for (int i = 0; i < dst_ref.size(); ++i) {
                dst_ref.mutable_data()[i] = (src_in.data()[i] >= 0) ? src_in.data()[i] : 0;
            }
            break;
        case Active_tanh:
            for (int i = 0; i < dst_ref.size(); ++i) {
                dst_ref.mutable_data()[i] = (exp(src_in.data()[i])-exp(-src_in.data()[i]))/(exp(src_in.data()[i])+exp(-src_in.data()[i]));
            }
            break;
        case Active_sigmoid:
            for (int i = 0; i < dst_ref.size(); ++i) {
                dst_ref.mutable_data()[i] = 1/(1+exp(-src_in.data()[i]));
            }
            break;
        case Active_prelu:
            for (int i = 0; i < dst_ref.size(); ++i) {
                if(channel_shared)
                    dst_ref.mutable_data()[i] = src_in.data()[i] < 0.f ?  src_in.data()[i]* slopes.data()[0]: src_in.data()[i];
                else{
                    int cin = i /(h_in*w_in);
                    dst_ref.mutable_data()[i] = src_in.data()[i] < 0.f ?  src_in.data()[i]* slopes.data()[cin]: src_in.data()[i];
                }
            }
            break;
        default:
            LOG(ERROR)<<"error activation type!";
            break;
    }
    
    Context<ARM> ctx_host;
    std::vector<Tensor4f*> inputs;
    std::vector<Tensor4f*> outputs;
    ctx_host.set_run_mode(SABER_POWER_FULL, threads);
    inputs.push_back(&src_in);
    dst_saber.re_alloc(shape_out);
    outputs.push_back(&dst_saber);

    PreluParam<Tensor4f> prelu_param(channel_shared, &slopes);
    ActivationParam<Tensor4f> param_host(active_type, 0.f, 1.0f, prelu_param);

    Activation<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> op;
    
    op.init(inputs, outputs, param_host, SPECIFY, SABER_IMPL, ctx_host);
    
    
    SaberTimer<ARM> my_time;
    LOG(INFO) << "run";
    double to = 0;
    double tmin = 1000000;
    double tmax = 0;
    my_time.start(ctx_host);
    SaberTimer<ARM> t1;
    for (int i = 0; i < test_iter; i++) {
        t1.clear();
        t1.start(ctx_host);
        op(inputs, outputs, param_host, ctx_host);
        t1.end(ctx_host);
        double tdiff = t1.get_average_ms();
        if (tdiff > tmax) {
            tmax = tdiff;
        }
        if (tdiff < tmin) {
            tmin = tdiff;
        }
        to += tdiff;
    }
    my_time.end(ctx_host);
    
    LOG(INFO) <<" average time " << to / test_iter << \
    ", min time: " << tmin << "ms, max time: " << tmax << " ms";

    bool pass = compare_tensor<Tensor4f>(dst_ref, dst_saber, 1e-6);
    if (pass) {
        LOG(INFO) << "Test Passed";
    }
    else {
        LOG(ERROR) << "Test Failed";
    }
}


TEST(TestSaberActivationARM, test_tensor_activation) {
    Shape shw{1, 128, 1, 1};
    Tensor4f slopes(shw);

    if(channel_shared){
        fill_tensor_host_const(slopes,0.6f);
    }else{
        fill_tensor_host_rand(slopes, -1.f, 1.f);
    }
    LOG(INFO) << "case 1:";
    test(1, 1, 1, 1024, channel_shared, slopes);
    LOG(INFO) << "case 2:";
    test(1, 32, 112, 112, channel_shared, slopes);
    LOG(INFO) << "case 3:";
    test(1, 128,128, 128, channel_shared, slopes);
}

int main(int argc, const char** argv) {
    Env<ARM>::env_init();
    if (argc >=2 ) {
        LOG(INFO)<<argv[1];
        if(strcmp(argv[1],"relu")==0)
            active_type= Active_relu;
        else if(strcmp(argv[1],"tanh")==0)
            active_type= Active_tanh;
        else if(strcmp(argv[1],"sigmoid")==0)
            active_type= Active_sigmoid;
        else if(strcmp(argv[1],"prelu")==0)
            active_type= Active_prelu;
        else
            active_type=Active_unknow;
    }
    if (argc >=3 ) {
        threads=atoi(argv[2]);
        LOG(INFO)<<"threads:"<<threads;
    }
    if(argc == 4){
        channel_shared = atoi(argv[3]) == 0 ? false: true;
    }
    if (argc  == 5) {
        test_iter = atoi(argv[4]);
    }
    
    if (argc> 5 || argc < 2){
        LOG(ERROR)<<"please use ./"<<argv[0]<<"[activation_tpye] [threads] [test_iter] [channel_shared]";
    }
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

