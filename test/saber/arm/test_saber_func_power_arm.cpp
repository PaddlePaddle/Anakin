#include <vector>
#include "saber/funcs/power.h"
#include "saber/funcs/impl/impl_power.h"
#include "test_saber_func_test_arm.h"
#include "saber/funcs/impl/arm/saber_power.h"
#include "saber/core/tensor_op.h"
#include "timer.h"
using namespace anakin::saber;

typedef TargetWrapper<ARM> ARM_API;
typedef Tensor<ARM, AK_FLOAT, NCHW> Tensor4f;
typedef Tensor<ARM, AK_FLOAT, HW> Tensor2f;
typedef Tensor<ARM, AK_FLOAT, W> Tensor1f;
int threads=1;
int test_iter=10;
float scale=1.0f;
float shift=2.0f;
float power=0.6f;
void power_basic(Tensor4f& tin,Tensor4f& tout,float scale,float shift,float power) {
    float* ptr_out = tout.mutable_data();
    const float* ptr_in = tin.data();
    for(int i=0;i<tin.valid_size();++i){
        ptr_out[i]=std::pow((ptr_in[i]*scale+shift),power);
    }
}

TEST(TestSaberFuncTest, test_func_power) {
    std::vector<int> act_ids;
    
    int w_in = 512;
    int h_in = 32;
    int ch_in =1;
    int num_in =1;
        
    
    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out(num_in, ch_in, h_in, w_in);
    Tensor4f src_in, dst_saber, dst_ref;
    src_in.re_alloc(shape_in);
    fill_tensor_host_rand(src_in,2.f,2.f);
    dst_ref.re_alloc(shape_out);
    power_basic(src_in,dst_ref,scale,shift ,power );
    
        
    Context<ARM> ctx_host;
    std::vector<Tensor4f*> inputs;
    std::vector<Tensor4f*> outputs;
    ctx_host.set_run_mode(SABER_POWER_FULL, threads);
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }
    inputs.push_back(&src_in);
    dst_saber.re_alloc(shape_out);
    outputs.push_back(&dst_saber);
    fill_tensor_host_rand(dst_saber,0.f,2.f);

    PowerParam<Tensor4f> param_host(power,scale,shift);
    
    Power<ARM,AK_FLOAT> op;
    
    SABER_CHECK(op.init(inputs, outputs, param_host,SPECIFY, SABER_IMPL,ctx_host));
        
        
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
        op(inputs, outputs, param_host,ctx_host);
        outputs[0]->record_event(ctx_host.get_compute_stream());
        outputs[0]->sync();
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
   
    double max_ratio = 0;
    double max_diff = 0;
    tensor_cmp_host(dst_ref.data(), dst_saber.data(), dst_saber.valid_size(), max_ratio, max_diff);
    
    LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
    

}
int main(int argc, const char** argv) {
    Env<ARM>::env_init();
   

    if (argc >= 2) {
        threads = atoi(argv[1]);
    }

    logger::init(argv[0]);
   InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

