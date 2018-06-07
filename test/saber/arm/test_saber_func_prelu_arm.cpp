
#include "saber/funcs/prelu.h"
#include "test_saber_func_test_arm.h"
#include "tensor_op.h"

#define DEFINE_GLOBAL(type, var, value) \
        type (GLB_##var) = (value)
DEFINE_GLOBAL(int, threads, 1);
DEFINE_GLOBAL(int, cluster_id, 0);
DEFINE_GLOBAL(bool, channel_shared, "true");

#define USE_COMPARE

using namespace anakin::saber;

typedef TargetWrapper<ARM> ARM_API;
typedef Tensor<ARM, AK_FLOAT, NCHW> TensorHf4;

void test_arm_prelu(const std::vector<TensorHf4 *>& tin,TensorHf4& slopes, \
    bool channel_shared, int threads, int cluster_id) {

    Context<ARM> ctx1;
    std::vector<int> act_ids;
    //! set runtime context
    LOG(INFO) << "set runtine context";
    std::vector<int> big_cores;
    std::vector<int> small_cores;
    for (int i = 0; i < ctx1.devs[0]._info._cluster_ids.size(); ++i) {
        if (ctx1.devs[0]._info._cluster_ids[i] == 0) {
            big_cores.push_back(ctx1.devs[0]._info._core_ids[i]);
        } else {
            small_cores.push_back(ctx1.devs[0]._info._cluster_ids[i]);
        }
    }

    if (cluster_id == 0) {
        if (big_cores.size() == 0) {
            LOG(FATAL) << "big cores are not supported";
        }
        if (threads > big_cores.size()) {
            LOG(WARNING) << "not enough big cores for inference";
            act_ids = big_cores;
        } else {
            for (int i = 0; i < threads; ++i) {
                act_ids.push_back(big_cores[i]);
            }
        }
    } else {
        if (small_cores.size() == 0) {
            LOG(FATAL) << "small cores are not supported";
        }
        if (threads > small_cores.size()) {
            LOG(WARNING) << "not enough small cores for inference";
            act_ids = small_cores;
        } else {
            for (int i = 0; i < threads; ++i) {
                act_ids.push_back(small_cores[i]);
            }
        }
    }
    ctx1.set_act_cores(act_ids);

    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int threads = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << threads;
#endif
    }
    int th_id = 0;
#pragma omp parallel private(th_id)
    {
#ifdef USE_OPENMP
        th_id = omp_get_thread_num();
#pragma omp parallel
        LOG(INFO) << "thread core ID: " << big_cores[th_id];
#endif
    }

    TensorHf4 tout_basic;
    TensorHf4 tout_saber;
    TensorHf4 tout_ncnn;

    TensorHf4* thin = tin[0];

    std::vector<TensorHf4*> tvout_saber;
    std::vector<TensorHf4*> tvout_basic;
    std::vector<TensorHf4*> tvout_ncnn;

   tvout_saber.push_back(&tout_saber);
   tvout_basic.push_back(&tout_basic);
   tvout_ncnn.push_back(&tout_ncnn);

    int num = tin[0]->num();
    int chin = tin[0]->channel();
    int hin = tin[0]->height();
    int win = tin[0]->width();

    LOG(INFO) << "prelu param: ";
    LOG(INFO) << " img_num = " << num;
    LOG(INFO) << " in_channels = " << chin;
    LOG(INFO) << " img_h = " << hin;
    LOG(INFO) << " img_w = " << win;
   // LOG(INFO) << " group = " << group;
   // LOG(INFO) << " pad = " << pad;
   // LOG(INFO) << " stride = " << stride;
   // LOG(INFO) << " dilation = " << dila;
   // LOG(INFO) << " kernel = " << kernel;
   // LOG(INFO) << " out_channels = " << ch_out;
    LOG(INFO) << " channel_shared : " << channel_shared;

   PreluParam<TensorHf4> prelu_param(channel_shared, &slopes);
   Prelu<ARM, AK_FLOAT> prelu_saber;
   LOG(INFO) << "Prelu compute output shape";
   prelu_saber.compute_output_shape(tin, tvout_saber, prelu_param);
   prelu_saber.compute_output_shape(tin, tvout_basic, prelu_param);
   prelu_saber.compute_output_shape(tin, tvout_ncnn, prelu_param);

   Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape_1: " << sh_out_saber[0] << ", " << sh_out_saber[1] << ", " \
        << sh_out_saber[2] << ", " << sh_out_saber[3];
    tvout_saber[0]->re_alloc(sh_out_saber);
    tvout_basic[0]->re_alloc(sh_out_saber);
    tvout_ncnn[0]->re_alloc(sh_out_saber);
    

#ifdef USE_COMPARE
    LOG(INFO) << "run basic prelu for precision comparation";
    size_t workspace_size = sizeof(float) * num * chin * hin  * win;
    void* work_space_data = fast_malloc(workspace_size);
    //Sgemm gemmer;
    int test_iter = 100;
    double to1 = 0.f;
    double to2 = 0.f;
    double to3 = 0.f;
    double min_time = 1000000;
    SaberTimer<ARM> t1;
    SaberTimer<ARM> t2;
    SaberTimer<ARM> t3;
     for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        //prelu_basic(ctx1,tout_basic, *thin, slopes, channel_shared);
        tvout_basic[0]->record_event(ctx1.get_compute_stream());
        tvout_basic[0]->sync();
        t1.end(ctx1);
        to1 += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    fast_free(work_space_data);
    LOG(INFO) << "basic prelu running time, ave: " << to1 / test_iter << ", min time: " << min_time;
    //print_tensor_host(tin[0]);
    //for(int i = 0; i < tvout_basic.size(); i ++)
    //	print_tensor_host(tvout_basic[i]);
    LOG(INFO) << "Prelu initialization";
    SABER_CHECK(prelu_saber.init(tin, tvout_saber, prelu_param, RUNTIME, SABER_IMPL, ctx1));
    LOG(INFO) << "saber prelu compute";
    min_time = 1000000;
    for (int i = 0; i < test_iter; ++i) {
        t2.clear();
        t2.start(ctx1);
        //printf("i: %d\n",i);
        prelu_saber(tin, tvout_saber, prelu_param, ctx1);
        //prelu_arm(ctx1,tout_saber, *thin, slopes, channel_shared);
        tvout_saber[0]->record_event(ctx1.get_compute_stream());
        tvout_saber[0]->sync();
        t2.end(ctx1);
        to2 += t2.get_average_ms();
        if (t2.get_average_ms() < min_time) {
            min_time = t2.get_average_ms();
        }
       // printf("to2: %.3f\n");
    }
    LOG(INFO) << "saber prelu running time, ave: " << to2 / test_iter << ", min time: " << min_time;
    //print_tensor_host(*tvout_saber[0]);

   LOG(INFO) << "ncnn prelu compute";
    min_time = 1000000;
    for (int i = 0; i < test_iter; ++i) {
        t3.clear();
        t3.start(ctx1);
       // pooling_saber(tin, tvout_saber, pooling_param, ctx1);
        //prelu_ncnn(ctx1,tout_ncnn, *thin, slopes, channel_shared);
        tvout_ncnn[0]->record_event(ctx1.get_compute_stream());
        tvout_ncnn[0]->sync();
        t3.end(ctx1);
        to3 += t3.get_average_ms();
        if (t3.get_average_ms() < min_time) {
            min_time = t3.get_average_ms();
        }
        //printf("to2: %.3f\n");
    }
    LOG(INFO) << "ncnn prelu running time, ave: " << to3 / test_iter << ", min time: " << min_time;
    //print_tensor_host(*tvout_saber[0]);
    
#endif
   
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


TEST(TestSaberFuncTest, test_func_slice_arm) {

    int num = 1;
    int chin = 64;
    int hin = 224;
    int win = 224;


   // bool bias_term = false;
   // bool global = true;
   // PoolingType type = 1;

    Shape shape_in(num, chin, hin, win);

    TensorHf4 tdin;

    tdin.re_alloc(shape_in);
    fill_tensor_host_rand(tdin, -1.f, 1.f);
    //fill_tensor_host_const(tdin, 1.f);

    std::vector<TensorHf4*> tin;
    tin.push_back(&tdin);

    Shape sh_slope{1, 1, 1, chin};
    TensorHf4 tslop(sh_slope);
    for (int i = 0; i < chin; ++i) {
        tslop.mutable_data()[i] = 0.1f * (i + 1);
    }

    
    test_arm_prelu(tin, tslop, GLB_channel_shared, GLB_threads, GLB_cluster_id);
    //LOG(WARNING) << "pooling not support yet";
}
int main(int argc, const char** argv){
	anakin::saber::Env<ARM>::env_init();
    // initial logger
    //logger::init(argv[0]);
     if (argc < 1) {
        LOG(INFO) << "Example of Usage:\n" << \
        "./output/unit_test/pooloing_test type global threads cluster_id";
         return 0;
    } else if (argc == 4){
        GLB_threads = atoi(argv[1]);
        GLB_cluster_id = atoi(argv[2]);
        GLB_channel_shared = std::string(argv[3]).compare("true")==0 ? true : false;
    }
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}