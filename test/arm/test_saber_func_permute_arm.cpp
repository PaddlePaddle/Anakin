#include "saber/funcs/permute.h"
#include "test_saber_func_test_arm.h"
#include "tensor_op.h"

#define DEFINE_GLOBAL(type, var, value) \
        type (GLB_##var) = (value)
DEFINE_GLOBAL(int, threads, 1);
DEFINE_GLOBAL(int, cluster_id, 0);
DEFINE_GLOBAL(int, num, 0);
DEFINE_GLOBAL(int, ch, 1);
DEFINE_GLOBAL(int, h, 2);
DEFINE_GLOBAL(int, w, 3);
#define USE_COMPARE

using namespace anakin::saber;

typedef TargetWrapper<ARM> ARM_API;
typedef Tensor<ARM, AK_FLOAT, NCHW> TensorHf4;

// global_pooling test
void test_arm_permute(std::vector<TensorHf4*>& tin, int num_axes, \
     std::vector<int> permute, int threads, int cluster_id) {
	
    int test_iter = 100;
    double to = 0;
    double min_time = 1000000;
    SaberTimer<ARM> t1;
    SaberTimer<ARM> t2;

    Context<ARM> ctx1;
    Context<ARM> ctx2;
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

    TensorHf4* thin = tin[0];

    std::vector<TensorHf4*> tvout_saber;
    std::vector<TensorHf4*> tvout_basic;

    tvout_saber.push_back(&tout_saber);
    tvout_basic.push_back(&tout_basic);

    int numin = tin[0]->num();
    int chin = tin[0]->channel();
    int hin = tin[0]->height();
    int win = tin[0]->width();
    int pad = 0;

    LOG(INFO) << "permute param: ";
    LOG(INFO) << " img_num = " << numin;
    LOG(INFO) << " in_channels = " << chin;
    LOG(INFO) << " img_h = " << hin;
    LOG(INFO) << " img_w = " << win;

    int input_dim = 1;
    Shape shape_out = tin[0]->valid_shape();
    for (int i = 0; i < num_axes; i ++){
    	shape_out[i] = tin[0]->valid_shape()[permute[i]];
    }
   //Shape shape_out{num, ch_out, h_out, w_out}

#ifdef USE_COMPARE

   // LOG(INFO) << "initial input tensor data:";
   // print_tensor_host(*thin);

    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];

    LOG(INFO) << "run basic permute for precision comparation";
    tout_basic.re_alloc(shape_out);
    size_t workspace_size = sizeof(float) * numin * chin * (hin + 2 * pad) * (win + 2 * pad);
    void* work_space_data = fast_malloc(workspace_size);
    int numout = tvout_basic[0]->num();
    int chout = tvout_basic[0]->channel();
    int hout = tvout_basic[0]->height();
    int wout = tvout_basic[0]->width(); 
    std::vector<int> new_steps;
    std::vector<int> old_steps;

    old_steps.push_back(chin * hin * win);
    old_steps.push_back(hin * win);
    old_steps.push_back(win);
    old_steps.push_back(1);
    
    new_steps.push_back(chout * hout * wout);
    new_steps.push_back(hout * wout);
    new_steps.push_back(wout);
    new_steps.push_back(1);

    bool need_permute = false;
    for (int i = 0; i < num_axes; i ++){
    	if (permute[i] != i){
    		need_permute = true;
    		break;
    	}
    }
    int num_ones = 0;
    if (chin == 1)num_ones ++;
    if (win == 1)num_ones ++;
    if (hin == 1)num_ones ++;
        // order_type
        // 0 = c h w 1 2 3
        // 1 = h w c 2 3 1
        // 2 = w c h 3 1 2
        // 3 = c w h 1 3 2
        // 4 = h c w 2 1 3
        // 5 = w h c 3 2 1
    bool transpose = false;
    int order_type = 0;
    if (permute[1] == 1){
        if (permute[2] == 2){
            order_type = 0;
        }else//3
            order_type = 3;
    }else if (permute[1] == 2){
        if (permute[2] == 1){
            order_type = 4;
        }else//3
            order_type = 1;
    }else if (permute[1] == 3){
        if (permute[2] == 2){
            order_type = 5;
        }else//1{
            order_type = 2;
    }
    if (need_permute){
            if (num_ones > 1)
                need_permute = false;
            else if (num_ones == 1){
                //panduan nchw
                if (chin == 1){
                    if (order_type == 0 || order_type == 1 || order_type == 4)
                        transpose = false;
                    else
                        transpose = true;
                }
                if (hin == 1){
                    if (order_type == 0 || order_type == 3 || order_type == 4)
                        transpose = false;
                    else
                        transpose = true;
                }
                if (win == 1){
                    if (order_type == 0 || order_type == 2 || order_type == 3)
                        transpose = false;
                    else
                        transpose = true;
                }
            }
        }
    int count = tout_basic.valid_size();
    //Sgemm gemmer;
    to = 0;
     for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        //permute_basic(ctx1,tout_basic, *thin, need_permute,num_axes, \
              count, new_steps, old_steps, permute);
        
        tvout_basic[0] ->record_event(ctx1.get_compute_stream());
        tvout_basic[0] ->sync();
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    fast_free(work_space_data);
    LOG(INFO) << "basic permute running time, ave: " << to / test_iter << ", min time: " << min_time;
   // print_tensor_host(tout_basic);

    LOG(INFO) << "run ncnn permute for precision comparation";
    TensorHf4 tout_basic2;
    tout_basic2.re_alloc(shape_out);
    t1.clear();
    t1.start(ctx1);
    //permute_ncnn(ctx1,tout_basic2, *thin, need_permute, order_type, num_axes, count, permute);
    t1.end(ctx1);
    LOG(INFO) << "ncnn permute running time, ave: " << t1.get_average_ms() << ", min time: " << min_time;
    double max_ratio1 = 0;
    double max_diff1 = 0;
    tensor_cmp_host(tout_basic.data(), tout_basic2.data(), tout_basic.valid_size(), max_ratio1, max_diff1);
   // LOG(INFO) << "tout_basic";
   // print_tensor_host(tout_basic);
  // LOG(INFO) << "tout_saber";
   // print_tensor_host(tout_saber);
    LOG(INFO) << "compare result, max diff: " << max_diff1 << ", max ratio: " << max_ratio1;
    CHECK_EQ(fabsf(max_ratio1) < 1e-5f, true) << "compute result error";

#endif
    
    Permute<ARM, AK_FLOAT> permute_saber;
    PermuteParam<Tensor<ARM, AK_FLOAT, NCHW>> permute_param(permute);

    permute_saber.compute_output_shape(tin, tvout_saber, permute_param);

    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape_1: " << sh_out_saber[0] << ", " << sh_out_saber[1] << ", " \
        << sh_out_saber[2] << ", " << sh_out_saber[3];
    //LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);

    LOG(INFO) << "saber permute impl init";
    SABER_CHECK(permute_saber.init(tin, tvout_saber, permute_param, SPECIFY, SABER_IMPL, ctx2));

    //! compute
    LOG(INFO) << "saber permute compute";
    to = 0;
    min_time = 1000000;
    for (int i = 0; i < test_iter; ++i) {
        t2.clear();
        t2.start(ctx2);
       // pooling_saber(tin, tvout_saber, pooling_param, ctx1);
       // permute_arm2(ctx2,tout_saber, *thin, need_permute,transpose, order_type, num_axes, count, permute);
       // permute_arm2(ctx2,tout_saber, *thin, permute_saber._need_permute,permute_saber._transpose, .permute_saber_order_type, permute_saber._num_axes, permute_saber._count, permute);
        permute_saber(tin, tvout_saber, permute_param, ctx2);
        tvout_saber[0]->record_event(ctx2.get_compute_stream());
        tvout_saber[0]->sync();
        t2.end(ctx2);
        //printf("i: %d \n",i);
        to += t2.get_average_ms();
        if (t2.get_average_ms() < min_time) {
            min_time = t2.get_average_ms();
        }
    }
    LOG(INFO) << "saber permute running time, ave: " << to / test_iter << ", min time: " << min_time;
    //print_tensor_host(tout_saber);
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
TEST(TestSaberFuncTest, test_func_pooling_global_arm) {

    int num = 1;
    int chin = 6;
    int hin = 224;
    int win = 224;

    int pad = 1;
    int stride = 2;
    int kernel = 3;
    //int chout = 3;

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

    int count = num * chin * win * hin;
    int num_axes = 4;
    std::vector<int> permute;
    /*
    permute.push_back(0);
    permute.push_back(1);
    permute.push_back(3);
    permute.push_back(2);
	*/
	permute.push_back(GLB_num);
	permute.push_back(GLB_ch);
	permute.push_back(GLB_h);
	permute.push_back(GLB_w);
    test_arm_permute(tin, num_axes, permute,GLB_threads, GLB_cluster_id);
    //LOG(WARNING) << "pooling not support yet";
}
#endif

int main(int argc, const char** argv){
    anakin::saber::Env<ARM>::env_init();

    // initial logger
    //logger::init(argv[0]);
     if (argc < 1) {
        LOG(INFO) << "Example of Usage:\n \
        ./output/unit_test/pooloing_test\n \
            type\n \
            global\n \
            threads\n \
            cluster_id\n ";
        exit(0);
    } else if (argc == 3){
        GLB_threads = atoi(argv[1]);
        GLB_cluster_id = atoi(argv[2]);
    }else if (argc == 7) {
    	GLB_threads = atoi(argv[1]);
        GLB_cluster_id = atoi(argv[2]);
        GLB_num = atoi(argv[3]);
        GLB_ch = atoi(argv[4]);
        GLB_h = atoi(argv[5]);
        GLB_w = atoi(argv[6]);
    }
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}


