#include "saber/funcs/slice.h"
#include "test_saber_func_test_arm.h"
#include "tensor_op.h"

#define DEFINE_GLOBAL(type, var, value) \
        type (GLB_##var) = (value)
DEFINE_GLOBAL(int, threads, 1);
DEFINE_GLOBAL(int, cluster_id, 0);
DEFINE_GLOBAL(int, axis, 1);

#define USE_COMPARE

using namespace anakin::saber;

typedef TargetWrapper<ARM> ARM_API;
typedef Tensor<ARM, AK_FLOAT, NCHW> TensorHf4;

void test_arm_slice(std::vector<TensorHf4*>& tin, int axis, \ 
    std::vector<int> slice_point, int threads, int cluster_id) {


    Context<ARM> ctx1;
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

    TensorHf4* thin = tin[0];

    std::vector<TensorHf4*> tvout_saber;
    std::vector<TensorHf4*> tvout_basic;

   // tvout_saber.push_back(&tout_saber);
    //tvout_basic.push_back(&tout_basic);

    int num = tin[0]->num();
    int chin = tin[0]->channel();
    int hin = tin[0]->height();
    int win = tin[0]->width();

    LOG(INFO) << "slice param: ";
    LOG(INFO) << " img_num = " << num;
    LOG(INFO) << " in_channels = " << chin;
    LOG(INFO) << " img_h = " << hin;
    LOG(INFO) << " img_w = " << win;
    LOG(INFO) << " axis = " << axis;

    LOG(INFO) << " slice_point : " ;
    for (int i = 0; i < slice_point.size(); i ++){
    	printf ("%d, ",slice_point[i]);
    }
    printf("\n");
    int num_out = 0;
    int ch_out = 0;
    int w_out = 0;
    int h_out = 0;
    int count = slice_point.size() + 1;
    std::vector<TensorHf4> tdev(count);
    for (int i = 0; i < count; ++i) {
        tvout_saber.push_back(&tdev[i]);
        tvout_basic.push_back(&tdev[i]);
    }

   SliceParam<TensorHf4> slice_param(axis, slice_point);
   Slice<ARM, AK_FLOAT> slice_saber;
   LOG(INFO) << "slice compute output shape";
   slice_saber.compute_output_shape(tin, tvout_saber, slice_param);
   slice_saber.compute_output_shape(tin, tvout_basic, slice_param);

    for (int j = 0; j < tvout_saber.size(); ++j) {
        Shape sh = tvout_saber[j]->valid_shape();
        LOG(INFO) << "output shape: "<< sh[0] << ", " << sh[1] << \
            ", " << sh[2] << ", " << sh[3];
    }

    for (int i = 0; i < 4; ++i) {
        tvout_saber[i]->re_alloc(tvout_saber[i]->shape());
        tvout_basic[i]->re_alloc(tvout_basic[i]->shape());
    }



#ifdef USE_COMPARE
    LOG(INFO) << "run basic slice for precision comparation";
    size_t workspace_size = sizeof(float) * num * chin * hin  * win;
    void* work_space_data = fast_malloc(workspace_size);
    //Sgemm gemmer;
    int test_iter = 10;
    double to1 = 0.f;
    double to2 = 0.f;
    double min_time = 1000000;
    SaberTimer<ARM> t1;
    SaberTimer<ARM> t2;
     for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        //slice_basic(ctx1,tvout_basic, tin, axis, slice_point);
        tvout_basic[0]->record_event(ctx1.get_compute_stream());
        tvout_basic[0]->sync();
        t1.end(ctx1);
        to1 += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    fast_free(work_space_data);
    LOG(INFO) << "basic slice running time, ave: " << to1 / test_iter << ", min time: " << min_time;
    //print_tensor_host(tin[0]);
    //for(int i = 0; i < tvout_basic.size(); i ++)
    //	print_tensor_host(tvout_basic[i]);
     LOG(INFO) << "slice initialization";
    SABER_CHECK(slice_saber.init(tin, tvout_saber, slice_param, RUNTIME, SABER_IMPL, ctx1));
    LOG(INFO) << "saber slice compute";
    min_time = 1000000;
    for (int i = 0; i < test_iter; ++i) {
        t2.clear();
        t2.start(ctx1);
        slice_saber(tin, tvout_saber, slice_param, ctx1);
       // slice_arm(ctx1,tvout_saber, tin, axis, slice_point);
        tvout_saber[0]->record_event(ctx1.get_compute_stream());
        tvout_saber[0]->sync();
        t2.end(ctx1);
        to2 += t2.get_average_ms();
        if (t2.get_average_ms() < min_time) {
            min_time = t2.get_average_ms();
        }
        //printf("to2: %.3f\n");
    }
    LOG(INFO) << "saber slice running time, ave: " << to2 / test_iter << ", min time: " << min_time;
    
#endif
   
#ifdef USE_COMPARE
    double max_ratio = 0;
    double max_diff = 0;
    //TensorHf4 tdiff(tout_basic.valid_shape());
    //tensor_diff(tout_basic, tout_saber, tdiff);
    //print_tensor_host(tdiff);
    for (int i = 0; i < count; i ++){
    	tensor_cmp_host(tvout_basic[i]->data(), tvout_saber[i]->data(), tvout_basic[i]->valid_size(), max_ratio, max_diff);
    	LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    	CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
    }
   // LOG(INFO) << "tout_basic";
   // print_tensor_host(tout_basic);
  // LOG(INFO) << "tout_saber";
   // print_tensor_host(tout_saber);
    //LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
    //CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
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

    std::vector<int> slice_point;
    slice_point.push_back(10);
    slice_point.push_back(32);
    slice_point.push_back(50);
    test_arm_slice(tin, GLB_axis, slice_point, GLB_threads, GLB_cluster_id);
    //LOG(WARNING) << "pooling not support yet";
}
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
    } else if (argc == 4){
        GLB_threads = atoi(argv[1]);
        GLB_cluster_id = atoi(argv[2]);
        GLB_axis = atoi(argv[3]);
    }
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}