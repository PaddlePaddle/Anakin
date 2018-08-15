#include "test_saber_func.h"
#include "saber/core/context.h"

using namespace anakin::saber;

#ifdef USE_CUDA
TEST(TestSaberFunc, test_NV_context) {
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    typedef TargetWrapper<NV> NVAPI;
    typedef TargetWrapper<NV> NVHX86API;
    typename NVAPI::event_t event_nv;
    typename NVHX86API::event_t event_nvx86;
    NVAPI::create_event(&event_nv);
    NVHX86API::create_event(&event_nvx86);
    LOG(INFO) << "test context constructor";
    Context<NV> ctx0;
    Context<NV> ctx1(0, 1, 1);

    Context<NVHX86> ctx2;
    Context<NVHX86> ctx3(0, 1, 1);

    LOG(INFO) << "test record event to context data stream and compute stream";
    NVAPI::record_event(event_nv, ctx0.get_data_stream());
    NVAPI::record_event(event_nv, ctx0.get_compute_stream());
    NVAPI::record_event(event_nv, ctx1.get_data_stream());
    NVAPI::record_event(event_nv, ctx1.get_compute_stream());

    NVHX86API::record_event(event_nvx86, ctx2.get_data_stream());
    NVHX86API::record_event(event_nvx86, ctx2.get_compute_stream());
    NVHX86API::record_event(event_nvx86, ctx3.get_data_stream());
    NVHX86API::record_event(event_nvx86, ctx3.get_compute_stream());
}
#endif //USE_CUDA

#ifdef USE_ARM_PLACE
TEST(TestSaberFuncTest, test_arm_context) {
    Env<NV>::env_init();
    Context<ARM> ctx;
    LOG(INFO) << "create runtime ctx";
    ctx.set_run_mode(SABER_POWER_HIGH, 4);
    LOG(INFO) << "set active ids";

    LOG(INFO) << "test threads activated";
    #pragma omp parallel
    {
        int threads = omp_get_num_threads();
        printf("number of threads: %d\n", threads);
    }
    int th_id;
    #pragma omp parallel private(th_id)
    {
        th_id = omp_get_thread_num();
        #pragma omp parallel
        printf("thread1 core ID: %d\n", th_id);
    }
}
#endif //USE_ARM_PLACE

#ifdef USE_BM
TEST(TestSaberFunc, test_BM_context) {
    Context<BM> ctx;
    CHECK_NOTNULL(ctx.get_handle()) << "Failed to get BM handle";
}
#endif //USE_BM

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

