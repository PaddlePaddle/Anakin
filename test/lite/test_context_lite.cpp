#include "test_lite.h"
#include "saber/lite/core/context_lite.h"

using namespace anakin;
using namespace anakin::saber;
using namespace anakin::saber::lite;

TEST(TestSaberLite, test_arm_context) {

    Context ctx;
    LOG(INFO) << "create runtime ctx";
    //ctx.set_power_mode(MERC_HIGH);
    //ctx.set_act_cores({4, 5, 6, 7});
    LOG(INFO) << "high mode, 4 threads";
    ctx.set_run_mode(SABER_POWER_HIGH, 4);
    LOG(INFO) << "set active ids";

    LOG(INFO) << "test threads activated";
#ifdef USE_OPENMP
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

    LOG(INFO) << "high mode, 2 threads";
    ctx.set_run_mode(SABER_POWER_HIGH, 2);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
        int threads = omp_get_num_threads();
        printf("number of threads: %d\n", threads);
    }
#pragma omp parallel private(th_id)
    {
        th_id = omp_get_thread_num();
#pragma omp parallel
        printf("thread1 core ID: %d\n", th_id);

    }

    LOG(INFO) << "high mode, 1 threads";
    ctx.set_run_mode(SABER_POWER_HIGH, 1);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
        int threads = omp_get_num_threads();
        printf("number of threads: %d\n", threads);
    }
#pragma omp parallel private(th_id)
    {
        th_id = omp_get_thread_num();
#pragma omp parallel
        printf("thread1 core ID: %d\n", th_id);

    }

    LOG(INFO) << "low mode, 4 threads";
    ctx.set_run_mode(SABER_POWER_LOW, 4);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
        int threads = omp_get_num_threads();
        printf("number of threads: %d\n", threads);
    }
#pragma omp parallel private(th_id)
    {
        th_id = omp_get_thread_num();
#pragma omp parallel
        printf("thread1 core ID: %d\n", th_id);

    }

    LOG(INFO) << "low mode, 2 threads";
    ctx.set_run_mode(SABER_POWER_LOW, 2);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
        int threads = omp_get_num_threads();
        printf("number of threads: %d\n", threads);
    }
#pragma omp parallel private(th_id)
    {
        th_id = omp_get_thread_num();
#pragma omp parallel
        printf("thread1 core ID: %d\n", th_id);

    }

    LOG(INFO) << "low mode, 1 threads";
    ctx.set_run_mode(SABER_POWER_LOW, 1);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
        int threads = omp_get_num_threads();
        printf("number of threads: %d\n", threads);
    }
#pragma omp parallel private(th_id)
    {
        th_id = omp_get_thread_num();
#pragma omp parallel
        printf("thread1 core ID: %d\n", th_id);

    }
#endif
}

int main(int argc, const char** argv){

    Env::env_init();

    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}