#include "test_saber_context_ARM.h"

#ifdef USE_ARM_PLACE

using namespace anakin::saber;

TEST(TestSaberContextARM, test_arm_context) {

    Context<ARM> ctx;
    LOG(INFO) << "create runtime ctx";
    //ctx.set_power_mode(MERC_HIGH);
    ctx.set_act_cores({4, 5, 6, 7});
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

#endif

int main(int argc, const char** argv){

    Env<ARM>::env_init(8);

    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}