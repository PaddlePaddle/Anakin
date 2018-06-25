#include "test_saber_context_BM.h"

#ifdef USE_BM

using namespace anakin::saber;

TEST(TestSaberContextBM, test_BM_context) {
    Env<BM>::env_init();
    typedef TargetWrapper<BM> API;
    typename API::event_t event;
    API::create_event(event);
    LOG(INFO) << "test context constructor";
    Context<BM> ctx0;
    Context<BM> ctx1(0, 1, 1);

    //for BM no need to test stream as it is not in use
}

#endif

int main(int argc, const char** argv) {
    //TODO: init in another place
    static bm_handle_t handle;
    bmdnn_init(&handle);
    
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

