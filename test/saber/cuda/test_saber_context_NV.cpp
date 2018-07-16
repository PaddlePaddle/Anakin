#include "test_saber_func_NV.h"
#include "saber/core/context.h"

#ifdef USE_CUDA

using namespace anakin::saber;

TEST(TestSaberFuncNV, test_NV_context) {
    Env<NV>::env_init();
    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(&event);
    LOG(INFO) << "test context constructor";
    Context<NV> ctx0;
    Context<NV> ctx1(0, 1, 1);
    LOG(INFO) << "test record event to context data stream and compute stream";
    API::record_event(event, ctx0.get_data_stream());
    API::record_event(event, ctx0.get_compute_stream());
    API::record_event(event, ctx1.get_data_stream());
    API::record_event(event, ctx1.get_compute_stream());
}

#endif

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

