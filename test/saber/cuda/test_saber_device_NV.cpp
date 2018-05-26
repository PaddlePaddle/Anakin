#include "test_saber_device_NV.h"

#ifdef USE_CUDA

using namespace anakin::saber;

TEST(TestSaberDeviceNV, test_NV_device) {
    Device<NV> dev_NV;
}

#endif

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

