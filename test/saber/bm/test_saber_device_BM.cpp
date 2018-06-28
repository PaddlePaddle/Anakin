#include "test_saber_device_BM.h"

#ifdef USE_BM

using namespace anakin::saber;

TEST(TestSaberDeviceBM, test_BM_device) {
    Device<BM> dev_BM;
}

#endif

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

