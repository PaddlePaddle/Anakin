#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "saber/core/context.h"
#include "saber/funcs/attension_lstm.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/core/tensor_op.h"
#include "debug.h"
#include "test_saber_func.h"
#include <cmath>
using namespace anakin::saber;
using namespace std;

#ifdef USE_X86_PLACE

TEST(TestSaberFunc, test_func_attension_lstm_x86) {
    Env<X86>::env_init();
}

#endif


#ifdef NVIDIA_GPU
TEST(TestSaberFunc, test_func_attension_lstm_nv) {
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
}

#endif

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}