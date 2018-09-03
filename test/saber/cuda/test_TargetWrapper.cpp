#include "saber_types.h"
#include "target_wrapper.h"
#include <iostream>

#ifdef USE_CUDA
using namespace anakin::saber;
int main() {
    typedef TargetWrapper<NV> API;
    float* ptr = nullptr;
    API::mem_free(ptr);
}
#endif

