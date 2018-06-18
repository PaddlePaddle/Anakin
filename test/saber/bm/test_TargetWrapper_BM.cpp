#include "saber_types.h"
#include "target_wrapper.h"
#include <iostream>

#ifdef USE_BM
using namespace anakin::saber;
int main() {
    typedef TargetWrapper<BM> API;
    void *pmem;
    int dev_count;
    API::get_device_count(&dev_count);
    API::mem_alloc(&pmem, 3*200*200);
    API::mem_free(pmem);
}
#endif

