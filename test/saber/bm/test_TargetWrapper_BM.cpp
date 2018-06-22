#include "saber_types.h"
#include "target_wrapper.h"
#include <iostream>

#ifdef USE_BM
using namespace anakin::saber;
static bm_handle_t handle;
int main() {
    bmdnn_init(&handle);
    typedef TargetWrapper<BM> API;
    void *pmem;
    int dev_count = 0;
    API::get_device_count(dev_count);
    std::cout << dev_count << std::endl;
    API::mem_alloc(&pmem, 3*200*200);
    //API::mem_free(pmem);
    std::cout << "Press any key to finish execution." << std::endl;
    int a;
    std::cin >> a;
    bmdnn_deinit(handle);
}
#endif

