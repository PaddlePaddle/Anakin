#include "saber_types.h"
#include "target_wrapper.h"
#include <iostream>

#ifdef USE_BM
using namespace anakin::saber;
//static bm_handle_t handle;
int main() {
    //bmdnn_init(&handle);
    typedef TargetWrapper<BM> API;
    //int dev_count = 0;
    //API::get_device_count(dev_count);
    //std::cout << "dev_count: " << dev_count << std::endl;
    
    //bm_device_mem_t *pmem = new bm_device_mem_t();
    void* pmem;
    std::cout << "mem addr before mem_alloc: " << pmem << std::endl;
    API::mem_alloc(&pmem, 3*200*400);
    std::cout << "mem addr after  mem_alloc: " << pmem << std::endl;
    std::cout << "Start mem_free test." << std::endl;
    API::mem_free(pmem);
    std::cout << "End mem_free test." << std::endl;
    //bmdnn_deinit(handle);
}
#endif

