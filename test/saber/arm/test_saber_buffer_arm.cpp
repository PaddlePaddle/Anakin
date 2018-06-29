#include "test_saber_buffer_arm.h"
#include "core/data_traits.h"
using namespace anakin::saber;
template <DataType datatype>
void test_buffer(){

    typedef TargetWrapper<ARM> ARM_API;
    typedef typename DataTrait<datatype>::dtype Dtype;
    typedef Buffer<ARM> BufferH;

    int n0 = 1024;
    int n1 = 2048;

    void* tmp_ptr = nullptr;
    Dtype* arm_ptr;
    ARM_API::mem_alloc(&tmp_ptr, sizeof(Dtype) * n0);
    arm_ptr = static_cast<Dtype*>(tmp_ptr);
    for(int i = 0; i < n0; i++){
        arm_ptr[i] = static_cast<Dtype>(i);
    }

    LOG(INFO) << "Buffer: test default(empty) constructor";
    BufferH arm_buf0;

    LOG(INFO) << "Buffer: test constructor with data size";
    BufferH arm_buf1(n0 * sizeof(Dtype));

    LOG(INFO) << "Buffer: test constructor with data pointer, size and device id";
    BufferH arm_buf2(arm_ptr, n0 * sizeof(Dtype), ARM_API::get_device_id());

    LOG(INFO) << "Buffer: test copy constructor";
    BufferH arm_buf3(arm_buf2);
    CHECK_EQ(arm_buf3.get_count(), arm_buf2.get_count()) << "shared buffer should have same data count";


    LOG(INFO) << "Buffer: test operator =";
    arm_buf0 = arm_buf2;
    CHECK_EQ(arm_buf0.get_count(), arm_buf2.get_count()) << "shared buffer should have same data count";

    LOG(INFO) << "Buffer: test re_alloc";
    arm_buf1.re_alloc(n1 * sizeof(Dtype));
    CHECK_EQ(arm_buf1.get_count(), n1 * sizeof(Dtype)) << "buffer count error";
    CHECK_EQ(arm_buf1.get_capacity(), n1 * sizeof(Dtype)) << "buffer capacity error";

    arm_buf1.re_alloc(n0 * sizeof(Dtype));
    CHECK_EQ(arm_buf1.get_count(), n0 * sizeof(Dtype)) << "buffer count error";
    CHECK_EQ(arm_buf1.get_capacity(), n1 * sizeof(Dtype)) << "buffer capacity error";

    LOG(INFO) << "Buffer: test get_id()";
    LOG(INFO) << "ARM device id: " << arm_buf0.get_id();
    CHECK_EQ(ARM_API::get_device_id(), arm_buf0.get_id()) << "ARM device id error";

    LOG(INFO) << "Buffer: test deep_cpy()";
    arm_buf1.sync_copy_from(arm_buf2);
    LOG(INFO) << "deep copy between two host buffer: ";
    Dtype* data_ptr1 = (Dtype*)arm_buf1.get_data();
    LOG(INFO) << "data in buffer 1";
    for(int i = 0; i < n0;i++) {
        printf("%.2f ", data_ptr1[i]);
        if ((i + 1) % 10 == 0) {
            printf("\n");
        }
    }
    Dtype* data_ptr2 = (Dtype*)arm_buf2.get_data();
    LOG(INFO) << "data in buffer2";
    for(int i = 0; i < n0;i++) {
        printf("%.2f ", data_ptr2[i]);
        if ((i + 1) % 10 == 0) {
            printf("\n");
        }
    }
    CHECK_EQ(data_ptr1[n0 / 2], data_ptr2[n0 / 2]) << "deep copy between host is incorrect";
    LOG(INFO) << "deep copy from host buffer to device buffer";
}

TEST(TestSaberBufferARM, test_buffer_memcpy) {
    test_buffer<AK_FLOAT>();
}

int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}


