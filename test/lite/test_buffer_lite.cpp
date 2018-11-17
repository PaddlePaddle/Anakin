#include "test_lite.h"
#include "saber/lite/core/buffer_lite.h"
using namespace anakin;
using namespace anakin::saber;
using namespace anakin::saber::lite;
//template <DataType datatype>
void test_buffer(){
    LOG(INFO) << "test buffer";
    typedef typename DataTrait<CPU, AK_FLOAT>::Dtype Dtype;
    typedef Buffer<CPU> BufferH;

    int n0 = 1024;
    int n1 = 2048;

    void* tmp_ptr = nullptr;
    Dtype* arm_ptr;

    tmp_ptr = fast_malloc(n0 * sizeof(Dtype));
    arm_ptr = static_cast<Dtype*>(tmp_ptr);
    for (int i = 0; i < n0; i++){
        arm_ptr[i] = static_cast<Dtype>(i);
    }

    LOG(INFO) << "Buffer: test default(empty) constructor";
    BufferH arm_buf0;

    LOG(INFO) << "Buffer: test constructor with data size";
    BufferH arm_buf1(n0 * sizeof(Dtype));

    LOG(INFO) << "Buffer: test constructor with data pointer, size and device id";
    BufferH arm_buf2(arm_ptr, n0 * sizeof(Dtype));

    LOG(INFO) << "Buffer: test copy constructor";
    BufferH arm_buf3(arm_buf2);
    CHECK_EQ(arm_buf3.get_capacity(), arm_buf2.get_capacity()) << "shared buffer should have same data count";


    LOG(INFO) << "Buffer: test operator =";
    arm_buf0 = arm_buf2;
    CHECK_EQ(arm_buf0.get_capacity(), arm_buf2.get_capacity()) << "shared buffer should have same data count";

    LOG(INFO) << "Buffer: test re_alloc";
    arm_buf1.re_alloc(n1 * sizeof(Dtype));
    CHECK_EQ(arm_buf1.get_capacity(), n1 * sizeof(Dtype)) << "buffer count error";

    arm_buf1.re_alloc(n0 * sizeof(Dtype));
    CHECK_EQ(arm_buf1.get_capacity(), n1 * sizeof(Dtype)) << "buffer capacity error";

    LOG(INFO) << "Buffer: test deep_cpy()";
    arm_buf1.copy_from(arm_buf2);
    LOG(INFO) << "deep copy between two host buffer: ";
    Dtype* data_ptr1 = (Dtype*)arm_buf1.get_data();
    LOG(INFO) << "data in buffer 1";
    for (int i = 0; i < n0; i++) {
        printf("%.2f ", data_ptr1[i]);
        if ((i + 1) % 10 == 0) {
            printf("\n");
        }
    }
    printf("\n");
    Dtype* data_ptr2 = (Dtype*)arm_buf2.get_data();
    LOG(INFO) << "data in buffer2";
    for (int i = 0; i < n0; i++) {
        printf("%.2f ", data_ptr2[i]);
        if ((i + 1) % 10 == 0) {
            printf("\n");
        }
    }
    printf("\n");
    CHECK_EQ(data_ptr1[n0 / 2], data_ptr2[n0 / 2]) << "deep copy between host is incorrect";
    LOG(INFO) << "deep copy from host buffer to device buffer";
}

TEST(TestSaberLite, test_buffer_lite) {
     test_buffer();
}

int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}


