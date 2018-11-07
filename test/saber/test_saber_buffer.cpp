#include "test_saber_func.h"
#include "saber/core/buffer.h"
#include "saber/core/env.h"
#include "saber/core/data_traits.h"

using namespace anakin::saber;


template <typename Td, typename Th, DataType datatype>
void test_buffer() {

    typedef TargetWrapper<Th> HAPI;
    typedef TargetWrapper<Td> DAPI;
    typedef typename DataTrait<Th, datatype>::Dtype Dtype;
    typedef Buffer<Th> BufferH;
    typedef Buffer<Td> BufferD;

    typedef typename DataTraitBase<Td>::PtrDtype TPtr;

    int n0 = 1024;
    int n1 = 2048;

    void* tmp_h_ptr;
    Dtype* h_ptr;
    HAPI::mem_alloc(&tmp_h_ptr, sizeof(Dtype) * n0);
    h_ptr = static_cast<Dtype*>(tmp_h_ptr);

    for (int i = 0; i < n0; i++) {
        h_ptr[i] = static_cast<Dtype>(i);
    }

    void* tmp_d_ptr;
    TPtr d_ptr;
    DAPI::mem_alloc(&tmp_d_ptr, sizeof(Dtype) * n0);
    d_ptr = static_cast<TPtr>(tmp_d_ptr);

    LOG(INFO) << "Buffer: test default(empty) constructor";
    BufferH h_buf0;
    BufferD d_buf0;

    LOG(INFO) << "Buffer: test constructor with data size";
    BufferH h_buf1(n0 * sizeof(Dtype));
    BufferD d_buf1(n0 * sizeof(Dtype));

    LOG(INFO) << "Buffer: test constructor with data pointer, size and device id";
    BufferH h_buf2(h_ptr, n0 * sizeof(Dtype), HAPI::get_device_id());
    BufferD d_buf2(d_ptr, n0 * sizeof(Dtype), DAPI::get_device_id());

    LOG(INFO) << "Buffer: test copy constructor";
    BufferH h_buf3(h_buf2);
    LOG(INFO) << "NV Buffer copy constructor";
    LOG(INFO) << "nv target id: " << DAPI::get_device_id();
    LOG(INFO) << "nv buffer target id: " << d_buf2.get_id();
    BufferD d_buf3(d_buf2);
    CHECK_EQ(h_buf3.get_count(), h_buf2.get_count()) << \
            "shared buffer should have same data count";
    CHECK_EQ(d_buf3.get_count(), d_buf2.get_count()) << \
            "shared buffer should have same data count";
    //CHECK_EQ(x86_buf3.get_data()[n0 / 2], x86_buf2.get_data()[n0 / 2]) << \
    // "shared buffer should have same data value";
    //CHECK_EQ(nv_buf3.get_data()[n0 / 2], nv_buf2.get_data()[n0 / 2]) << \
    // "shared buffer should have same data value";

    LOG(INFO) << "Buffer: test operator =";
    h_buf0 = h_buf2;
    d_buf0 = d_buf2;
    CHECK_EQ(h_buf0.get_count(), h_buf2.get_count()) << \
            "shared buffer should have same data count";
    CHECK_EQ(d_buf0.get_count(), d_buf2.get_count()) << \
            "shared buffer should have same data count";
    //CHECK_EQ(x86_buf0.get_data()[n0 / 2], x86_buf2.get_data()[n0 / 2]) << \
    // "shared buffer should have same data value";
    //CHECK_EQ(nv_buf0.get_data()[n0 / 2], nv_buf2.get_data()[n0 / 2]) << \
    // "shared buffer should have same data value";

    LOG(INFO) << "Buffer: test re_alloc";
    h_buf1.re_alloc(n1 * sizeof(Dtype));
    d_buf1.re_alloc(n1 * sizeof(Dtype));
    CHECK_EQ(h_buf1.get_count(), n1 * sizeof(Dtype)) << "buffer count error";
    CHECK_EQ(h_buf1.get_capacity(), n1 * sizeof(Dtype)) << "buffer capacity error";
    CHECK_EQ(d_buf1.get_count(), n1 * sizeof(Dtype)) << "buffer count error";
    CHECK_EQ(d_buf1.get_capacity(), n1 * sizeof(Dtype)) << "buffer capacity error";
    h_buf1.re_alloc(n0 * sizeof(Dtype));
    d_buf1.re_alloc(n0 * sizeof(Dtype));
    CHECK_EQ(h_buf1.get_count(), n0 * sizeof(Dtype)) << "buffer count error";
    CHECK_EQ(h_buf1.get_capacity(), n1 * sizeof(Dtype)) << "buffer capacity error";
    CHECK_EQ(h_buf1.get_count(), n0 * sizeof(Dtype)) << "buffer count error";
    CHECK_EQ(h_buf1.get_capacity(), n1 * sizeof(Dtype)) << "buffer capacity error";

    LOG(INFO) << "Buffer: test get_id()";
    LOG(INFO) << "X86 device id: " << h_buf0.get_id() << \
              ", nv device id: " << h_buf0.get_id();
    CHECK_EQ(HAPI::get_device_id(), h_buf0.get_id()) << "x86 device id error";
    CHECK_EQ(DAPI::get_device_id(), d_buf0.get_id()) << "nv device id error";

    LOG(INFO) << "Buffer: test deep_cpy()";
    h_buf1.sync_copy_from(h_buf2);
    LOG(INFO) << "deep copy between two host buffer: ";
    const Dtype* ptr1 = static_cast<const Dtype*>(h_buf1.get_data());
    const Dtype* ptr2 = static_cast<const Dtype*>(h_buf2.get_data());

    for (int i = 0; i < 10; i++) {
        printf("%.6f  ", static_cast<float>(ptr1[i]));
    }

    printf("\n");

    CHECK_EQ(ptr1[n0 / 2], ptr2[n0 / 2]) << "deep copy between host is incorrect";
    LOG(INFO) << "deep copy from host buffer to device buffer";
    d_buf1.sync_copy_from(h_buf2);
    h_buf1.sync_copy_from(d_buf1);
    LOG(INFO) << "deep copy from device buffer to host buffer: ";
    ptr1 = static_cast<const Dtype*>(h_buf1.get_data());

    for (int i = 0; i < 10; i++) {
        printf("%.6f  ", static_cast<float>(ptr1[i]));
    }

    printf("\n");
}

TEST(TestSaberFunc, test_saber_buffer) {
#ifdef USE_CUDA
    LOG(INFO) << "test NV FP32 buffer";
    test_buffer<NV, NVHX86, AK_FLOAT>();
    LOG(INFO) << "test NV INT8 buffer";
    test_buffer<NV, NVHX86, AK_INT8>();
#endif

#ifdef AMD_GPU
    Env<AMD>::env_init();
    Env<AMDHX86>::env_init();
    LOG(INFO) << "test AMD FP32 buffer";
    test_buffer<AMD, AMDHX86, AK_FLOAT>();
    //LOG(INFO) << "test NV INT8 buffer";
    //test_buffer<AMD, AMDHX86, AK_INT8>();
#endif

#ifdef USE_X86_PLACE
    LOG(INFO) << "test X86 FP32 buffer";
    test_buffer<X86, X86, AK_FLOAT>();
    LOG(INFO) << "test X86 INT8 buffer";
    test_buffer<X86, X86, AK_INT8>();
#endif

#ifdef USE_ARM_PLACE
    LOG(INFO) << "test ARM FP32 buffer";
    test_buffer<ARM, ARM, AK_FLOAT>();
    LOG(INFO) << "test ARM INT8 buffer";
    test_buffer<ARM, ARM, AK_INT8>();
#endif

#ifdef USE_BM
    LOG(INFO) << "test BM FP32 buffer";
    //test_buffer<BM, X86, AK_FLOAT>();
#endif

}

template <typename TargetType, typename TargetType_h, typename dtype>
void test_buff_from_vector(std::vector<dtype>& data) {
    Buffer<TargetType> buff_d;
    Buffer<TargetType_h> buff_h;
    buff_d.from_vector(data);

    buff_h.re_alloc(sizeof(dtype) * data.size());
    buff_h.sync_copy_from(buff_d);
    const dtype* host_data = (const dtype*)buff_h.get_data();

    for (int i = 0; i < data.size(); ++i) {
        if (fabs(data[i] - host_data[i]) > 1e-5) {
            LOG(FATAL) << "check error! " << data[i] << "vector copy " << host_data[i];
        }
    }
}

TEST(TestSaberFunc, test_buff_apis) {

    std::vector<float> test1 {1.5f};
    std::vector<float> test2 {1.f, 2.f, 1.f, 3.f, 4.f, 2.f, 7.f, 2.f};
    std::vector<int> test3 {7};
    std::vector<int> test4 {1, 2, 1, 3, 4, 2, 7, 2};
#ifdef USE_CUDA
    test_buff_from_vector<NV, NVHX86, float>(test1);
    test_buff_from_vector<NV, NVHX86, float>(test2);
    test_buff_from_vector<NV, NVHX86, int>(test3);
    test_buff_from_vector<NV, NVHX86, int>(test4);
    test_buff_from_vector<NVHX86, NVHX86, float>(test1);
    test_buff_from_vector<NVHX86, NVHX86, float>(test2);
    test_buff_from_vector<NVHX86, NVHX86, int>(test3);
    test_buff_from_vector<NVHX86, NVHX86, int>(test4);
#endif
#ifdef USE_X86_PLACE
    test_buff_from_vector<X86, X86, float>(test1);
    test_buff_from_vector<X86, X86, float>(test2);
    test_buff_from_vector<X86, X86, int>(test3);
    test_buff_from_vector<X86, X86, int>(test4);
#endif
    LOG(INFO) << "Buffer api: from_buffer check pass";
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
