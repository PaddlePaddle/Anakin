#include "test_saber_buffer_bm.h"
#include "saber/core/buffer.h"
#include "saber/core/data_traits.h"

using namespace anakin::saber;

template <DataType datatype>
void test_buffer() {

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<BM> BM_API;
    typedef typename DataTrait<datatype>::dtype Dtype;
    typedef Buffer<X86> BufferH;
    typedef Buffer<BM> BufferD;

    int n0 = 1024;
    int n1 = 2048;

    void* tmp_x86;
    Dtype* x86_ptr;
    X86_API::mem_alloc(&tmp_x86, sizeof(Dtype) * n0);
    x86_ptr = static_cast<Dtype*>(tmp_x86);

    for (int i = 0; i < n0; i++) {
        x86_ptr[i] = static_cast<Dtype>(i);
    }

    void* tmp_bm;
    Dtype* bm_ptr;
    BM_API::mem_alloc(&tmp_bm, sizeof(Dtype) * n0);
    bm_ptr = static_cast<Dtype*>(tmp_bm);

    LOG(INFO) << "Buffer: test default(empty) constructor";
    BufferH x86_buf0;
    BufferD bm_buf0;

    LOG(INFO) << "Buffer: test constructor with data size";
    BufferH x86_buf1(n0 * sizeof(Dtype));
    BufferD bm_buf1(n0 * sizeof(Dtype));

    LOG(INFO) << "Buffer: test constructor with data pointer, size and device id";
    BufferH x86_buf2(x86_ptr, n0 * sizeof(Dtype), X86_API::get_device_id());
    BufferD bm_buf2(bm_ptr, n0 * sizeof(Dtype), BM_API::get_device_id());

    LOG(INFO) << "Buffer: test copy constructor";
    BufferH x86_buf3(x86_buf2);
    LOG(INFO) << "BM Buffer copy constructor";
    LOG(INFO) << "bm target id: " << BM_API::get_device_id();
    LOG(INFO) << "bm buffer target id: " << bm_buf2.get_id();
    BufferD bm_buf3(bm_buf2);
    CHECK_EQ(x86_buf3.get_count(), x86_buf2.get_count()) << \
            "shared buffer should have same data count";
    CHECK_EQ(bm_buf3.get_count(), bm_buf2.get_count()) << \
            "shared buffer should have same data count";

    LOG(INFO) << "Buffer: test operator =";
    x86_buf0 = x86_buf2;
    bm_buf0 = bm_buf2;
    CHECK_EQ(x86_buf0.get_count(), x86_buf2.get_count()) << \
            "shared buffer should have same data count";
    CHECK_EQ(bm_buf0.get_count(), bm_buf2.get_count()) << \
            "shared buffer should have same data count";

    LOG(INFO) << "Buffer: test re_alloc";
    x86_buf1.re_alloc(n1 * sizeof(Dtype));
    bm_buf1.re_alloc(n1 * sizeof(Dtype));
    CHECK_EQ(x86_buf1.get_count(), n1 * sizeof(Dtype)) << "buffer count error";
    CHECK_EQ(x86_buf1.get_capacity(), n1 * sizeof(Dtype)) << "buffer capacity error";
    CHECK_EQ(bm_buf1.get_count(), n1 * sizeof(Dtype)) << "buffer count error";
    CHECK_EQ(bm_buf1.get_capacity(), n1 * sizeof(Dtype)) << "buffer capacity error";
    x86_buf1.re_alloc(n0 * sizeof(Dtype));
    bm_buf1.re_alloc(n0 * sizeof(Dtype));
    CHECK_EQ(x86_buf1.get_count(), n0 * sizeof(Dtype)) << "buffer count error";
    CHECK_EQ(x86_buf1.get_capacity(), n1 * sizeof(Dtype)) << "buffer capacity error";
    CHECK_EQ(x86_buf1.get_count(), n0 * sizeof(Dtype)) << "buffer count error";
    CHECK_EQ(x86_buf1.get_capacity(), n1 * sizeof(Dtype)) << "buffer capacity error";

    LOG(INFO) << "Buffer: test get_id()";
    LOG(INFO) << "X86 device id: " << x86_buf0.get_id() << \
              ", bm device id: " << bm_buf0.get_id();
    CHECK_EQ(X86_API::get_device_id(), x86_buf0.get_id()) << "x86 device id error";
    CHECK_EQ(BM_API::get_device_id(), bm_buf0.get_id()) << "bm device id error";

    LOG(INFO) << "Buffer: test deep_cpy()";
    x86_buf1.sync_copy_from(x86_buf2);
    LOG(INFO) << "deep copy between two host buffer: ";
    const Dtype* ptr1 = static_cast<const Dtype*>(x86_buf1.get_data());
    const Dtype* ptr2 = static_cast<const Dtype*>(x86_buf1.get_data());

    for (int i = 0; i < 10; i++) {
        std::cout << ptr1[i] << std::endl;
    }

    CHECK_EQ(ptr1[n0 / 2], ptr2[n0 / 2]) << "deep copy between host is incorrect";
    LOG(INFO) << "deep copy from host buffer to device buffer";
    bm_buf1.sync_copy_from(x86_buf2);
    x86_buf1.sync_copy_from(bm_buf1);
    LOG(INFO) << "deep copy from device buffer to host buffer: ";
    ptr1 = static_cast<const Dtype*>(x86_buf1.get_data());

    for (int i = 0; i < 10; i++) {
        std::cout << ptr1[i] << std::endl;
    }
}

TEST(TestSaberBufferBM, test_buffer_memcpy) {
    test_buffer<AK_BM>();
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
