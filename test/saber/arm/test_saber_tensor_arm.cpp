#include "test_saber_func_test_arm.h"
#include "tensor_op.h"
#include <vector>
using namespace anakin::saber;

typedef TargetWrapper<ARM> ARM_API;
typedef Tensor<ARM, AK_FLOAT, NCHW> Tensor4f;
//typedef Tensor<ARM, AK_FLOAT, HW> Tensor2f;

TEST(TestSaberFuncTest, test_tensor_constructor) {

//! test empty constructor
    LOG(INFO) << "test default (empty) constructor";
    Tensor4f thost0;

//! test tensor re_alloc function empty constructor
    Shape sh0(2, 3, 10, 10);
    LOG(INFO) << "|--test tensor re_alloc function on empty tensor";
    thost0.re_alloc(sh0);
    LOG(INFO) << "|--tensor size of host: " << thost0.size();
    CHECK_EQ(thost0.size(), 600) << "error with tensor size";

//! test tensor re_alloc function on tensor with data
    LOG(INFO) << "|--test tensor re_alloc function on tensor with data";
    Shape sh1(1, 3, 10, 10);
    thost0.re_alloc(sh1);
    LOG(INFO) << "|--tensor size of host: " << thost0.size();
    CHECK_EQ(thost0.size(), 300) << "error with tensor size";


//! test tensor shape() function
    LOG(INFO) << "|--test tensor shape() function";
    Shape sho = thost0.shape();
    LOG(INFO) << "|--shape of tensor: " << sho[0] << ", " << sho[1] << "," << sho[2] << "," <<sho[3];
    LOG(INFO) << "|--test get tensor n, c, h, w function, num = " \
        << thost0.num() << ", channel = " << thost0.channel() << ", height = " \
        << thost0.height() << ", width = " << thost0.width();

//! test tensor mutable_data() function
    LOG(INFO) << "|--test tensor mutable_data() function, write tensor data buffer with 1.f";
    fill_tensor_host_const(thost0, 1.f);
    LOG(INFO) << "|--test tensor data() function, show the const data, 1.f";
    print_tensor_host(thost0);

//! test tensor constructor with shape
    LOG(INFO) << "test tensor constructor with shape";
    Tensor4f thost1(sh1);

//! test tensor copy_from() function
    LOG(INFO) << "test copy_from() function, input tensor could be any target";
    thost1.copy_from(thost0);
    print_tensor_host(thost1);

//! test tensor constructor with data, if target is different, create buffer, and copy the data
    LOG(INFO) << "test tensor constructor with data, if target is different, create buffer, and copy the data";
    float* host_data_ptr;
    void* tmp_ptr;
    ARM_API::mem_alloc(&tmp_ptr, sizeof(float) * sh1.count());
    host_data_ptr = static_cast<float*>(tmp_ptr);
    for (int i = 0; i < sh1.count(); ++i) {
        host_data_ptr[i] = i;
    }
    LOG(INFO) << "|--construct host tensor from host data ptr";
    Tensor4f thost3(host_data_ptr, ARM(), ARM_API::get_device_id(), sh1);
    print_tensor_host(thost3);

//! test tensor copy constructor
    LOG(INFO) << "test tensor copy constructor";
    LOG(INFO) << "|--normal copy constructor";
    Tensor4f thost4(thost3);

    LOG(INFO) << "|--push back to vector";
    std::vector<Tensor4f> vthost;
    vthost.push_back(thost0);
    vthost.push_back(thost1);
    vthost.push_back(thost3);
    vthost.push_back(thost4);
    print_tensor_host(vthost[3]);

//! test share_from function, if targets are the same, buffer is shared, otherwise, buffer is copied
    LOG(INFO) << "test share_from function";
    Tensor4f thost5;
    Shape sh2(1, 3, 5, 5);
    Shape offset(0, 0, 5, 5);
    LOG(INFO) << "|--shared host";
    thost5.set_shape(sh2, thost3.shape(), offset);
    thost5.share_from(thost3);

    LOG(INFO) << "|--change data in shared tensor";
    Shape sh_real = thost5.shape();
    Shape sh_act = thost5.valid_shape();
    Shape offset_act = thost5.offset();
    int start_w = offset_act[3];
    int start_h = offset_act[2];
    int start_c = offset_act[1];
    int start_n = offset_act[0];
    int stride_h = sh_real.count(3);
    int stride_c = sh_real.count(2);
    int stride_n = sh_real.count(1);
//int stride_n = sh_real.count(0);
    int w = thost5.width();
    int h = thost5.height();
    int c = thost5.channel();
    int n = thost5.num();
    float* ptr_host = thost5.mutable_data();
    for (int in = 0; in < n; ++in) {
        float* ptr_batch = ptr_host + (in + start_n) * stride_n;
        for (int ic = 0; ic < c; ++ic) {
            float* ptr_channel = ptr_batch + (ic + start_c) * stride_c;
            for (int ih = 0; ih < h; ++ih) {
                float* ptr_row = ptr_channel + (ih + start_h) * stride_h;
                for (int iw = 0; iw < w; ++iw) {
                    ptr_row[start_w + iw] = 1.f;
                }
            }
        }
    }

    LOG(INFO) << "|--show root tensor while data is changed by shared tensor";
    print_tensor_host(thost3);
}
#if 0
TEST(TestSaberTensorARM, test_tensor_deepcopy) {
    //! tensor constructor with alloc data, if target is different, create buffer, and copy the data
    LOG(INFO) << "tensor constructor with data, if target is different, create buffer, and copy the data";

    Shape sh0(2, 4, 8, 8);
    Shape va_sh0(2, 4, 4, 4);
    Shape off_sh0(0, 0, 2, 2);
    Shape sh1(2, 4, 10, 4);
    Shape va_sh1(va_sh0);
    Shape off_sh1(0, 0, 4, 0);
    Shape sh2(4, 64);
    Shape va_sh2(2, 64);
    Shape off_sh2(1, 0);

    LOG(INFO) << "|--construct host tensor from host data ptr";
    //! create thost0, thost1, thost01 are source tensor
    Tensor4f thost0(sh0);
    for (int i = 0; i < sh0.count(); ++i) {
        thost0.mutable_data()[i] = i;
    }
    print_tensor_host(thost0);
    //! create shared tensor, with valid shape and offset
    Tensor4f thost01;
    thost01.set_shape(va_sh0, sh0, off_sh0);
    thost01.share_from(thost0);
    //! create tensor with entire shape, valid shape and offset
    Tensor4f thost1(va_sh0);
    for (int i = 0; i < va_sh0.count(); ++i) {
        thost1.mutable_data()[i] = i;
    }

    //! create thost2, thost3, thost21 as dst tensor, same layout with src
    Tensor4f thost2(sh1);
    fill_tensor_host_const(thost2, 0.f);
    Tensor4f thost21;
    thost21.set_shape(va_sh1, sh1, off_sh1);
    thost21.share_from(thost2);
    Tensor4f thost3(va_sh1);

    //! create thost4, thost5, thost41 as dst tensor, different layout with src
    Tensor2f thost4(sh2);
    fill_tensor_host_const(thost4, 0.f);
    Tensor2f thost41;
    thost41.set_shape(va_sh2, sh2, off_sh2);
    thost41.share_from(thost4);
    Tensor2f thost5(va_sh2);

    //! test tensor deep copy, entire buffer copy
    LOG(INFO) << "test tensor deep copy, entire buffer copy";
    thost3.copy_from(thost1);
    print_tensor_host(thost3);

    //! test tensor deep copy, src with roi
    LOG(INFO) << "test tensor deep copy, src with roi";
    thost3.copy_from(thost01);
    print_tensor_host(thost3);

    //! test tensor deep copy, dst with roi
    LOG(INFO) << "test tensor deep copy, dst with roi";
    thost21.copy_from(thost1);
    print_tensor_host(thost21);

    //! test tensor deep copy, src and dst are with roi
    LOG(INFO) << "test tensor deep copy, src and dst are with roi";
    thost21.copy_from(thost01);
    print_tensor_host(thost21);

    //! test tensor deep copy, entire buffer copy
    LOG(INFO) << "test tensor deep copy, entire buffer copy, different layout";
    thost5.copy_from(thost1);
    print_tensor_host(thost5);

    //! test tensor deep copy, src with roi
    LOG(INFO) << "test tensor deep copy, src with roi, different layout";
    thost5.copy_from(thost01);
    print_tensor_host(thost5);

    //! test tensor deep copy, dst with roi
    LOG(INFO) << "test tensor deep copy, dst with roi, different layout";
    thost41.copy_from(thost1);
    print_tensor_host(thost41);

    //! test tensor deep copy, src and dst are with roi
    LOG(INFO) << "test tensor deep copy, src and dst are with roi, different layout";
    thost41.copy_from(thost01);
    print_tensor_host(thost41);
}
#endif

int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}