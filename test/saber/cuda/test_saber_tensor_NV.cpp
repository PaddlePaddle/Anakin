#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include <vector>
using namespace anakin::saber;

typedef TargetWrapper<NVHX86> X86_API;
typedef TargetWrapper<NV> NV_API;
typedef Tensor<NVHX86> TensorH;
typedef Tensor<NV> TensorD;


template <DataType Dtype, typename dtype>
void tensor_constructor() {

    //! test empty constructor
    LOG(INFO) << "test default (empty) constructor";
    TensorH thost0;
    TensorD tdev0;

    //! test tensor re_alloc function empty constructor
    Shape sh0({2, 2, 8, 8}, Layout_NCHW);
    LOG(INFO) << "|--test tensor re_alloc function on empty tensor";
    thost0.re_alloc(sh0, Dtype);
    tdev0.re_alloc(sh0, Dtype);
    LOG(INFO) << "|--tensor size of host: " << thost0.size();
    LOG(INFO) << "|--tensor size of device: " << tdev0.size();
    CHECK_EQ(thost0.size(), 256) << "error with tensor size";
    CHECK_EQ(tdev0.size(), 256) << "error with tensor size";

    //! test tensor re_alloc function on tensor with data
    LOG(INFO) << "|--test tensor re_alloc function on tensor with data";
    Shape sh1({1, 2, 4, 4}, Layout_NCHW);
    thost0.re_alloc(sh1, Dtype);
    tdev0.re_alloc(sh1, Dtype);
    LOG(INFO) << "|--tensor size of host: " << thost0.size();
    LOG(INFO) << "|--tensor size of device: " << tdev0.size();
    CHECK_EQ(thost0.size(), 32) << "error with tensor size";
    CHECK_EQ(tdev0.size(), 32) << "error with tensor size";

    //! test tensor shape() function
    LOG(INFO) << "|--test tensor shape() function";
    Shape sho = thost0.shape();
    LOG(INFO) << "|--shape of tensor: " << sho[0] << ", " << sho[1] << "," << sho[2] << "," << sho[3];
    LOG(INFO) << "|--test get tensor n, c, h, w function, num = " \
              << thost0.num() << ", channel = " << thost0.channel() << ", height = " \
              << thost0.height() << ", width = " << thost0.width();

    //! test tensor mutable_data() function
    LOG(INFO) << "|--test tensor mutable_data() function, write tensor data buffer with 1.f";
    fill_tensor_const(thost0, 1.f);
    LOG(INFO) << "|--test tensor data() function, show the const data, 1.f";
    print_tensor(thost0);

    //! test tensor constructor with shape
    LOG(INFO) << "test tensor constructor with shape";
    TensorH thost1(sh1, Dtype);
    TensorD tdev1(sh1, Dtype);

    //! test tensor copy_from() function
    LOG(INFO) << "test copy_from() function, input tensor could be any target";
    thost1.copy_from(thost0);
    tdev1.copy_from(thost0);
    print_tensor(tdev1);
    cudaDeviceSynchronize();
    thost1.copy_from(tdev1);
    tdev1.copy_from(tdev0);
    print_tensor(thost1);

    //! test tensor constructor with shape and real_shape
    LOG(INFO) << "test tensor constructor with shape and real_shape";
    //! constructor with 3 shapes is removed
    TensorH thost2(sh0, Dtype);
    TensorD tdev2(sh0, Dtype);

    //! test tensor constructor with data, if target is different, create buffer, and copy the data
    LOG(INFO) <<
              "test tensor constructor with data, if target is different, create buffer, and copy the data";
    dtype* host_data_ptr;
    dtype* dev_data_ptr;
    void* tmp_pt_host;
    void* tmp_pt_dev;
    X86_API::mem_alloc(&tmp_pt_host, sizeof(dtype) * sh1.count());
    host_data_ptr = static_cast<dtype*>(tmp_pt_host);

    for (int i = 0; i < sh1.count(); ++i) {
        host_data_ptr[i] = i;
    }

    NV_API::mem_alloc(&tmp_pt_dev, sizeof(dtype) * sh1.count());
    dev_data_ptr = static_cast<dtype*>(tmp_pt_dev);
    cudaMemcpy(dev_data_ptr, host_data_ptr, sizeof(dtype) * sh1.count(), cudaMemcpyHostToDevice);
    LOG(INFO) << "|--construct host tensor from host data ptr";
    TensorH thost3(host_data_ptr, NVHX86(), X86_API::get_device_id(), sh1, Dtype);
    LOG(INFO) << "|--constructor device tensor from host data ptr";
    TensorD tdev3(host_data_ptr, NVHX86(), X86_API::get_device_id(), sh1, Dtype);
    print_tensor(thost3);
    print_tensor(tdev3);
    cudaDeviceSynchronize();

    LOG(INFO) << "|--construct host tensor from device data ptr";
    TensorH thost4(dev_data_ptr, NV(), NV_API::get_device_id(), sh1, Dtype);
    LOG(INFO) << "|--constructor device tensor from device data ptr";
    TensorD tdev4(dev_data_ptr, NV(), NV_API::get_device_id(), sh1, Dtype);
    print_tensor(thost4);
    print_tensor(tdev4);
    NV_API::stream_t dev_stream0;
    NV_API::create_stream_with_flag(&dev_stream0, 1);
    cudaDeviceSynchronize();

    //! test tensor copy constructor
    LOG(INFO) << "test tensor copy constructor";
    LOG(INFO) << "|--normal copy constructor";
    TensorH thost5(thost4);
    TensorD tdev5(tdev4);

    LOG(INFO) << "|--push back to vector";
    std::vector<TensorH> vthost;
    std::vector<TensorD> vtdev;
    vthost.push_back(thost0);
    vthost.push_back(thost1);
    vthost.push_back(thost2);
    vthost.push_back(thost3);
    vthost.push_back(thost4);
    vthost.push_back(thost5);
    vtdev.push_back(tdev0);
    vtdev.push_back(tdev1);
    vtdev.push_back(tdev2);
    vtdev.push_back(tdev3);
    vtdev.push_back(tdev4);
    vtdev.push_back(tdev5);
    print_tensor(vthost[5]);
    print_tensor(vtdev[5]);
    cudaDeviceSynchronize();

    //! test share_from function, if targets are the same, buffer is shared, otherwise, buffer is copied
    LOG(INFO) << "test share_from function";
    TensorH thost6(Dtype), thost7(Dtype);
    TensorD tdev6(Dtype), tdev7(Dtype);
    thost6.set_shape(thost4.shape());
    thost7.set_shape(thost4.shape());
    tdev6.set_shape(thost4.shape());
    tdev7.set_shape(thost4.shape());
    Shape sh2({1, 2, 2, 2}, Layout_NCHW);
    Shape offset({0, 0, 1, 1}, Layout_NCHW);
    LOG(INFO) << "|--shared host";
    thost6.share_sub_buffer(thost4, sh2, offset);
    LOG(INFO) << "|--copied host";
    tdev6.reshape(sh1);
    tdev6.copy_from(thost4);
    LOG(INFO) << "|--copied device";
    thost7.reshape(sh1);
    thost7.copy_from(tdev4);
    LOG(INFO) << "|--shared device";
    tdev7.share_from(tdev4);

    LOG(INFO) << "|--change data in shared tensor";

    //Shape sh_real = thost6.shape();
    //Shape sh_act = thost6.valid_shape();
    //Shape offset_act = thost6.offset();

    //int start_w = offset_act[3];
    //int start_h = offset_act[2];
    //int start_c = offset_act[1];
    //int start_n = offset_act[0];
    //int stride_h = sh_real.count(3);
    //int stride_c = sh_real.count(2);
    //int stride_n = sh_real.count(1);
    //int stride_n = sh_real.count(0);
    Shape stride = thost6.get_stride();
    int w = thost6.width();
    int h = thost6.height();
    int c = thost6.channel();
    int n = thost6.num();

    dtype* ptr_host = (dtype*)thost6.mutable_data() + thost6.data_offset();

    for (int in = 0; in < n; ++in) {
        dtype* ptr_batch = ptr_host + in * stride[0];

        for (int ic = 0; ic < c; ++ic) {
            dtype* ptr_channel = ptr_batch + ic * stride[1];

            for (int ih = 0; ih < h; ++ih) {
                dtype* ptr_row = ptr_channel + ih * stride[2];

                for (int iw = 0; iw < w; ++iw) {
                    ptr_row[iw] = 1.f;
                }
            }
        }
    }

    LOG(INFO) << "|--show root tensor while data is changed by shared tensor";
    print_tensor(thost4);

    //! test record tensor event
    LOG(INFO) << "test record tensor event";
    NV_API::stream_t dev_stream;
    NV_API::stream_t dev_stream1;
    NV_API::create_stream_with_flag(&dev_stream, 1);
    NV_API::create_stream_with_flag(&dev_stream1, 1);
    X86_API::stream_t host_stream;
    X86_API::create_stream_with_flag(&host_stream, 1);
    LOG(INFO) << "|--test record event on host tensor";
    fill_tensor_const(thost4, 63.f);
    thost4.record_event(host_stream);
    thost4.sync();
    print_tensor(thost4);
    LOG(INFO) << "|--test record event on device tensor";
    fill_tensor_const(tdev4, 127.f, dev_stream);
    tdev4.record_event(dev_stream);
    tdev4.sync();
    print_tensor(tdev4, dev_stream1);
    tdev4.record_event(dev_stream1);
    tdev4.sync();
#if 0
    TensorD td;
    Shape sh({1, 3, 10, 10}, Layout_NCHW);
    td.re_alloc(sh, AK_FLOAT);
    NV_API::stream_t stream00, stream01;
    NV_API::create_stream(&stream00);
    NV_API::create_stream(&stream01);
    fill_tensor_const(td, 666);
    cudaDeviceSynchronize();
    print_tensor(td, stream00);
    td.record_event(stream00);
    //! comment the flowing line and turn off cudaDeviceSynchronize in print_tensor_device will print wrong result
    //td.sync();
    fill_tensor_const(td, 888, stream01);
    cudaDeviceSynchronize();
#endif
}

TEST(TestSaberFuncNV, test_tensor_constructor) {
    LOG(INFO) << "test FP32 tensor";
    tensor_constructor<AK_FLOAT, float>();
    LOG(INFO) << "test INT8 tensor";
    tensor_constructor<AK_INT8, char>();
}

#if 1
template <DataType Dtype, typename dtype>
void tensor_deepcopy() {
    //! tensor constructor with alloc data, if target is different, create buffer, and copy the data
    LOG(INFO) << "test tensor deep copy";
    Shape sh0({2, 2, 4, 4}, Layout_NCHW);
    Shape va_sh0({2, 2, 2, 2}, Layout_NCHW);
    Shape off_sh0({0, 0, 1, 1}, Layout_NCHW);

    Shape sh1({2, 2, 4, 4}, Layout_NCHW);
    Shape va_sh1(va_sh0);
    Shape off_sh1({0, 0, 1, 0}, Layout_NCHW);

    Shape sh2({2, 32}, Layout_NW);
    Shape va_sh2({2, 8}, Layout_NW);
    Shape off_sh2({0, 8}, Layout_NW);

    X86_API::stream_t x86_stream;
    NV_API::stream_t nv_stream;
    X86_API::create_stream(&x86_stream);
    NV_API::create_stream(&nv_stream);

    //! create source tensor, th0, td0, th01, td01, th1, td1;
    TensorH th0(sh0, Dtype);
    dtype* ptr0 = (dtype*)th0.mutable_data();

    for (int i = 0; i < sh0.count(); ++i) {
        ptr0[i] = i;
    }

    TensorH th1(va_sh0, Dtype);
    dtype* ptr1 = (dtype*)th1.mutable_data();
    for (int i = 0; i < va_sh0.count(); ++i) {
        ptr1[i] = i;
    }

    TensorH th01(Dtype);
    th01.share_sub_buffer(th0, va_sh0, off_sh0);

    TensorD td0(Dtype), td1(Dtype), td01(Dtype);
    td0.set_shape(th0.shape());
    td1.set_shape(th1.shape());
    td0.copy_from(th0);
    td1.copy_from(th1);
    TensorD dev_tmp0(Dtype);
    dev_tmp0.set_shape(th0.shape());
    dev_tmp0.copy_from(th0);
    td01.share_sub_buffer(dev_tmp0, va_sh0, off_sh0);

    print_tensor(th0);
    print_tensor(th1);
    print_tensor(td0);
    print_tensor(td1);

    //! create th2, th3, th21, td2, td3, td21 as dst tensor
    TensorH th2(sh2, Dtype);
    fill_tensor_const(th2, 0.f);
    TensorH th21(Dtype);
    th21.share_sub_buffer(th2, va_sh2, off_sh2);
    TensorH th3(va_sh2, Dtype);

    TensorD td2(sh2, Dtype);
    fill_tensor_const(td2, 0.f);
    cudaDeviceSynchronize();
    TensorD td21(Dtype);
    td21.share_sub_buffer(td2, va_sh2, off_sh2);
    TensorD td3(va_sh2, Dtype);

    double max_diff;
    double  max_ratio;
    //! test tensor deep copy, entire buffer copy
    LOG(INFO) << "test tensor deep copy, entire buffer copy, H2H";
    th3.copy_from(th1);
    print_tensor(th3);
    tensor_cmp_host((const dtype*)th1.data(), (const dtype*)th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, H2H";
    fill_tensor_const(th3, 0.f);
    th3.async_copy_from(th1, x86_stream);
    th3.record_event(x86_stream);
    th3.sync();
    tensor_cmp_host((const dtype*)th1.data(), (const dtype*)th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, async, H2H";

    LOG(INFO) << "test tensor deep copy, entire buffer copy, D2H";
    th3.copy_from(td1);
    print_tensor(th3);
    tensor_cmp_host((const dtype*)th1.data(), (const dtype*)th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, D2H";
    fill_tensor_const(th3, 0.f);
    th3.async_copy_from(td1, x86_stream);
    th3.record_event(x86_stream);
    th3.sync();
    tensor_cmp_host((const dtype*)th1.data(), (const dtype*)th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, async, D2H";

    LOG(INFO) << "test tensor deep copy, entire buffer copy, H2D";
    td3.copy_from(th1);
    print_tensor(td3);
    cudaDeviceSynchronize();
    tensor_cmp_host((const dtype*)th1.data(), (const dtype*)th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, D2H";
    fill_tensor_const(td3, 0.f);
    cudaDeviceSynchronize();
    td3.async_copy_from(th1, nv_stream);
    td3.record_event(nv_stream);
    td3.sync();
    tensor_cmp_host((const dtype*)th1.data(), (const dtype*)th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, async, D2H";

    LOG(INFO) << "test tensor deep copy, entire buffer copy, D2D";
    td3.copy_from(td1);
    print_tensor(td3);
    cudaDeviceSynchronize();
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, D2D";
    fill_tensor_const(td3, 0.f);
    cudaDeviceSynchronize();
    td3.async_copy_from(td1, nv_stream);
    td3.record_event(nv_stream);
    td3.sync();
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, async, D2D";


    //! test tensor deep copy, src with roi
    LOG(INFO) << "test tensor deep copy, src with roi, H2H";
    th3.copy_from(th01);
    print_tensor(th3);

    LOG(INFO) << "test tensor deep copy, src with roi, D2H";
    th3.copy_from(td01);
    print_tensor(th3);

    LOG(INFO) << "test tensor deep copy, src with roi, H2D";
    td3.copy_from(th01);
    print_tensor(td3);
    cudaDeviceSynchronize();

    LOG(INFO) << "test tensor deep copy, src with roi, D2D";
    td3.copy_from(td01);
    print_tensor(td3);
    cudaDeviceSynchronize();


    //! test tensor deep copy, dst with roi
    LOG(INFO) << "test tensor deep copy, dst with roi, H2H";
    print_tensor(th21);
    print_tensor(th1);
    th21.copy_from(th1);
    print_tensor(th21);

    LOG(INFO) << "test tensor deep copy, dst with roi, D2H";
    th21.copy_from(td1);
    print_tensor(th21);

    LOG(INFO) << "test tensor deep copy, dst with roi, H2D";
    td21.copy_from(th1);
    print_tensor(td21);
    cudaDeviceSynchronize();

    LOG(INFO) << "test tensor deep copy, dst with roi, D2D";
    td21.copy_from(td1);
    print_tensor(td21);
    cudaDeviceSynchronize();


    //! test tensor deep copy, src and dst are with roi
    LOG(INFO) << "test tensor deep copy, src and dst are with roi, H2H";
    th21.copy_from(th01);
    print_tensor(th21);

    LOG(INFO) << "test tensor deep copy, src and dst are with roi, D2H";
    th21.copy_from(td01);
    print_tensor(th21);

    LOG(INFO) << "test tensor deep copy, src and dst are with roi, H2D";
    td21.copy_from(th01);
    print_tensor(td21);
    cudaDeviceSynchronize();

    LOG(INFO) << "test tensor deep copy, src and dst are with roi, D2D";
    td21.copy_from(td01);
    print_tensor(td21);
    cudaDeviceSynchronize();
}

TEST(TestSaberFuncNV, test_tensor_deepcopy) {
    LOG(INFO) << "test FP32 tensor deep copy";
    tensor_deepcopy<AK_FLOAT, float>();
    LOG(INFO) << "test INT8 tensor deep copy";
    tensor_deepcopy<AK_INT8, char>();
}

#endif
#if 1
TEST(TestSaberFuncNV, test_tensor_shape) {
    typedef Tensor<NVHX86> Tensor4_0;
    typedef Tensor<NVHX86> Tensor4_1;
    typedef Tensor<NVHX86> Tensor2;

    int nin = 2;
    int cin = 4;
    int hin = 4;
    int win = 4;

    LOG(INFO) << "test tensor interface";

    Tensor4_0 t1(Shape({nin, cin, hin, win}, Layout_NCHW));
    Tensor4_1 t2(Shape({nin, hin, win, cin}, Layout_NHWC));
    Tensor2 t3(Shape({hin, win}, Layout_HW));

    LOG(INFO) << "test tensor with layout of NCHW";
    LOG(INFO) << "num: " << t1.num() << ", num idx: " << t1.num_index() << \
              ", channel: " << t1.channel() << ", channel idx: " << t1.channel_index() << \
              ", height: " << t1.height() << ", height idx: " << t1.height_index() << \
              ", widhth: " << t1.width() << ", width idx: " << t1.width_index();

    CHECK_EQ(t1.num(), nin) << "NCHW get num error";
    CHECK_EQ(t1.channel(), cin) << "NCHW get channel error";
    CHECK_EQ(t1.height(), hin) << "NCHW get height error";
    CHECK_EQ(t1.width(), win) << "NCHW get width error";

    CHECK_EQ(t1.num_index(), 0) << "NCHW get num index error";
    CHECK_EQ(t1.channel_index(), 1) << "NCHW get channel index error";
    CHECK_EQ(t1.height_index(), 2) << "NCHW get height index error";
    CHECK_EQ(t1.width_index(), 3) << "NCHW get width index error";

    LOG(INFO) << "test tensor with layout of NHWC";
    LOG(INFO) << "num: " << t2.num() << ", num idx: " << t2.num_index() << \
              ", channel: " << t2.channel() << ", channel idx: " << t2.channel_index() << \
              ", height: " << t2.height() << ", height idx: " << t2.height_index() << \
              ", widhth: " << t2.width() << ", width idx: " << t2.width_index();

    CHECK_EQ(t2.num(), nin) << "NHWC get num error";
    CHECK_EQ(t2.channel(), cin) << "NHWC get channel error";
    CHECK_EQ(t2.height(), hin) << "NHWC get height error";
    CHECK_EQ(t2.width(), win) << "NHWC get width error";

    CHECK_EQ(t2.num_index(), 0) << "NHWC get num index error";
    CHECK_EQ(t2.channel_index(), 3) << "NHWC get channel index error";
    CHECK_EQ(t2.height_index(), 1) << "NHWC get height index error";
    CHECK_EQ(t2.width_index(), 2) << "NHWC get width index error";

    LOG(INFO) << "test tensor with layout of HW";
    LOG(INFO) << "num: " << t3.num() << ", num idx: " << t3.num_index() << \
              ", channel: " << t3.channel() << ", channel idx: " << t3.channel_index() << \
              ", height: " << t3.height() << ", height idx: " << t3.height_index() << \
              ", widhth: " << t3.width() << ", width idx: " << t3.width_index();

    CHECK_EQ(t3.num(), 1) << "HW get num error";
    CHECK_EQ(t3.channel(), 1) << "HW get channel error";
    CHECK_EQ(t3.height(), hin) << "HW get height error";
    CHECK_EQ(t3.width(), win) << "HW get width error";

    CHECK_EQ(t3.num_index(), -1) << "HW get num index error";
    CHECK_EQ(t3.channel_index(), -1) << "HW get channel index error";
    CHECK_EQ(t3.height_index(), 0) << "HW get height index error";
    CHECK_EQ(t3.width_index(), 1) << "HW get width index error";

    t1.set_layout(Layout_NCHW_C4);
    CHECK_EQ(t1.num(), nin) << "NCHW get num error";
    CHECK_EQ(t1.channel(), cin) << "NCHW get channel error";
    CHECK_EQ(t1.height(), hin) << "NCHW get height error";
    CHECK_EQ(t1.width(), win) << "NCHW get width error";

    CHECK_EQ(t1.num_index(), 0) << "NCHW get num index error";
    CHECK_EQ(t1.channel_index(), 1) << "NCHW get channel index error";
    CHECK_EQ(t1.height_index(), 2) << "NCHW get height index error";
    CHECK_EQ(t1.width_index(), 3) << "NCHW get width index error";


}
#endif
#if 1
template <DataType Dtype, typename dtype>
void tensor_reshape_realloc() {

    LOG(INFO) << "test tensor reshape and re_alloc funcs";

    Shape sh0({2, 2, 2, 2}, Layout_NCHW);
    Shape sh1({2, 2, 4, 4}, Layout_NCHW);
    TensorH th0(sh1, Dtype);
    TensorD td0(sh1, Dtype);
    fill_tensor_const(th0, 1);
    fill_tensor_const(td0, 1);
    LOG(INFO) << "ori tensor with size: " << th0.valid_size();
    print_tensor(th0);
    print_tensor(td0);
    cudaDeviceSynchronize();

    th0.reshape(sh0);
    td0.reshape(sh0);
    LOG(INFO) << "tensor after reshape(from big space to small) with size: " << th0.valid_size();
    print_tensor(th0);
    print_tensor(td0);
    cudaDeviceSynchronize();
    fill_tensor_const(th0, 1);
    fill_tensor_const(td0, 1);
    cudaDeviceSynchronize();

    th0.reshape(sh1);
    td0.reshape(sh1);
    LOG(INFO) << "tensor after reshape(from small to big, not larger than ori) with size: " <<
              th0.valid_size();
    print_tensor(th0);
    print_tensor(td0);
    cudaDeviceSynchronize();

    th0.re_alloc(sh0, Dtype);
    td0.re_alloc(sh0, Dtype);
    LOG(INFO) << "tensor after re_alloc(from big space to small) with size: " << th0.valid_size();
    print_tensor(th0);
    print_tensor(td0);
    cudaDeviceSynchronize();

    TensorH th1(sh0, Dtype);
    TensorD td1(sh0, Dtype);
    LOG(INFO) << "ori tensor with size: " << th1.valid_size();
    fill_tensor_const(th1, 1);
    fill_tensor_const(td1, 1);
    cudaDeviceSynchronize();
    print_tensor(th1);
    print_tensor(td1);
    cudaDeviceSynchronize();

    th1.reshape(sh1);
    td1.reshape(sh1);
    LOG(INFO) << "tensor after reshape(from small space to big) with size: " << th1.valid_size();
    //printf("real_shape: %d,%d, %d, %d, valid_shape: %d, %d, %d, %d\n", \
    th1.shape()[0], th1.shape()[1], th1.shape()[2], th1.shape()[3], \
    th1.valid_shape()[0], th1.valid_shape()[1], th1.valid_shape()[2], th1.valid_shape()[3]);
    print_tensor(th1);
    print_tensor(td1);
    cudaDeviceSynchronize();
    fill_tensor_const(th1, 1);
    fill_tensor_const(td1, 1);
    cudaDeviceSynchronize();

    th1.reshape(sh0);
    td1.reshape(sh0);

    LOG(INFO) << "tensor after re_alloc(from small space to big) with size: " << th1.valid_size();
    th1.re_alloc(sh1, Dtype);
    td1.re_alloc(sh1, Dtype);
    print_tensor(th1);
    print_tensor(td1);
    cudaDeviceSynchronize();

}

TEST(TestSaberFuncNV, test_tensor_reshape_realloc) {
    LOG(INFO) << "FP32 Tensor realloc";
    tensor_reshape_realloc<AK_FLOAT, float>();
    LOG(INFO) << "INT8 Tensor realloc";
    tensor_reshape_realloc<AK_INT8, char>();
}
#endif
#if 1
template <DataType Dtype, typename dtype>
void test_tensor_op() {
    Shape sh({1, 2, 2, 10}, Layout_NCHW);
    TensorD td1(sh, Dtype);
    TensorH th1(sh, Dtype);

    LOG(INFO) << "testing host fill tensor with const 1.";
    fill_tensor_const(th1, 1.f);
    print_tensor(th1);

    LOG(INFO) << "testing device fill tensor with const 1.";
    fill_tensor_const(td1, 1.f);
    print_tensor(td1);

    LOG(INFO) << "testing host fill tensor with rand";
    fill_tensor_rand(th1);
    print_tensor(th1);

    LOG(INFO) << "testing device fill tensor with rand";
    fill_tensor_rand(td1);
    print_tensor(td1);

    LOG(INFO) << "testing host fill tensor with rand from 1 to 10";
    fill_tensor_rand(th1, 1, 10);
    print_tensor(th1);

    LOG(INFO) << "testing device fill tensor with rand from 1 to 10";
    fill_tensor_rand(td1, 1, 10);
    print_tensor(td1);
}
TEST(TestSaberFuncNV, test_tensor_ops) {
    LOG(INFO) << "test tensor op FP32";
    test_tensor_op<AK_FLOAT, float>();
    LOG(INFO) << "test tensor op INT8";
    test_tensor_op<AK_INT8, char>();
}

#endif
#if 1
TEST(TestSaberFuncNV, test_tensor_share_diff_dtype) {
    Shape sh({1, 1, 2, 10}, Layout_NCHW);
    Tensor<NV> td1(sh, AK_FLOAT);
    Tensor<NVHX86> th1(sh, AK_FLOAT);
    Tensor<NV> td2(AK_INT8);
    Tensor<NVHX86> th2(AK_INT8);
    td2.set_shape(sh);
    th2.set_shape(sh);
    LOG(INFO) << "testing host fill tensor with const 1.";
    fill_tensor_const(th1, -1);
    LOG(INFO) << "data type: float";
    print_tensor(th1);
    fill_tensor_const(td1, -1);
    print_tensor(td1);
    cudaDeviceSynchronize();
    
    LOG(INFO) << "INT8 Tensor shared from FP32 tensor";
    td2.share_from(td1);
    th2.share_from(th1);

    print_tensor(th2);
    print_tensor(td2);
    cudaDeviceSynchronize();
}
#endif
int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

