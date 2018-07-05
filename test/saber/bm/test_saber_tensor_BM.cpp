#include "test_saber_tensor_BM.h"
#include "tensor_op.h"
#include <vector>
using namespace anakin::saber;

typedef TargetWrapper<X86> X86_API;
typedef TargetWrapper<BM> BM_API;
typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<BM, AK_BM, NCHW> TensorDf4;
typedef TensorHf4::Dtype dtype;
typedef TensorDf4::Dtype dtype2;


static bm_handle_t handle;
TEST(TestSaberTensorBM, test_tensor_constructor) {
    bmdnn_init(&handle);

    //! test empty constructor
    LOG(INFO) << "test default (empty) constructor";
    TensorHf4 thost0;
    TensorDf4 tdev0;

    //! test tensor re_alloc function empty constructor
    Shape sh0(2, 2, 8, 8);
    LOG(INFO) << "|--test tensor re_alloc function on empty tensor";
    thost0.re_alloc(sh0);
    tdev0.re_alloc(sh0);
    LOG(INFO) << "|--tensor size of host: " << thost0.size();
    LOG(INFO) << "|--tensor size of device: " << tdev0.size();
    CHECK_EQ(thost0.size(), 256) << "error with tensor size";
    CHECK_EQ(tdev0.size(), 256) << "error with tensor size";

    //! test tensor re_alloc function on tensor with data
    LOG(INFO) << "|--test tensor re_alloc function on tensor with data";
    Shape sh1(1, 4, 4, 4);
    thost0.re_alloc(sh1);
    tdev0.re_alloc(sh1);
    LOG(INFO) << "|--tensor size of host: " << thost0.size();
    LOG(INFO) << "|--tensor size of device: " << tdev0.size();
    CHECK_EQ(thost0.size(), 64) << "error with tensor size";
    CHECK_EQ(tdev0.size(), 64) << "error with tensor size";

    //! test tensor shape() function
    LOG(INFO) << "|--test tensor shape() function";
    Shape sho = thost0.shape();
    LOG(INFO) << "|--shape of tensor: " << sho[0] << ", " << sho[1] << "," << sho[2] << "," << sho[3];
    LOG(INFO) << "|--test get tensor n, c, h, w function, num = " \
              << thost0.num() << ", channel = " << thost0.channel() << ", height = " \
              << thost0.height() << ", width = " << thost0.width();

    //! test tensor mutable_data() function
    LOG(INFO) << "|--xxxxxxxxtest tensor mutable_data() function, write tensor data buffer with 2.f";
    fill_tensor_host_const(thost0, 2.f);
    LOG(INFO) << "|--test tensor data() function, show the const data, 2.f";
    print_tensor_host(thost0);

    //! test tensor constructor with shape
    LOG(INFO) << "test tensor constructor with shape";
    TensorHf4 thost1(sh1);
    TensorDf4 tdev1(sh1);

    //! test tensor copy_from() function
    LOG(INFO) << "test copy_from() function, input tensor could be any target";

    // host to host
    thost1.copy_from(thost0);
    print_tensor_host(thost1);

    // host to device
    tdev1.copy_from(thost0);
    print_tensor_device(tdev1);

    // device to host
    thost1.copy_from(tdev1);
    print_tensor_host(thost1);

    LOG(INFO) << "test copy_from() function device to device";

    tdev1.copy_from(tdev0);
    print_tensor_device(tdev1);

    
    //! test tensor constructor with shape and real_shape
    LOG(INFO) << "test tensor constructor with shape and real_shape";
    //! constructor with 3 shapes is removed
    TensorHf4 thost2(sh0);
    TensorDf4 tdev2(sh0);

    //! test tensor constructor with data, if target is different, create buffer, and copy the data
    LOG(INFO) <<
              "test tensor constructor with data, if target is different, create buffer, and copy the data";
    dtype* host_data_ptr;
    dtype2* dev_data_ptr;
    void* tmp_pt_host;
    void* tmp_pt_dev;
    X86_API::mem_alloc(&tmp_pt_host, sizeof(dtype) * sh1.count());
    host_data_ptr = static_cast<dtype*>(tmp_pt_host);

    for (int i = 0; i < sh1.count(); ++i) {
        host_data_ptr[i] = i;
    }

    BM_API::mem_alloc(&tmp_pt_dev, sizeof(dtype2) * sh1.count());
    dev_data_ptr = static_cast<dtype2*>(tmp_pt_dev);
//---    cudaMemcpy(dev_data_ptr, host_data_ptr, sizeof(dtype) * sh1.count(), cudaMemcpyHostToDevice);
    BM_API::sync_memcpy(dev_data_ptr,0,host_data_ptr,0,0,__HtoD());
    LOG(INFO) << "|--construct host tensor from host data ptr";
    TensorHf4 thost3(host_data_ptr, X86(), X86_API::get_device_id(), sh1);
    LOG(INFO) << "|--constructor device tensor from host data ptr";

//    TensorDf4 tdev3(&bm_mem_from_system(const_cast<float *>(host_data_ptr)), X86(), X86_API::get_device_id(), sh1);

    TensorDf4 tdev3(host_data_ptr, X86(), X86_API::get_device_id(), sh1);


    print_tensor_host(thost3);

    print_tensor_device(tdev3);

//    TensorHf4 thost_lian(sh1);
//    thost_lian.copy_from(tdev3);
//    print_tensor_host(thost_lian);
//
//    thost_lian.copy_from(thost3);
//    print_tensor_host(thost_lian);

    //cudaDeviceSynchronize();
    //

    LOG(INFO) << "|--construct host tensor from device data ptr";
    TensorHf4 thost4(host_data_ptr, X86(), X86_API::get_device_id(), sh1);

    TensorDf4 tdev4(host_data_ptr, X86(), X86_API::get_device_id(), sh1);

//    TensorDf4 tdev3(&bm_mem_from_system(const_cast<float *>(host_data_ptr)), X86(), X86_API::get_device_id(), sh1);

//    TensorHf4 thost4(dev_data_ptr, BM(), BM_API::get_device_id(), sh1);
//    LOG(INFO) << "|--constructor device tensor from device data ptr";
//    TensorDf4 tdev4(dev_data_ptr, BM(), BM_API::get_device_id(), sh1);
//    print_tensor_host(thost4);
//    print_tensor_device(tdev4);


    //BM_API::stream_t dev_stream0;
    //BM_API::create_stream_with_flag(dev_stream0, 1);
    //cudaDeviceSynchronize();

    //! test tensor copy constructor
    LOG(INFO) << "test tensor copy constructor";
    LOG(INFO) << "|--normal copy constructor";
    TensorHf4 thost5(thost4);
    TensorDf4 tdev5(tdev4);

    LOG(INFO) << "|--push back to vector";
    std::vector<TensorHf4> vthost;
    std::vector<TensorDf4> vtdev;
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
    print_tensor_host(vthost[5]);
    print_tensor_device(vtdev[5]);
    //cudaDeviceSynchronize();

    //! test share_from function, if targets are the same, buffer is shared, otherwise, buffer is copied
    LOG(INFO) << "test share_from function";
    TensorHf4 thost6, thost7;
    TensorDf4 tdev6, tdev7;
    thost6.set_shape(thost4.shape());
    thost7.set_shape(thost4.shape());
    tdev6.set_shape(thost4.shape());
    tdev7.set_shape(thost4.shape());
    Shape sh2(1, 2, 2, 2);
    Shape offset(0, 0, 1, 1);
    LOG(INFO) << "|--shared host";

    thost6.share_sub_buffer(thost4, sh2, offset);

    LOG(INFO) << "|--copied host";
    tdev6.share_from(thost4);
    LOG(INFO) << "|--copied device";
    thost7.share_from(tdev4);
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

    dtype* ptr_host = thost6.mutable_data();

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
    print_tensor_host(thost4);
    bmdnn_deinit(handle);
}

/*
TEST(TestSaberTensorBM, test_tensor_deepcopy) {
    //! tensor constructor with alloc data, if target is different, create buffer, and copy the data
    LOG(INFO) << "test tensor deep copy";
    Shape sh0(2, 2, 4, 4);
    Shape va_sh0(2, 2, 2, 2);
    Shape off_sh0(0, 0, 1, 1);

    Shape sh1(2, 2, 4, 4);
    Shape va_sh1(va_sh0);
    Shape off_sh1(0, 0, 1, 0);

    Shape sh2(2, 32);
    Shape va_sh2(2, 8);
    Shape off_sh2(0, 8);

    X86_API::stream_t x86_stream;
    BM_API::stream_t nv_stream;
    X86_API::create_stream(x86_stream);
    BM_API::create_stream(nv_stream);

    //! create source tensor, th0, td0, th01, td01, th1, td1;
    TensorHf4 th0(sh0);

    for (int i = 0; i < sh0.count(); ++i) {
        th0.mutable_data()[i] = i;
    }

    TensorHf4 th1(va_sh0);

    for (int i = 0; i < va_sh0.count(); ++i) {
        th1.mutable_data()[i] = i;
    }

    TensorHf4 th01;
    th01.share_sub_buffer(th0, va_sh0, off_sh0);

    TensorDf4 td0, td1, td01;
    td0.set_shape(th0.shape());
    td1.set_shape(th1.shape());
    td0.share_from(th0);
    td1.share_from(th1);
    TensorDf4 dev_tmp0;
    dev_tmp0.set_shape(th0.shape());
    dev_tmp0.share_from(th0);
    td01.share_sub_buffer(dev_tmp0, va_sh0, off_sh0);

    print_tensor_host(th0);
    print_tensor_host(th1);
    print_tensor_device(td0);
    print_tensor_device(td1);

    //! create th2, th3, th21, td2, td3, td21 as dst tensor
    TensorHf2 th2(sh2);
    fill_tensor_host_const(th2, 0.f);
    TensorHf2 th21;
    th21.share_sub_buffer(th2, va_sh2, off_sh2);
    TensorHf2 th3(va_sh2);

    TensorDf2 td2(sh2);
    fill_tensor_device_const(td2, 0.f);
    //cudaDeviceSynchronize();
    TensorDf2 td21;
    td21.share_sub_buffer(td2, va_sh2, off_sh2);
    TensorDf2 td3(va_sh2);

    double max_diff;
    double  max_ratio;
    //! test tensor deep copy, entire buffer copy
    LOG(INFO) << "test tensor deep copy, entire buffer copy, H2H";
    th3.copy_from(th1);
    print_tensor_host(th3);
    tensor_cmp_host(th1.data(), th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, H2H";
    fill_tensor_host_const(th3, 0.f);
    th3.async_copy_from(th1, x86_stream);
    th3.record_event(x86_stream);
    th3.sync();
    tensor_cmp_host(th1.data(), th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, async, H2H";

    LOG(INFO) << "test tensor deep copy, entire buffer copy, D2H";
    th3.copy_from(td1);
    print_tensor_host(th3);
    tensor_cmp_host(th1.data(), th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, D2H";
    fill_tensor_host_const(th3, 0.f);
    th3.async_copy_from(td1, nv_stream);
    th3.record_event(x86_stream);
    th3.sync();
    tensor_cmp_host(th1.data(), th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, async, D2H";

    LOG(INFO) << "test tensor deep copy, entire buffer copy, H2D";
    td3.copy_from(th1);
    print_tensor_device(td3);
    //cudaDeviceSynchronize();
    tensor_cmp_host(th1.data(), th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, D2H";
    fill_tensor_device_const(td3, 0.f);
    //cudaDeviceSynchronize();
    td3.async_copy_from(th1, nv_stream);
    td3.record_event(nv_stream);
    td3.sync();
    tensor_cmp_host(th1.data(), th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, async, D2H";

    LOG(INFO) << "test tensor deep copy, entire buffer copy, D2D";
    td3.copy_from(td1);
    print_tensor_device(td3);
    //cudaDeviceSynchronize();
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, D2D";
    fill_tensor_device_const(td3, 0.f);
    //cudaDeviceSynchronize();
    td3.async_copy_from(td1, nv_stream);
    td3.record_event(nv_stream);
    td3.sync();
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, async, D2D";


    //! test tensor deep copy, src with roi
    LOG(INFO) << "test tensor deep copy, src with roi, H2H";
    th3.copy_from(th01);
    print_tensor_host(th3);

    LOG(INFO) << "test tensor deep copy, src with roi, D2H";
    th3.copy_from(td01);
    print_tensor_host(th3);

    LOG(INFO) << "test tensor deep copy, src with roi, H2D";
    td3.copy_from(th01);
    print_tensor_device(td3);
    //cudaDeviceSynchronize();

    LOG(INFO) << "test tensor deep copy, src with roi, D2D";
    td3.copy_from(td01);
    print_tensor_device(td3);
    //cudaDeviceSynchronize();


    //! test tensor deep copy, dst with roi
    LOG(INFO) << "test tensor deep copy, dst with roi, H2H";
    print_tensor_host(th21);
    print_tensor_host(th1);
    th21.copy_from(th1);
    print_tensor_host(th21);

    LOG(INFO) << "test tensor deep copy, dst with roi, D2H";
    th21.copy_from(td1);
    print_tensor_host(th21);

    LOG(INFO) << "test tensor deep copy, dst with roi, H2D";
    td21.copy_from(th1);
    print_tensor_device(td21);
    //cudaDeviceSynchronize();

    LOG(INFO) << "test tensor deep copy, dst with roi, D2D";
    td21.copy_from(td1);
    print_tensor_device(td21);
    //cudaDeviceSynchronize();


    //! test tensor deep copy, src and dst are with roi
    LOG(INFO) << "test tensor deep copy, src and dst are with roi, H2H";
    th21.copy_from(th01);
    print_tensor_host(th21);

    LOG(INFO) << "test tensor deep copy, src and dst are with roi, D2H";
    th21.copy_from(td01);
    print_tensor_host(th21);

    LOG(INFO) << "test tensor deep copy, src and dst are with roi, H2D";
    td21.copy_from(th01);
    print_tensor_device(td21);
    //cudaDeviceSynchronize();

    LOG(INFO) << "test tensor deep copy, src and dst are with roi, D2D";
    td21.copy_from(td01);
    print_tensor_device(td21);
    //cudaDeviceSynchronize();
}*/

TEST(TestSaberTensorBM, test_tensor_shape) {
    typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4_0;
    typedef Tensor<X86, AK_FLOAT, NHWC> Tensor4_1;
    typedef Tensor<X86, AK_FLOAT, HW> Tensor2;

    int nin = 2;
    int cin = 2;
    int hin = 4;
    int win = 4;

    LOG(INFO) << "test tensor interface";

    Tensor4_0 t1(Shape(nin, cin, hin, win));
    Tensor4_1 t2(Shape(nin, hin, win, cin));
    Tensor2 t3(Shape(hin, win));

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

}

TEST(TestSaberTensorBM, test_tensor_reshape_realloc) {

    LOG(INFO) << "test tensor reshape and re_alloc funcs";

    Shape sh0(2, 2, 2, 2);
    Shape sh1(2, 2, 4, 4);
    TensorHf4 th0(sh1);
    TensorDf4 td0(sh1);
    fill_tensor_host_const(th0, 1);
    fill_tensor_device_const(td0, 1);
    LOG(INFO) << "ori tensor with size: " << th0.valid_size();
    print_tensor_host(th0);
    print_tensor_device(td0);
    //cudaDeviceSynchronize();

    th0.reshape(sh0);
    td0.reshape(sh0);
    LOG(INFO) << "tensor after reshape(from big space to small) with size: " << th0.valid_size();
    print_tensor_host(th0);
    print_tensor_device(td0);
    //cudaDeviceSynchronize();
    fill_tensor_host_const(th0, 1);
    fill_tensor_device_const(td0, 1);
    //cudaDeviceSynchronize();

    th0.reshape(sh1);
    td0.reshape(sh1);
    LOG(INFO) << "tensor after reshape(from small to big, not larger than ori) with size: " <<
              th0.valid_size();
    print_tensor_host(th0);
    print_tensor_device(td0);
    //cudaDeviceSynchronize();

    th0.re_alloc(sh0);
    td0.re_alloc(sh0);
    LOG(INFO) << "tensor after re_alloc(from big space to small) with size: " << th0.valid_size();
    print_tensor_host(th0);
    print_tensor_device(td0);
    //cudaDeviceSynchronize();

    TensorHf4 th1(sh0);
    TensorDf4 td1(sh0);
    LOG(INFO) << "ori tensor with size: " << th1.valid_size();
    fill_tensor_host_const(th1, 1);
    fill_tensor_device_const(td1, 1);
    //cudaDeviceSynchronize();
    print_tensor_host(th1);
    print_tensor_device(td1);
    //cudaDeviceSynchronize();

    th1.reshape(sh1);
    td1.reshape(sh1);
    LOG(INFO) << "tensor after reshape(from small space to big) with size: " << th1.valid_size();
    //printf("real_shape: %d,%d, %d, %d, valid_shape: %d, %d, %d, %d\n", \
    th1.shape()[0], th1.shape()[1], th1.shape()[2], th1.shape()[3], \
    th1.valid_shape()[0], th1.valid_shape()[1], th1.valid_shape()[2], th1.valid_shape()[3]);
    print_tensor_host(th1);
    print_tensor_device(td1);
    //cudaDeviceSynchronize();
    fill_tensor_host_const(th1, 1);
    fill_tensor_device_const(td1, 1);
    //cudaDeviceSynchronize();

    th1.reshape(sh0);
    td1.reshape(sh0);

    LOG(INFO) << "tensor after re_alloc(from small space to big) with size: " << th1.valid_size();
    th1.re_alloc(sh1);
    td1.re_alloc(sh1);
    print_tensor_host(th1);
    print_tensor_device(td1);
    //cudaDeviceSynchronize();

}

TEST(TestSaberTensorBM, test_tensor_op) {
    Shape sh{1, 2, 2, 10};
    TensorDf4 td1(sh);
    TensorHf4 th1(sh);
    Tensor<BM, AK_BM, NCHW> td2(sh);
    Tensor<X86, AK_FLOAT, NCHW> th2(sh);
    LOG(INFO) << "testing host fill tensor with const 1.";
    fill_tensor_host_const(th1, 1.f);
    LOG(INFO) << "data type: float";
    print_tensor_host(th1);
    fill_tensor_host_const(th2, 1);
    LOG(INFO) << "data type: int8";
    print_tensor_host(th2);

    LOG(INFO) << "testing device fill tensor with const 1.";
    fill_tensor_device_const(td1, 1.f);
    LOG(INFO) << "data type: float";
    print_tensor_device(td1);
    fill_tensor_device_const(td2, 1);
    LOG(INFO) << "data type: int8";
    print_tensor_device(td2);

    LOG(INFO) << "testing host fill tensor with rand";
    fill_tensor_host_rand(th1);
    LOG(INFO) << "data type: float";
    print_tensor_host(th1);
    fill_tensor_host_rand(th2);
    LOG(INFO) << "data type: int8";
    print_tensor_host(th2);

    LOG(INFO) << "testing device fill tensor with rand";
    fill_tensor_device_rand(td1);
    LOG(INFO) << "data type: float";
    print_tensor_device(td1);
    fill_tensor_device_rand(td2);
    LOG(INFO) << "data type: int8";
    print_tensor_device(td2);

    LOG(INFO) << "testing host fill tensor with rand from 1 to 10";
    fill_tensor_host_rand(th1, 1, 10);
    LOG(INFO) << "data type: float";
    print_tensor_host(th1);
    fill_tensor_host_rand(th2, 1, 10);
    LOG(INFO) << "data type: int8";
    print_tensor_host(th2);

    LOG(INFO) << "testing device fill tensor with rand from 1 to 10";
    fill_tensor_device_rand(td1, 1, 10);
    LOG(INFO) << "data type: float";
    print_tensor_device(td1);
    fill_tensor_device_rand(td2, 1, 10);
    LOG(INFO) << "data type: int8";
    print_tensor_device(td2);
}

TEST(TestSaberTensorBM, test_tensor_share_diff_dtype) {
    Shape sh{1, 1, 2, 10};
    Tensor<BM, AK_BM, NCHW> td1(sh);
    Tensor<X86, AK_FLOAT, NCHW> th1(sh);
    Tensor<BM, AK_BM, NCHW> td2;
    Tensor<X86, AK_FLOAT, NCHW> th2;
    td2.set_shape(sh);
    th2.set_shape(sh);
    LOG(INFO) << "testing host fill tensor with const 1.";
    fill_tensor_host_const(th1, -1);
    LOG(INFO) << "data type: float";
    print_tensor_host(th1);
    fill_tensor_device_const(td1, -1);
    LOG(INFO) << "data type: int8";
    print_tensor_device(td1);
    //cudaDeviceSynchronize();

    td2.share_from(td1);
    th2.share_from(th1);

    print_tensor_host(th2);
    print_tensor_device(td2);
    //cudaDeviceSynchronize();
}

TEST(TestSaberTensorBM, test_tensor_base_type) {
    Shape sh(1, 3, 10, 10);
    Tensor<BM, AK_BM, NCHW> td1(sh);
    Tensor<X86, AK_FLOAT, NCHW> th1(sh);
    fill_tensor_host_rand(th1, 0.f, 255.f);
    td1.copy_from(th1);
    TensorBase* tb1;
    TensorBase* tb2;
    tb1 = &th1;
    Shape sh1(1, 1, 10, 10);
    tb1->set_shape(sh1);
    Shape sh11 = th1.valid_shape();
    LOG(INFO) << "base tensor call set shape: " << "n=" << sh11[0] << ", c=" << sh11[1] << \
              ", h=" << sh11[2] << ", w=" << sh11[3];
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

