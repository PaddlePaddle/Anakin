#include "test_saber_func.h"
#include "tensor_op.h"
#include <vector>

using namespace anakin::saber;

template <typename TargetD, typename TargetH, DataType Dtype>
void tensor_constructor() {

    typedef TargetWrapper<TargetH> HAPI;
    typedef TargetWrapper<TargetD> DAPI;

    typedef typename TargetTypeTraits<TargetH>::target_category target_H;
    typedef typename TargetTypeTraits<TargetD>::target_category target_D;
    typedef typename IF<std::is_same<target_D, target_H>::value, __HtoH, __DtoH>::Type then_type;
    typedef typename IF<std::is_same<target_D, target_H>::value, __DtoD, __HtoD>::Type else_type;
    typedef typename IF<std::is_same<target_D, __host_target>::value, then_type, else_type>::Type flag_type;
    typedef typename IF<std::is_same<target_D, __host_target>::value, HAPI, DAPI>::Type copy_API;

    typedef Tensor<TargetH> TensorH;
    typedef Tensor<TargetD> TensorD;

    typedef typename DataTrait<TargetH, Dtype>::Dtype dtype;

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
    DAPI::device_sync();
    thost1.copy_from(tdev1);
    tdev1.copy_from(tdev0);
    print_tensor(thost1);

    //! test tensor constructor with shape and real_shape
    LOG(INFO) << "test tensor constructor with shape and real_shape";
    //! constructor with 3 shapes is removed
    TensorH thost2(sh0, Dtype);
    TensorD tdev2(sh0, Dtype);

    //! test tensor constructor with data, if target is different, create buffer, and copy the data
    LOG(INFO) << "test tensor constructor with data, if target is different, create buffer, and copy the data";
    dtype* host_data_ptr;
    dtype* dev_data_ptr;
    void* tmp_pt_host;
    void* tmp_pt_dev;
    HAPI::mem_alloc(&tmp_pt_host, sizeof(dtype) * sh1.count());
    host_data_ptr = static_cast<dtype*>(tmp_pt_host);

    for (int i = 0; i < sh1.count(); ++i) {
        host_data_ptr[i] = static_cast<dtype>(i);
    }

    DAPI::mem_alloc(&tmp_pt_dev, sizeof(dtype) * sh1.count());
    dev_data_ptr = static_cast<dtype*>(tmp_pt_dev);

    copy_API::sync_memcpy(dev_data_ptr, 0, DAPI::get_device_id(), \
        host_data_ptr, 0, HAPI::get_device_id(), \
        sizeof(dtype) * sh1.count(), flag_type());

    LOG(INFO) << "|--construct host tensor from host data ptr";
    TensorH thost3(host_data_ptr, TargetH(), HAPI::get_device_id(), sh1, Dtype);
    LOG(INFO) << "|--constructor device tensor from host data ptr";
    TensorD tdev3(host_data_ptr, TargetH(), HAPI::get_device_id(), sh1, Dtype);
    print_tensor(thost3);
    print_tensor(tdev3);
    DAPI::device_sync();

    LOG(INFO) << "|--construct host tensor from device data ptr";
    TensorH thost4(dev_data_ptr, TargetD(), DAPI::get_device_id(), sh1, Dtype);
    LOG(INFO) << "|--constructor device tensor from device data ptr";
    TensorD tdev4(dev_data_ptr, TargetD(), DAPI::get_device_id(), sh1, Dtype);
    print_tensor(thost4);
    print_tensor(tdev4);
    typename DAPI::stream_t dev_stream0;
    DAPI::create_stream_with_flag(&dev_stream0, 1);
    DAPI::device_sync();

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
    DAPI::device_sync();

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
                    ptr_row[iw] = static_cast<dtype>(1);
                }
            }
        }
    }

    LOG(INFO) << "|--show root tensor while data is changed by shared tensor";
    print_tensor(thost4);

    //! test record tensor event
    LOG(INFO) << "test record tensor event";
    Context<TargetD> ctx1(DAPI::get_device_id(), 0, 0);
    Context<TargetD> ctx2(DAPI::get_device_id(), 1, 1);
    typename DAPI::stream_t dev_stream1 = ctx1.get_compute_stream();
    typename DAPI::stream_t dev_stream2 = ctx2.get_compute_stream();

    Context<TargetH> ctx3(HAPI::get_device_id(), 0, 0);
    typename HAPI::stream_t host_stream = ctx3.get_compute_stream();

    LOG(INFO) << "|--test record event on host tensor";
    fill_tensor_const(thost4, 63.f);
    thost4.record_event(host_stream);
    thost4.sync();
    print_tensor(thost4);
    LOG(INFO) << "|--test record event on device tensor";
    fill_tensor_const(tdev4, 127.f, dev_stream1);
    tdev4.record_event(dev_stream1);
    tdev4.sync();
    print_tensor(tdev4, dev_stream2);
    tdev4.record_event(dev_stream2);
    tdev4.sync();
#if 0
    TensorD td;
    Shape sh({1, 3, 10, 10}, Layout_NCHW);
    td.re_alloc(sh, AK_FLOAT);
    DAPI::stream_t stream00, stream01;
    DAPI::create_stream(&stream00);
    DAPI::create_stream(&stream01);
    fill_tensor_const(td, 666);
    DAPI::device_sync();
    print_tensor(td, stream00);
    td.record_event(stream00);
    //! comment the flowing line and turn off cudaDeviceSynchronize in print_tensor_device will print wrong result
    //td.sync();
    fill_tensor_const(td, 888, stream01);
    DAPI::device_sync();
#endif
}

TEST(TestSaberFunc, test_tensor_constructor) {

#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    LOG(INFO) << "test CUDA FP32 tensor";
    tensor_constructor<NV, NVHX86, AK_FLOAT>();
    LOG(INFO) << "test CUDA INT8 tensor";
    tensor_constructor<NV, NVHX86, AK_INT8>();
#endif

#ifdef USE_X86_PLACE
    Env<X86>::env_init();
    LOG(INFO) << "test X86 FP32 tensor";
    tensor_constructor<X86, X86, AK_FLOAT>();
    LOG(INFO) << "test X86 INT8 tensor";
    tensor_constructor<X86, X86, AK_INT8>();
#endif

#ifdef USE_ARM_PLACE
    Env<ARM>::env_init();
    LOG(INFO) << "test ARM FP32 tensor";
    tensor_constructor<ARM, ARM, AK_FLOAT>();
    LOG(INFO) << "test ARM INT8 tensor";
    tensor_constructor<ARM, ARM, AK_INT8>();
#endif

#ifdef USE_BM
    Env<BM>::env_init();
    LOG(INFO) << "test BM FP32 tensor";
    tensor_constructor<BM, X86, AK_FLOAT>();
#endif
}

#if 1
template <typename TargetD, typename TargetH, DataType Dtype>
void tensor_deepcopy() {

    typedef TargetWrapper<TargetH> HAPI;
    typedef TargetWrapper<TargetD> DAPI;

    typedef typename TargetTypeTraits<TargetH>::target_category target_H;
    typedef typename TargetTypeTraits<TargetD>::target_category target_D;
    typedef typename IF<std::is_same<target_D, target_H>::value, __HtoH, __DtoH>::Type then_type;
    typedef typename IF<std::is_same<target_D, target_H>::value, __DtoD, __HtoD>::Type else_type;
    typedef typename IF<std::is_same<target_D, __host_target>::value, then_type, else_type>::Type flag_type;
    typedef typename IF<std::is_same<target_D, __host_target>::value, HAPI, DAPI>::Type copy_API;

    typedef Tensor<TargetH> TensorH;
    typedef Tensor<TargetD> TensorD;

    typedef typename DataTrait<TargetH, Dtype>::Dtype dtype;

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

    Context<TargetH> ctxh1(HAPI::get_device_id(), 0, 0);
    Context<TargetD> ctxd1(DAPI::get_device_id(), 0, 0);
    typename HAPI::stream_t x86_stream = ctxh1.get_compute_stream();
    typename DAPI::stream_t nv_stream = ctxd1.get_compute_stream();

    //! create source tensor, th0, td0, th01, td01, th1, td1;
    TensorH th0(sh0, Dtype);
    dtype* ptr0 = (dtype*)th0.mutable_data();

    for (int i = 0; i < sh0.count(); ++i) {
        ptr0[i] = static_cast<dtype>(i);
    }

    TensorH th1(va_sh0, Dtype);
    dtype* ptr1 = (dtype*)th1.mutable_data();
    for (int i = 0; i < va_sh0.count(); ++i) {
        ptr1[i] = static_cast<dtype>(i);
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
    DAPI::device_sync();
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
    DAPI::device_sync();
    tensor_cmp_host((const dtype*)th1.data(), (const dtype*)th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, D2H";
    fill_tensor_const(td3, 0.f);
    DAPI::device_sync();
    td3.async_copy_from(th1, nv_stream);
    td3.record_event(nv_stream);
    td3.sync();
    tensor_cmp_host((const dtype*)th1.data(), (const dtype*)th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, async, D2H";

    LOG(INFO) << "test tensor deep copy, entire buffer copy, D2D";
    td3.copy_from(td1);
    print_tensor(td3);
    DAPI::device_sync();
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, D2D";
    fill_tensor_const(td3, 0.f);
    DAPI::device_sync();
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
    DAPI::device_sync();

    LOG(INFO) << "test tensor deep copy, src with roi, D2D";
    td3.copy_from(td01);
    print_tensor(td3);
    DAPI::device_sync();


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
    DAPI::device_sync();

    LOG(INFO) << "test tensor deep copy, dst with roi, D2D";
    td21.copy_from(td1);
    print_tensor(td21);
    DAPI::device_sync();


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
    DAPI::device_sync();

    LOG(INFO) << "test tensor deep copy, src and dst are with roi, D2D";
    td21.copy_from(td01);
    print_tensor(td21);
    DAPI::device_sync();
}

TEST(TestSaberFunc, test_tensor_deepcopy) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    LOG(INFO) << "test CUDA FP32 tensor deep copy";
    tensor_deepcopy<NV, NVHX86, AK_FLOAT>();
    LOG(INFO) << "test CUDA INT8 tensor deep copy";
    tensor_deepcopy<NV, NVHX86, AK_INT8>();
#endif //USE_CUDA

#ifdef USE_X86_PLACE
    Env<X86>::env_init();
    LOG(INFO) << "test X86 FP32 tensor deep copy";
    tensor_deepcopy<X86, X86, AK_FLOAT>();
    LOG(INFO) << "test X86 INT8 tensor deep copy";
    tensor_deepcopy<X86, X86, AK_INT8>();
#endif //USE_X86_PLACE

#ifdef USE_ARM_PLACE
    Env<ARM>::env_init();
    LOG(INFO) << "test ARM FP32 tensor deep copy";
    tensor_deepcopy<ARM, ARM, AK_FLOAT>();
    LOG(INFO) << "test ARM INT8 tensor deep copy";
    tensor_deepcopy<ARM, ARM, AK_INT8>();
#endif //USE_ARM_PLACE

#ifdef USE_BM
    Env<BM>::env_init();
    LOG(INFO) << "test BM FP32 tensor deep copy";
    //tensor_deepcopy<BM, X86, AK_FLOAT>();
#endif //USE_BM
}
#endif

#if 1
template <typename Target>
void test_tensor_shape() {
    typedef Tensor<Target> Tensor4_0;
    typedef Tensor<Target> Tensor4_1;
    typedef Tensor<Target> Tensor2;

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

TEST(TestSaberFunc, test_saber_tensor_shape) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    LOG(INFO) << "test CUDA tensor shape API";
    test_tensor_shape<NV>();
#endif //USE_CUDA

#ifdef USE_X86_PLACE
    Env<X86>::env_init();
    LOG(INFO) << "test X86 tensor shape API";
    test_tensor_shape<X86>();
#endif //USE_X86_PLACE

#ifdef USE_ARM_PLACE
    Env<ARM>::env_init();
    LOG(INFO) << "test ARM tensor shape API";
    test_tensor_shape<ARM>();
#endif //USE_ARM_PLACE

#ifdef USE_BM
    Env<BM>::env_init();
    LOG(INFO) << "test BM tensor shape API";
    test_tensor_shape<BM>();
#endif //USE_BM
}
#endif

#if 1
template <typename TargetD, typename TargetH, DataType Dtype>
void tensor_reshape_realloc() {

    typedef TargetWrapper<TargetH> HAPI;
    typedef TargetWrapper<TargetD> DAPI;

    typedef typename TargetTypeTraits<TargetH>::target_category target_H;
    typedef typename TargetTypeTraits<TargetD>::target_category target_D;
    typedef typename IF<std::is_same<target_D, target_H>::value, __HtoH, __DtoH>::Type then_type;
    typedef typename IF<std::is_same<target_D, target_H>::value, __DtoD, __HtoD>::Type else_type;
    typedef typename IF<std::is_same<target_D, __host_target>::value, then_type, else_type>::Type flag_type;
    typedef typename IF<std::is_same<target_D, __host_target>::value, HAPI, DAPI>::Type copy_API;

    typedef Tensor<TargetH> TensorH;
    typedef Tensor<TargetD> TensorD;

    typedef typename DataTrait<TargetH, Dtype>::Dtype dtype;

    LOG(INFO) << "test tensor reshape and re_alloc funcs";

    Shape sh0({2, 2, 2, 2}, Layout_NCHW);
    Shape sh1({2, 2, 4, 4}, Layout_NCHW);
    TensorH th0(sh1, Dtype);
    TensorD td0(sh1, Dtype);
    fill_tensor_const(th0, 1);
    fill_tensor_const(td0, 1);
    DAPI::device_sync();
    LOG(INFO) << "ori tensor with size: " << th0.valid_size();
    print_tensor(th0);
    print_tensor(td0);

    th0.reshape(sh0);
    td0.reshape(sh0);
    LOG(INFO) << "tensor after reshape(from big space to small) with size: " << th0.valid_size();
    print_tensor(th0);
    print_tensor(td0);

    fill_tensor_const(th0, 1);
    fill_tensor_const(td0, 1);
    DAPI::device_sync();

    th0.reshape(sh1);
    td0.reshape(sh1);
    LOG(INFO) << "tensor after reshape(from small to big, not larger than ori) with size: " <<
                      th0.valid_size();
    print_tensor(th0);
    print_tensor(td0);

    th0.re_alloc(sh0, Dtype);
    td0.re_alloc(sh0, Dtype);
    LOG(INFO) << "tensor after re_alloc(from big space to small) with size: " << th0.valid_size();
    print_tensor(th0);
    print_tensor(td0);

    TensorH th1(sh0, Dtype);
    TensorD td1(sh0, Dtype);
    LOG(INFO) << "ori tensor with size: " << th1.valid_size();
    fill_tensor_const(th1, 1);
    fill_tensor_const(td1, 1);
    DAPI::device_sync();
    print_tensor(th1);
    print_tensor(td1);

    th1.reshape(sh1);
    td1.reshape(sh1);
    LOG(INFO) << "tensor after reshape(from small space to big) with size: " << th1.valid_size();
    //printf("real_shape: %d,%d, %d, %d, valid_shape: %d, %d, %d, %d\n", \
    th1.shape()[0], th1.shape()[1], th1.shape()[2], th1.shape()[3], \
    th1.valid_shape()[0], th1.valid_shape()[1], th1.valid_shape()[2], th1.valid_shape()[3]);
    print_tensor(th1);
    print_tensor(td1);
    fill_tensor_const(th1, 1);
    fill_tensor_const(td1, 1);

    th1.reshape(sh0);
    td1.reshape(sh0);

    LOG(INFO) << "tensor after re_alloc(from small space to big) with size: " << th1.valid_size();
    th1.re_alloc(sh1, Dtype);
    td1.re_alloc(sh1, Dtype);
    print_tensor(th1);
    print_tensor(td1);

}

TEST(TestSaberFunc, test_tensor_reshape_realloc) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    LOG(INFO) << "test CUDA FP32 tensor reshape realloc";
    tensor_reshape_realloc<NV, NVHX86, AK_FLOAT>();
    LOG(INFO) << "test CUDA INT8 tensor reshape realloc";
    tensor_reshape_realloc<NV, NVHX86, AK_INT8>();
#endif //USE_CUDA

#ifdef USE_X86_PLACE
    Env<X86>::env_init();
    LOG(INFO) << "test X86 FP32 tensor reshape realloc";
    tensor_reshape_realloc<X86, X86, AK_FLOAT>();
    LOG(INFO) << "test X86 INT8 tensor reshape realloc";
    tensor_reshape_realloc<X86, X86, AK_INT8>();
#endif //USE_X86_PLACE

#ifdef USE_ARM_PLACE
    Env<ARM>::env_init();
    LOG(INFO) << "test ARM FP32 tensor reshape realloc";
    tensor_reshape_realloc<ARM, ARM, AK_FLOAT>();
    LOG(INFO) << "test ARM INT8 tensor reshape realloc";
    tensor_reshape_realloc<ARM, ARM, AK_INT8>();
#endif //USE_ARM_PLACE

#ifdef USE_BM
    Env<BM>::env_init();
    LOG(INFO) << "test BM FP32 tensor reshape realloc";
    tensor_reshape_realloc<BM, X86, AK_FLOAT>();
#endif //USE_BM
}
#endif

#if 1
template <typename TargetD, typename TargetH, DataType Dtype>
void test_tensor_op() {
    typedef TargetWrapper<TargetH> HAPI;
    typedef TargetWrapper<TargetD> DAPI;

    typedef typename TargetTypeTraits<TargetH>::target_category target_H;
    typedef typename TargetTypeTraits<TargetD>::target_category target_D;
    typedef typename IF<std::is_same<target_D, target_H>::value, __HtoH, __DtoH>::Type then_type;
    typedef typename IF<std::is_same<target_D, target_H>::value, __DtoD, __HtoD>::Type else_type;
    typedef typename IF<std::is_same<target_D, __host_target>::value, then_type, else_type>::Type flag_type;
    typedef typename IF<std::is_same<target_D, __host_target>::value, HAPI, DAPI>::Type copy_API;

    typedef Tensor<TargetH> TensorH;
    typedef Tensor<TargetD> TensorD;

    typedef typename DataTrait<TargetH, Dtype>::Dtype dtype;

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
TEST(TestSaberFunc, test_tensor_ops) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    LOG(INFO) << "test CUDA FP32 tensor op";
    test_tensor_op<NV, NVHX86, AK_FLOAT>();
    LOG(INFO) << "test CUDA INT8 tensor op";
    test_tensor_op<NV, NVHX86, AK_INT8>();
#endif //USE_CUDA

#ifdef USE_X86_PLACE
    Env<X86>::env_init();
    LOG(INFO) << "test X86 FP32 tensor op";
    test_tensor_op<X86, X86, AK_FLOAT>();
    LOG(INFO) << "test X86 INT8 tensor op";
    test_tensor_op<X86, X86, AK_INT8>();
#endif //USE_X86_PLACE

#ifdef USE_ARM_PLACE
    Env<ARM>::env_init();
    LOG(INFO) << "test ARM FP32 tensor op";
    test_tensor_op<ARM, ARM, AK_FLOAT>();
    LOG(INFO) << "test ARM INT8 tensor op";
    test_tensor_op<ARM, ARM, AK_INT8>();
#endif //USE_ARM_PLACE

#ifdef USE_BM
    Env<BM>::env_init();
    LOG(INFO) << "test BM FP32 tensor op";
    test_tensor_op<BM, X86, AK_FLOAT>();
#endif //USE_BM
}
#endif

#if 1
template <typename TargetD, typename TargetH>
void tensor_share_diff_dtype() {
    Shape sh({1, 1, 2, 10}, Layout_NCHW);
    Tensor<TargetD> td1(sh, AK_FLOAT);
    Tensor<TargetH> th1(sh, AK_FLOAT);
    Tensor<TargetD> td2(AK_INT8);
    Tensor<TargetH> th2(AK_INT8);
    td2.set_shape(sh);
    th2.set_shape(sh);
    LOG(INFO) << "testing host fill tensor with const 1.";
    fill_tensor_const(th1, -1);
    LOG(INFO) << "data type: float";
    print_tensor(th1);
    fill_tensor_const(td1, -1);
    print_tensor(td1);
    LOG(INFO) << "INT8 Tensor shared from FP32 tensor";
    td2.share_from(td1);
    th2.share_from(th1);

    print_tensor(th2);
    print_tensor(td2);
}

TEST(TestSaberFunc, test_tensor_share_diff_dtype) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
    LOG(INFO) << "test CUDA tensor share different data type";
    tensor_share_diff_dtype<NV, NVHX86>();
#endif //USE_CUDA

#ifdef USE_X86_PLACE
    Env<X86>::env_init();
    LOG(INFO) << "test X86 tensor share different data type";
    tensor_share_diff_dtype<X86, X86>();
#endif //USE_X86_PLACE

#ifdef USE_ARM_PLACE
    Env<ARM>::env_init();
    LOG(INFO) << "test ARM tensor share different data type";
    tensor_share_diff_dtype<ARM, ARM>();
#endif //USE_ARM_PLACE

//BM does not support this yet
}
#endif
int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
