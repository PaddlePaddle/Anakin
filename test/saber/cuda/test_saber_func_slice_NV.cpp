
#include "core/context.h"
#include "funcs/slice.h"
#include "test_saber_func_slice_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncSliceNV, test_func_concat_NV) {

    Env<NV>::env_init();
    typedef TargetWrapper<NV> API;

    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;

    typedef TensorDf4::Dtype dtype;

    int slice_axis = 1; // channel
    int w_in = 4;
    int h_in = 4;
    int ch_in = 8;
    std::vector<int> slice_pts = {1, 3, 6};
    int num_in = 2;

    std::vector<int> size_out = {1, 2, 3, 2};

    Shape shape_in(num_in, ch_in, h_in, w_in);

    SliceParam<TensorDf4> param(slice_axis, slice_pts);

    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;

    LOG(INFO) << "slice axis= " << param.axis;
    LOG(INFO) << "slice points = " << param.slice_points[0] << ", " << \
              param.slice_points[1] << ", " << param.slice_points[2];

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in);

    for (int i = 0; i < thin.size(); ++i) {
        thin.mutable_data()[i] = i;
    }

    TensorDf4 tdin;
    tdin.re_alloc(shape_in);
    tdin.copy_from(thin);
    input_dev_4d.push_back(&tdin);

    std::vector<TensorDf4> tdev(4);

    // start Reshape & doInfer
    Context<NV> ctx_dev(0, 1, 1);

    Slice<NV, AK_FLOAT> slice_dev;

    typedef std::vector<Shape> Shape_v;
    Shape_v shape_out_v;

    for (int i = 0; i < 4; ++i) {
        output_dev_4d.push_back(&tdev[i]);
    }

    LOG(INFO) << "slice compute output shape";
    slice_dev.compute_output_shape(input_dev_4d, output_dev_4d, param);

    for (int j = 0; j < output_dev_4d.size(); ++j) {
        Shape sh = output_dev_4d[j]->valid_shape();
        LOG(INFO) << "output shape: " << sh[0] << ", " << sh[1] << \
                  ", " << sh[2] << ", " << sh[3];
        Shape shout{num_in, size_out[j], h_in, w_in};
        CHECK_EQ(shout == sh, true) << "compute shape error";
    }

    for (int i = 0; i < 4; ++i) {
        output_dev_4d[i]->re_alloc(output_dev_4d[i]->shape());
    }

    LOG(INFO) << "slice initialization";
    slice_dev.init(input_dev_4d, output_dev_4d, param, RUNTIME, SABER_IMPL, ctx_dev);

    LOG(INFO) << "slice compute";
    slice_dev(input_dev_4d, output_dev_4d, param, ctx_dev);

    for (int i = 0; i < 4; ++i) {
        output_dev_4d[i]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[i]->sync();
        print_tensor_device(*output_dev_4d[i]);
        cudaDeviceSynchronize();
    }

    TensorHf4 th1, th2, th3, th4;
    th1.re_alloc(output_dev_4d[0]->valid_shape());
    th2.re_alloc(output_dev_4d[1]->valid_shape());
    th3.re_alloc(output_dev_4d[2]->valid_shape());
    th4.re_alloc(output_dev_4d[3]->valid_shape());

    th1.copy_from(*output_dev_4d[0]);
    th2.copy_from(*output_dev_4d[1]);
    th3.copy_from(*output_dev_4d[2]);
    th4.copy_from(*output_dev_4d[3]);

    print_tensor_host(th1);
    print_tensor_host(th2);
    print_tensor_host(th3);
    print_tensor_host(th4);

    //! slice from sliced tensor
    LOG(INFO) << "do slice to a sliced tensor";
    int slice_axis1 = 3; // channel
    int w_in_sum = 4;
    int h_in1 = 4;
    int ch_in1 = 1;
    std::vector<int> slice_pts1 = {2};
    int num_in1 = 2;

    std::vector<int> size_out1 = {2, 2};

    Shape shape_in1 = tdev[0].valid_shape();

    SliceParam<TensorDf4> param1(slice_axis1, slice_pts1);

    std::vector<TensorDf4*> input_dev_4d1;
    std::vector<TensorDf4*> output_dev_4d1;
    std::vector<TensorDf4> tdev1(2);
    input_dev_4d1.push_back(&tdev[0]);
    output_dev_4d1.push_back(&tdev1[0]);
    output_dev_4d1.push_back(&tdev1[1]);

    Slice<NV, AK_FLOAT> slice_dev1;

    //! compute output shape
    slice_dev1.compute_output_shape(input_dev_4d1, output_dev_4d1, param1);

    LOG(INFO) << "real shape:" << tdev[0].shape()[0] << ", " << tdev[0].shape()[1] \
              << ", " << tdev[0].shape()[2] << ", " << tdev[0].shape()[3];

    for (int j = 0; j < output_dev_4d1.size(); ++j) {
        Shape sh = output_dev_4d1[j]->valid_shape();
        LOG(INFO) << "output shape: " << sh[0] << ", " << sh[1] << \
                  ", " << sh[2] << ", " << sh[3];
        Shape shout{num_in1, ch_in1, h_in1, size_out1[j]};
        CHECK_EQ(shout == sh, true) << "compute shape error";
    }

    for (int i = 0; i < 2; ++i) {
        output_dev_4d1[i]->re_alloc(output_dev_4d1[i]->shape());
    }

    slice_dev1.init(input_dev_4d1, output_dev_4d1, param1, RUNTIME, SABER_IMPL, ctx_dev);

    LOG(INFO) << "slice compute";
    slice_dev1(input_dev_4d1, output_dev_4d1, param1, ctx_dev);

    for (int i = 0; i < 2; ++i) {
        output_dev_4d[i]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[i]->sync();
        print_tensor_device(*output_dev_4d1[i]);
        cudaDeviceSynchronize();
    }

    TensorHf4 th11, th21;
    th11.re_alloc(output_dev_4d1[0]->valid_shape());
    th21.re_alloc(output_dev_4d1[1]->valid_shape());

    th11.copy_from(*output_dev_4d1[0]);
    th21.copy_from(*output_dev_4d1[1]);

    print_tensor_host(th11);
    print_tensor_host(th21);
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

