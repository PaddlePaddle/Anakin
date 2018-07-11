#include "core/context.h"
#include "funcs/eltwise.h"
#include "test_saber_func_BM.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;


TEST(TestSaberFuncBM, test_func_prod) {

    Env<BM>::env_init();
    typedef TargetWrapper<BM> API;

    typedef TargetWrapper<BM> BM_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

    EltwiseType elt_type = Eltwise_prod;

    EltwiseParam<TensorDf4> param(elt_type);

    int w_in = 10;
    int h_in = 2;
    int ch_in = 2;
    int num_in = 1;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out = shape_in;

    // Host Tensor
    Tensor<X86, AK_FLOAT, NCHW> thin0(shape_in);
    Tensor<X86, AK_FLOAT, NCHW> thin1(shape_in);
    Tensor<X86, AK_FLOAT, NCHW> thin2(shape_in);
    for (int i = 0; i < thin0.size(); ++i) {
        thin0.mutable_data()[i] = i;
    }
    for (int i = 0; i < thin1.size(); ++i) {
        thin1.mutable_data()[i] = i + 1;
    }
    for (int i = 0; i < thin2.size(); ++i) {
        thin2.mutable_data()[i] = 1;
    }

    // Device Tensor
    TensorDf4 tdin0, tdin1, tdin2, tdout;
    tdin0.re_alloc(shape_in);
    tdin1.re_alloc(shape_in);
    tdin2.re_alloc(shape_in);
    tdin0.copy_from(thin0);
    tdin1.copy_from(thin1);
    tdin2.copy_from(thin2);
    tdout.re_alloc(shape_out);

    // Device Vector of Tensor
    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;
    input_dev_4d.push_back(&tdin0);
    input_dev_4d.push_back(&tdin1);
    input_dev_4d.push_back(&tdin2);
    output_dev_4d.push_back(&tdout);


    Context<BM> ctx_dev(0, 1, 1);
    Eltwise<BM, AK_BM, AK_BM, AK_BM, NCHW> eltwise_dev;

    LOG(INFO) << "eltwise compute output shape";
    eltwise_dev.compute_output_shape(input_dev_4d, output_dev_4d, param);

    // Verify output shape
    Shape sh = output_dev_4d[0]->valid_shape();
    LOG(INFO) << "output shape: "<< sh[0] << ", " << sh[1] << \
        ", " << sh[2] << ", " << sh[3];
    Shape shout{num_in, ch_in, h_in, w_in};
    CHECK_EQ(shout == sh, true) << "compute shape error";

    output_dev_4d[0]->re_alloc(output_dev_4d[0]->shape());

    LOG(INFO) << "eltwise initialization";
    eltwise_dev.init(input_dev_4d, output_dev_4d, param, SPECIFY, VENDER_IMPL, ctx_dev);


    LOG(INFO) << "eltwise compute";
    eltwise_dev(input_dev_4d, output_dev_4d, param, ctx_dev);

    print_tensor_device(*output_dev_4d[0]);
}


TEST(TestSaberFuncBM, test_func_sum) {

    Env<BM>::env_init();

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

    EltwiseType elt_type = Eltwise_sum;

    int w_in = 10;
    int h_in = 2;
    int ch_in = 2;
    int num_in = 1;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out = shape_in;

    // Host Tensor
    TensorHf4 thin1(shape_in);
    TensorHf4 thin2(shape_in);

    for (int i = 0; i < thin1.size(); ++i) {
        thin1.mutable_data()[i] = 1.0;
    }

    for (int i = 0; i < thin2.size(); ++i) {
        thin2.mutable_data()[i] = 2.0;
    }

    // Device Tensor
    TensorDf4 tdin0, tdin1, tdout;
    tdin0.re_alloc(shape_in);
    tdin1.re_alloc(shape_in);
    tdin0.copy_from(thin1);
    tdin1.copy_from(thin2);
    tdout.re_alloc(shape_out);

    // Device Vector of Tensor
    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;
    input_dev_4d.push_back(&tdin0);
    input_dev_4d.push_back(&tdin1);
    input_dev_4d.push_back(&tdin1);
    output_dev_4d.push_back(&tdout);

    EltwiseParam<TensorDf4> param(elt_type);

    Context<BM> ctx_dev(0, 1, 1);
    Eltwise<BM, AK_BM, AK_BM, AK_BM, NCHW> eltwise_dev;

    LOG(INFO) << "eltwise compute output shape";
    eltwise_dev.compute_output_shape(input_dev_4d, output_dev_4d, param);


    // Verify output shape
    Shape sh = output_dev_4d[0]->valid_shape();
    LOG(INFO) << "output shape: " << sh[0] << ", " << sh[1] << \
              ", " << sh[2] << ", " << sh[3];
    Shape shout{num_in, ch_in, h_in, w_in};
    CHECK_EQ(shout == sh, true) << "compute shape error";

    output_dev_4d[0]->re_alloc(output_dev_4d[0]->shape());

    LOG(INFO) << "eltwise initialization";
    eltwise_dev.init(input_dev_4d, output_dev_4d, param, SPECIFY, VENDER_IMPL, ctx_dev);

    LOG(INFO) << "eltwise compute";
    eltwise_dev(input_dev_4d, output_dev_4d, param, ctx_dev);
    print_tensor_device(*output_dev_4d[0]);
}

TEST(TestSaberFuncBM, test_func_max) {

    Env<BM>::env_init();

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

    EltwiseType elt_type = Eltwise_max;

    EltwiseParam<TensorDf4> param(elt_type);

    int w_in = 10;
    int h_in = 2;
    int ch_in = 2;
    int num_in = 1;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out = shape_in;

    // Host Tensor
    Tensor<X86, AK_FLOAT, NCHW> thin0(shape_in);
    Tensor<X86, AK_FLOAT, NCHW> thin1(shape_in);
    Tensor<X86, AK_FLOAT, NCHW> thin2(shape_in);
    for (int i = 0; i < thin0.size(); ++i) {
        thin0.mutable_data()[i] = i;
    }
    for (int i = 0; i < thin1.size(); ++i) {
        thin1.mutable_data()[i] = i + 2;
    }
    for (int i = 0; i < thin2.size(); ++i) {
        thin2.mutable_data()[i] = i + 1;
    }

    // Device Tensor
    TensorDf4 tdin0, tdin1, tdin2, tdout;
    tdin0.re_alloc(shape_in);
    tdin1.re_alloc(shape_in);
    tdin2.re_alloc(shape_in);
    tdin0.copy_from(thin0);
    tdin1.copy_from(thin1);
    tdin2.copy_from(thin2);
    tdout.re_alloc(shape_out);

    // Device Vector of Tensor
    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;
    input_dev_4d.push_back(&tdin0);
    input_dev_4d.push_back(&tdin1);
    input_dev_4d.push_back(&tdin2);
    output_dev_4d.push_back(&tdout);

    Context<BM> ctx_dev(0, 1, 1);
    Eltwise<BM, AK_BM, AK_BM, AK_BM, NCHW> eltwise_dev;

    LOG(INFO) << "eltwise compute output shape";
    eltwise_dev.compute_output_shape(input_dev_4d, output_dev_4d, param);

    // Verify output shape
    Shape sh = output_dev_4d[0]->valid_shape();
    LOG(INFO) << "output shape: "<< sh[0] << ", " << sh[1] << \
        ", " << sh[2] << ", " << sh[3];
    Shape shout{num_in, ch_in, h_in, w_in};
    CHECK_EQ(shout == sh, true) << "compute shape error";

    output_dev_4d[0]->re_alloc(output_dev_4d[0]->shape());

    LOG(INFO) << "eltwise initialization";
    eltwise_dev.init(input_dev_4d, output_dev_4d, param, SPECIFY, VENDER_IMPL, ctx_dev);

    LOG(INFO) << "eltwise compute";
    eltwise_dev(input_dev_4d, output_dev_4d, param, ctx_dev);
    print_tensor_device(*output_dev_4d[0]);

}

/*   0   1   2   3   4
 *  10  11  12  13  14   (tdin_roi1, c=0)
 *   (tdin_roi0, c=0)   25  26  27  28  29
 *                      35  36  37  38  39
 * =======================================
 *  40  41  42  43  44
 *  50  51  52  53  54   (tdin_roi1, c=1)
 *   (tdin_roi0, c=1)   65  66  67  68  69
 *                      75  76  77  78  79
 */
/*
TEST(TestSaberFuncBM, test_func_prod_roi) {

    Env<BM>::env_init();
    typedef TargetWrapper<BM> API;

    typedef TargetWrapper<BM> BM_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

    EltwiseType elt_type = Eltwise_prod;

    EltwiseParam<TensorDf4> param(elt_type);

    int w_in = 10;
    int h_in = 4;
    int ch_in = 2;
    int num_in = 1;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_in_roi{num_in, ch_in, h_in / 2, w_in / 2};
    Shape off0{0, 0, 0, 0};
    Shape off1{0, 0, 2, 5};
    Shape shape_out = shape_in_roi;

    // Host Tensor
    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in);
    for (int i = 0; i < thin.size(); ++i) {
        thin.mutable_data()[i] = i;
    }

    // Device Tensor
    TensorDf4 tdin, tdin_roi0, tdin_roi1, tdout;
    tdin.re_alloc(shape_in);
    tdin.copy_from(thin);
    tdin_roi0.share_sub_buffer(tdin, shape_in_roi, off0);
    tdin_roi1.share_sub_buffer(tdin, shape_in_roi, off1);
    tdout.re_alloc(shape_out);


    // Device Vector of Tensor
    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;
    input_dev_4d.push_back(&tdin_roi0);
    input_dev_4d.push_back(&tdin_roi1);
    input_dev_4d.push_back(&tdin_roi1);
    output_dev_4d.push_back(&tdout);


    Context<BM> ctx_dev(0, 1, 1);
    Eltwise<BM, AK_BM, AK_BM, AK_BM, NCHW> eltwise_dev;

    LOG(INFO) << "eltwise compute output shape";
    eltwise_dev.compute_output_shape(input_dev_4d, output_dev_4d, param);


    // Verify output shape
    Shape sh = output_dev_4d[0]->valid_shape();
    LOG(INFO) << "output shape: "<< sh[0] << ", " << sh[1] << \
        ", " << sh[2] << ", " << sh[3];
    Shape shout(shape_in_roi);
    CHECK_EQ(shout == sh, true) << "compute shape error";

    output_dev_4d[0]->re_alloc(output_dev_4d[0]->shape());

    LOG(INFO) << "eltwise initialization";
    eltwise_dev.init(input_dev_4d, output_dev_4d, param, SPECIFY, VENDER_IMPL, ctx_dev);


    LOG(INFO) << "eltwise compute";
    eltwise_dev(input_dev_4d, output_dev_4d, param, ctx_dev);


    output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
    output_dev_4d[0]->sync();
    print_tensor_device(*output_dev_4d[0]);
    cudaDeviceSynchronize();


    TensorHf4 th_for_print;
    th_for_print.re_alloc(output_dev_4d[0]->valid_shape());
    th_for_print.copy_from(*output_dev_4d[0]);
    print_tensor_host(th_for_print);

    CUDA_CHECK(cudaPeekAtLastError());
}

*/

/*   0   1   2   3   4
 *  10  11  12  13  14   (tdin_roi1, c=0)
 *   (tdin_roi0, c=0)   25  26  27  28  29
 *                      35  36  37  38  39
 * =======================================
 *  40  41  42  43  44
 *  50  51  52  53  54   (tdin_roi1, c=1)
 *   (tdin_roi0, c=1)   65  66  67  68  69
 *                      75  76  77  78  79
 */
/*
TEST(TestSaberFuncBM, test_func_sum_roi_new) {

    Env<BM>::env_init();
    typedef TargetWrapper<BM> API;

    typedef TargetWrapper<BM> BM_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

    EltwiseType elt_type = Eltwise_sum;

    int w_in = 10;
    int h_in = 4;
    int ch_in = 2;
    int num_in = 1;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_in_roi{num_in, ch_in, h_in / 2, w_in / 2};

    Shape off0{0, 0, 0, 0};
    Shape off1{0, 0, 2, 5};
    Shape shape_out = shape_in_roi;

    // Host Tensor
    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in);
    for (int i = 0; i < thin.size(); ++i) {
        thin.mutable_data()[i] = i;
    }

    // Device Tensor
    TensorDf4 tdin, tdin_roi0, tdin_roi1, tdout;
    tdin.re_alloc(shape_in);
    tdin.copy_from(thin);
    tdin_roi0.share_sub_buffer(tdin, shape_in_roi, off0);
    tdin_roi1.share_sub_buffer(tdin, shape_in_roi, off1);
    tdout.re_alloc(shape_out);

    // Device Vector of Tensor
    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;
    input_dev_4d.push_back(&tdin_roi0);
    input_dev_4d.push_back(&tdin_roi1);
//    input_dev_4d.push_back(&tdin_roi1);
//    input_dev_4d.push_back(&tdin_roi1);
    output_dev_4d.push_back(&tdout);

//    Shape shape_coeff(1, 1, 1, input_dev_4d.size());
//    TensorHf4 thcoeff(shape_coeff);
//    for (int i = 0; i < thcoeff.size(); ++i) {
//        thcoeff.mutable_data()[i] = 1;
//    }

    EltwiseParam<TensorDf4> param(elt_type);

    Context<BM> ctx_dev(0, 1, 1);
    Eltwise<BM, AK_BM, AK_BM, AK_BM, NCHW> eltwise_dev;

    LOG(INFO) << "eltwise compute output shape";
    eltwise_dev.compute_output_shape(input_dev_4d, output_dev_4d, param);

    // Verify output shape
    Shape sh = output_dev_4d[0]->valid_shape();
    LOG(INFO) << "output shape: "<< sh[0] << ", " << sh[1] << \
        ", " << sh[2] << ", " << sh[3];
    Shape shout(shape_in_roi);
    CHECK_EQ(shout == sh, true) << "compute shape error";

    output_dev_4d[0]->re_alloc(output_dev_4d[0]->shape());

    LOG(INFO) << "eltwise initialization";
    eltwise_dev.init(input_dev_4d, output_dev_4d, param, SPECIFY, VENDER_IMPL, ctx_dev);

    print_tensor_device(*input_dev_4d[0]);
    print_tensor_device(*input_dev_4d[1]);
    cudaDeviceSynchronize();
    LOG(INFO) << "eltwise compute";
    eltwise_dev(input_dev_4d, output_dev_4d, param, ctx_dev);

    output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
    output_dev_4d[0]->sync();
    print_tensor_device(*output_dev_4d[0]);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}
*/
/*
TEST(TestSaberFuncBM, test_func_sum_roi) {

    Env<BM>::env_init();
    typedef TargetWrapper<BM> API;

    typedef TargetWrapper<BM> BM_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

    EltwiseType elt_type = Eltwise_sum;

    int w_in = 10;
    int h_in = 4;
    int ch_in = 2;
    int num_in = 1;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_in_roi{num_in, ch_in, h_in / 2, w_in / 2};
    Shape off0{0, 0, 0, 0};
    Shape off1{0, 0, 2, 5};
    Shape shape_out = shape_in_roi;

    // Host Tensor
    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in);
    for (int i = 0; i < thin.size(); ++i) {
        thin.mutable_data()[i] = i;
    }

    // Device Tensor
    TensorDf4 tdin, tdin_roi0, tdin_roi1, tdout;
    tdin.re_alloc(shape_in);
    tdin.copy_from(thin);
    tdin_roi0.share_sub_buffer(tdin, shape_in_roi, off0);
    tdin_roi1.share_sub_buffer(tdin, shape_in_roi, off1);
    tdout.re_alloc(shape_out);

    // Device Vector of Tensor
    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;
    input_dev_4d.push_back(&tdin_roi0);
    input_dev_4d.push_back(&tdin_roi1);
    output_dev_4d.push_back(&tdout);

    //Shape shape_coeff(1, 1, 1, 3);
    Shape shape_coeff(1, 1, 1, input_dev_4d.size());
    TensorHf4 thcoeff(shape_coeff);

    for (int i = 0; i < thcoeff.size(); ++i) {
        thcoeff.mutable_data()[i] = i;
    }
    TensorDf4 tdcoeff;
    tdcoeff.re_alloc(shape_coeff);
    tdcoeff.copy_from(thcoeff);

    EltwiseParam<TensorDf4> param(elt_type, &tdcoeff);

    Context<BM> ctx_dev(0, 1, 1);
    Eltwise<BM, AK_BM, AK_BM, AK_BM, NCHW> eltwise_dev;

    LOG(INFO) << "eltwise compute output shape";
    eltwise_dev.compute_output_shape(input_dev_4d, output_dev_4d, param);

    // Verify output shape
    Shape sh = output_dev_4d[0]->valid_shape();
    LOG(INFO) << "output shape: "<< sh[0] << ", " << sh[1] << \
        ", " << sh[2] << ", " << sh[3];
    Shape shout(shape_in_roi);
    CHECK_EQ(shout == sh, true) << "compute shape error";

    output_dev_4d[0]->re_alloc(output_dev_4d[0]->shape());

    LOG(INFO) << "eltwise initialization";
    eltwise_dev.init(input_dev_4d, output_dev_4d, param, SPECIFY, VENDER_IMPL, ctx_dev);

    LOG(INFO) << "eltwise compute";
    eltwise_dev(input_dev_4d, output_dev_4d, param, ctx_dev);

    output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
    output_dev_4d[0]->sync();
    print_tensor_device(*output_dev_4d[0]);
    cudaDeviceSynchronize();

    TensorHf4 th_for_print;
    th_for_print.re_alloc(output_dev_4d[0]->valid_shape());
    th_for_print.copy_from(*output_dev_4d[0]);
    print_tensor_host(th_for_print);

    CUDA_CHECK(cudaPeekAtLastError());
}
*/

/*
TEST(TestSaberFuncBM, test_func_max_roi) {

    Env<BM>::env_init();
    typedef TargetWrapper<BM> API;

    typedef TargetWrapper<BM> BM_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

    EltwiseType elt_type = Eltwise_max;

    int w_in = 10;
    int h_in = 4;
    int ch_in = 2;
    int num_in = 1;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_in_roi{num_in, ch_in, h_in / 2, w_in / 2};
    Shape off0{0, 0, 0, 0};
    Shape off1{0, 0, 2, 5};
    Shape shape_out = shape_in_roi;

    // Host Tensor
    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in);
    for (int i = 0; i < thin.size(); ++i) {
        thin.mutable_data()[i] = i;
    }

    // Device Tensor
    TensorDf4 tdin, tdin_roi0, tdin_roi1, tdout;
    tdin.re_alloc(shape_in);
    tdin.copy_from(thin);
    tdin_roi0.share_sub_buffer(tdin, shape_in_roi, off0);
    tdin_roi1.share_sub_buffer(tdin, shape_in_roi, off1);
    tdout.re_alloc(shape_out);

    // Device Vector of Tensor
    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;
    input_dev_4d.push_back(&tdin_roi0);
    input_dev_4d.push_back(&tdin_roi1);
    output_dev_4d.push_back(&tdout);

    EltwiseParam<TensorDf4> param(elt_type);

    Context<BM> ctx_dev(0, 1, 1);
    Eltwise<BM, AK_BM, AK_BM, AK_BM, NCHW> eltwise_dev;

    LOG(INFO) << "eltwise compute output shape";
    eltwise_dev.compute_output_shape(input_dev_4d, output_dev_4d, param);

    // Verify output shape
    Shape sh = output_dev_4d[0]->valid_shape();
    LOG(INFO) << "output shape: "<< sh[0] << ", " << sh[1] << \
        ", " << sh[2] << ", " << sh[3];
    Shape shout(shape_in_roi);
    CHECK_EQ(shout == sh, true) << "compute shape error";

    output_dev_4d[0]->re_alloc(output_dev_4d[0]->shape());

    LOG(INFO) << "eltwise initialization";
    eltwise_dev.init(input_dev_4d, output_dev_4d, param, SPECIFY, VENDER_IMPL, ctx_dev);

    LOG(INFO) << "eltwise compute";
    eltwise_dev(input_dev_4d, output_dev_4d, param, ctx_dev);

    output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
    output_dev_4d[0]->sync();
    print_tensor_device(*output_dev_4d[0]);
    cudaDeviceSynchronize();

    TensorHf4 th_for_print;
    th_for_print.re_alloc(output_dev_4d[0]->valid_shape());
    th_for_print.copy_from(*output_dev_4d[0]);
    print_tensor_host(th_for_print);

    CUDA_CHECK(cudaPeekAtLastError());
}

*/

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
