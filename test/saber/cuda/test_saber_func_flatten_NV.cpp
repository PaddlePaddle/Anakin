
#include "core/context.h"
#include "saber/funcs/flatten.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncNV, test_func_flatten) {

    Env<NV>::env_init();
    //Env<X86>::env_init();
    typedef TargetWrapper<NV> API;
    typedef TargetWrapper<X86> X86_API;

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef TensorHf4::Dtype dtype;

    int w_in = 8;
    int h_in = 8;
    int ch_in = 4;
    int num_in = 2;

    FlattenParam<TensorDf4> param_dev;

    LOG(INFO) << "Flatten param: ";
    LOG(INFO) << " input size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out(num_in, ch_in * h_in * w_in, 1, 1);

    TensorDf4 tdin(shape_in);
    TensorDf4 tdout;

    fill_tensor_device_rand(tdin, -1.f, 1.f);

    // start Reshape & doInfer
    Context<NV> ctx_dev(0, 1, 1);

    std::vector<TensorDf4*> vin;
    std::vector<TensorDf4*> vout;

    vin.push_back(&tdin);
    vout.push_back(&tdout);

    Flatten<NV, AK_FLOAT> dev_flatten;

    LOG(INFO) << "flatten compute output shape";
    dev_flatten.compute_output_shape(vin, vout, param_dev);

    Shape va_sh = vout[0]->valid_shape();
    CHECK_EQ(va_sh == shape_out, true) << "compute output shape error";
    LOG(INFO) << "shape out 2d: " << vout[0] << ", " << vout[1] << ", " \
              << vout[2] << ", " << vout[3];

    tdout.set_shape(va_sh);
    tdout.share_from(tdin);
    //tdout.re_alloc(va_sh);

    // init assume output tensor has been reshpaed by user.
    LOG(INFO) << "flatten initialization";
    SABER_CHECK(dev_flatten.init(vin, vout, param_dev, \
                                 RUNTIME, SABER_IMPL, ctx_dev));

    LOG(INFO) << "flatten compute";
    dev_flatten(vin, vout, param_dev, ctx_dev);
    print_tensor_device(*vin[0]);
    cudaDeviceSynchronize();
    print_tensor_device(*vout[0]);
    cudaDeviceSynchronize();
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

