#include "core/context.h"
#include "funcs/batch_norm.h"
#include "test_saber_func_BM.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;


TEST(TestSaberFuncBM, test_func_batch_norm_BM) {

    typedef TargetWrapper<BM> API;
    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;
    typedef TensorDf4::Dtype dtype;

    //Input / output tensor
    Shape shape_in(1, 1, 2, 2);
    Shape shape_out = shape_in;

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in);
    for (int i = 0; i < thin.size(); ++i) {
        thin.mutable_data()[i] = 1+i;
    }

    TensorDf4 tdin, tdout;
    tdin.re_alloc(shape_in);
    tdin.copy_from(thin);
    input_dev_4d.push_back(&tdin);

    LOG(INFO) << "Input tensor is:";
    print_tensor_device(*input_dev_4d[0]);

    //Batch norm param
    std::vector<float> mean;
    mean.push_back(1);

    std::vector<float> variance;
    variance.push_back(0.001);

    float scale_in = 1;
    float eps_in = float(1e-5);

    BatchnormParam<TensorDf4> param(mean, variance, scale_in);

    //BatachNorm
    BatchNorm<BM, AK_BM, AK_BM, AK_BM, NCHW> batchNorm;

    output_dev_4d.push_back(&tdout);
    batchNorm.compute_output_shape(input_dev_4d, output_dev_4d, param);

    LOG(INFO) << "re-alloc tensor buffer";
    output_dev_4d[0]->re_alloc(output_dev_4d[0]->shape());

    LOG(INFO) << "batch norm initialized to bm impl";
    Context<BM> ctx_dev(0, 1, 1);
    batchNorm.init(input_dev_4d, output_dev_4d, param, SPECIFY, VENDER_IMPL, ctx_dev);

    LOG(INFO) << "bm batch norm compute";
    SaberTimer<BM> t1;
    t1.clear();
    t1.start(ctx_dev);

    batchNorm(input_dev_4d, output_dev_4d, param, ctx_dev);

    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    printf("bm batch norm total time : %.4f, avg time : %.4f\n", ts, ts);

    print_tensor_device(*output_dev_4d[0]);
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    //Env<BM>::env_init();
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

