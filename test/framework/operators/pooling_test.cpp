#include "operator_tests.h"
#include "thread_pool.h"

TEST(OperatorsTest, PoolingFactoryTest) {
    OpContext<NV> opctx;
    std::vector<Tensor4dPtr<NV, AK_FLOAT> > in;
    std::vector<Tensor4dPtr<NV, AK_FLOAT> > out;

    OpContext<ARM> opctx_arm;
    std::vector<Tensor4dPtr<ARM, AK_FLOAT> > in_arm;
    std::vector<Tensor4dPtr<ARM, AK_FLOAT> > out_arm;


    /*Operator<RTCUDA, float>*/ auto* Op_name0 =
        OpFactory<ARM, AK_FLOAT, Precision::FP32>::Global().Create("Pooling");
    /*Operator<RTCUDA, float>*/ auto* Op_name1 =
        OpFactory<NV, AK_FLOAT, Precision::FP32>::Global()["pooling"];
    /*Operator<RTCUDA, float>**/auto* Op_name2 =
        OpFactory<NV, AK_FLOAT, Precision::FP32>::Global()["pool"];
    auto& op_list = OpFactory<NV, AK_FLOAT, Precision::FP32>::Global().get_list_op_name();

    for (auto& item : op_list) {
        LOG(INFO) << " op: " << item;
    }

    LOG(WARNING) << " op name alias 0 : Pooling";
    LOG(INFO) << "  run forward function";
    (*Op_name0)(opctx_arm, in_arm, out_arm);
    LOG(WARNING) << " op name alias 1 : pooling";
    LOG(INFO) << "  run forward function";
    (*Op_name1)(opctx, in, out);
    LOG(WARNING) << " op name alias 2 : pool";
    LOG(INFO) << "  run forward function";
    (*Op_name2)(opctx, in, out);
}


int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
