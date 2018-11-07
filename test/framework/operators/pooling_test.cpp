#include "operator_tests.h"
#include "thread_pool.h"

#ifdef USE_CUDA
using Target = NV;
#elif defined(AMD_GPU)
using Target = AMD;
#elif defined(USE_X86_PLACE)
using Target = X86;
#else
using Target = ARM;
#endif


TEST(OperatorsTest, PoolingFactoryTest) {
    OpContext<Target> opctx;
    std::vector<Tensor4dPtr<Target> > in;
    std::vector<Tensor4dPtr<Target> > out;

    auto* Op_name_in =
        OpFactory<Target, Precision::FP32>::Global()["input"];

    /*Operator<RTCUDA, float>*/ auto* Op_name1 =
        OpFactory<Target, Precision::FP32>::Global()["pooling"];
    /*Operator<RTCUDA, float>**/auto* Op_name2 =
        OpFactory<Target, Precision::FP32>::Global()["pool"];
    auto& op_list = OpFactory<Target, Precision::FP32>::Global().get_list_op_name();

    for (auto & item : op_list) {
        LOG(INFO) << " op: " << item;
    }

    LOG(WARNING) << " op name alias 1 : pooling";
    LOG(INFO) << "  run forward function";
    CHECK(Op_name1 != nullptr);
    (*Op_name1)(opctx, in, out);
    LOG(WARNING) << " op name alias 2 : pool";
    LOG(INFO) << "  run forward function";
    (*Op_name2)(opctx, in, out);
}


int main(int argc, const char** argv) {
#ifdef AMD_GPU
    Env<AMD>::env_init();
#endif
    // initial logger
    Env<Target>::env_init();
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
