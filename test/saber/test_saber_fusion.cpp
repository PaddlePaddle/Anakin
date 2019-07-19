#include "saber/core/context.h"
#include "saber/funcs/fusion.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>
#include <cmath>
using namespace anakin::saber;

const char* model_path = "/home/mayongqiang01/release/bmnetc/resnet50";


TEST(TestSaberFunc, test_func_fusion) {
#ifdef USE_BM_PLACE
    LOG(INFO) << "BM test......";

    Env<BM>::env_init();
    
    Context<BM> ctx1(0, 1, 1);
    
    FusionParam<BM> param(model_path);

    std::vector<Tensor<BM>*> input_devices;
    std::vector<Tensor<BM>*> output_devices;

    std::vector<Tensor<BMX86>*> input_hosts;
    std::vector<Tensor<BMX86>*> output_hosts;

    Tensor<BMX86> input_host, output_host;
    Tensor<BM> input_dev, output_dev;

    Shape input_s({1, 3, 224, 224}, Layout_NCHW);
    input_host.re_alloc(input_s, AK_FLOAT);
    input_dev.re_alloc(input_s, AK_FLOAT);

    Shape output_s({1, 1000, 1, 1}, Layout_NCHW);
    output_host.re_alloc(output_s, AK_FLOAT);
    output_dev.re_alloc(input_s, AK_FLOAT);

    input_devices.push_back(&input_dev);
    output_devices.push_back(&output_dev);

    Fusion<BM, AK_FLOAT> fusion;
    
    fusion.init(input_devices, output_devices, param, SPECIFY, SABER_IMPL, ctx1);
    fusion(input_devices, output_devices, param, ctx1);

    LOG(INFO) << "BM test end.";
#endif

}

int main(int argc, const char** argv) {
    if (argc < 2) {
        printf("usage: model_path\n");
        exit(0);
    }

    model_path = argv[1];

    
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

