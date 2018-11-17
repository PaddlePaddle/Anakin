#include <string>
#include "net_test.h"
#include "framework/core/net/entropy_calibrator.h"
#include "saber/funcs/timer.h"
#include <chrono>
#ifdef USE_CUDA

#if defined(NVIDIA_GPU)
using Target = NV;
using Target_H = NVHX86;
#elif defined(USE_X86_PLACE)
using Target = X86;
using Target_H = X86;
#elif defined(USE_ARM_PLACE)
using Target = ARM;
using Target_H = ARM;
#elif defined(AMD_GPU)
using Target = AMD;
using Target_H = AMDHX86;
#endif

//#define USE_DIEPSE

std::string model_path = "/home/zhangshuai20/workspace/baidu/personal-code/anakin_test/public/alexnet_relu_bn/model/anakin/alexnet_relu_bn.anakin.bin";

std::string data_file = "./data_list.txt";
std::string calibrator_file = "./calibrator.txt";
int bin_num = 256;
int batch_size = 2;
#if defined(NVIDIA_GPU)
TEST(NetTest, calibrator) {
    Graph<NV, Precision::FP32>* graph = new Graph<NV, Precision::FP32>();
    // load anakin model files.
    auto status = graph->load(model_path);
    if (!status ) {
        delete graph;
        LOG(FATAL) << " [ERROR] " << status.info();
        exit(-1);
    }

    //anakin graph optimization
    graph->Optimize();

    // constructs the executer net
    Net<NV, Precision::FP32, OpRunType::SYNC> net_executer(*graph);
    BatchStream<NV> batch_stream(data_file, batch_size);
    EntropyCalibrator<NV> entropy_calibrator(&batch_stream, batch_size, calibrator_file, &net_executer, bin_num);
    entropy_calibrator.generate_calibrator_table();

    delete graph;

}
#endif

int main(int argc, const char** argv){

	Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
#else
int main(int argc, const char** argv) {
    return -1;
}
#endif