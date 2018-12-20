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

std::string g_model_path;
std::string g_data_file = "./data_list.txt";
std::string g_calibrator_file = "./calibrator.txt";
int g_batch_size = 1;
int g_bin_num = 2048;
#if defined(NVIDIA_GPU)
TEST(NetTest, calibrator) {
#ifdef USE_OPENCV
    Graph<NV, Precision::FP32>* graph = new Graph<NV, Precision::FP32>();
    // load anakin model files.
    auto status = graph->load(g_model_path);
    if (!status ) {
        delete graph;
        LOG(FATAL) << " [ERROR] " << status.info();
        exit(-1);
    }

    //anakin graph optimization
    graph->Optimize();

    // constructs the executer net
    Net<NV, Precision::FP32, OpRunType::SYNC> net_executer(*graph);
    BatchStream<NV> batch_stream(g_data_file, 3, 192, 192, {104.008f, 116.669f, 122.675f}, {1.f, 1.f, 1.f});
    EntropyCalibrator<NV> entropy_calibrator(&batch_stream, g_batch_size, g_calibrator_file, &net_executer, g_bin_num);
    entropy_calibrator.generate_calibrator_table();

    delete graph;
#else
    LOG(ERROR) << "turn on USE_OPENCV first";
#endif
}
#endif

int main(int argc, const char** argv){

	Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);

    LOG(INFO) << "usage:";
    LOG(INFO) << argv[0] << " <lite model> <data_file> <calibrate_file>";
    LOG(INFO) << "   lite_model:     path to anakin lite model";
    LOG(INFO) << "   data_file:      path to image data list";
    LOG(INFO) << "   calibrate file: path to calibrate data path";
    if(argc < 4) {
        LOG(ERROR) << "useage: " << argv[0] << " <lite model> <data_file> <calibrate_file>";
        return 0;
    }
    g_model_path = argv[1];
    g_data_file = argv[2];
    g_calibrator_file = argv[3];

	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
#else
int main(int argc, const char** argv) {
    return -1;
}
#endif
