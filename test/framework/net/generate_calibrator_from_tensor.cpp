#include <string>
#include "net_test.h"
#include "framework/core/net/entropy_calibrator.h"
#include "saber/funcs/timer.h"
#include <chrono>
#if defined(NVIDIA_GPU)|| defined(USE_X86_PLACE)

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

Tensor<X86> g_tensor;
Shape g_shape;
std::vector<std::vector<int>> g_seq_offset;
Tensor<X86>* data_producer() {
    static int cnt = 0;
    const int data_num = 5;
    cnt++;
    g_tensor.reshape(g_shape);
    fill_tensor_const(g_tensor, 1.f);
    g_tensor.set_seq_offset(g_seq_offset);

    if (cnt <= data_num) {
        return &g_tensor;
    } else {
        return nullptr;
    }
}
TEST(NetTest, calibrator) {
    Graph<Target, Precision::FP32>* graph = new Graph<Target, Precision::FP32>();
    // load anakin model files.
    auto status = graph->load(g_model_path);

    if (!status) {
        delete graph;
        LOG(FATAL) << " [ERROR] " << status.info();
        exit(-1);
    }

    auto input_names = graph->get_ins();
    graph->ResetBatchSize(input_names[0], g_batch_size);
    //anakin graph optimization
    graph->Optimize(false);
    // constructs the executer net
    g_seq_offset.push_back({0, g_batch_size});

    Net<Target, Precision::FP32, OpRunType::SYNC> net_executer(*graph);
    g_shape = net_executer.get_in(input_names[0])->valid_shape();
    BatchStream<Target> batch_stream(data_producer);
    EntropyCalibrator<Target> entropy_calibrator(&batch_stream, g_batch_size, g_calibrator_file,
            &net_executer, g_bin_num);
    entropy_calibrator.generate_calibrator_table();

    delete graph;

}


int main(int argc, const char** argv) {

    Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);

    LOG(INFO) << "usage:";
    LOG(INFO) << argv[0] << " <lite model> <data_file> <calibrate_file>";
    LOG(INFO) << "   lite_model:     path to anakin lite model";
    LOG(INFO) << "   data_file:      path to image data list";
    LOG(INFO) << "   calibrate file: path to calibrate data path";

    if (argc < 5) {
        LOG(ERROR) << "useage: " << argv[0] << " <lite model> <data_file> <calibrate_file>";
        return 0;
    }

    g_model_path = argv[1];
    g_data_file = argv[2];
    g_calibrator_file = argv[3];
    g_batch_size = atoi(argv[4]);

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
#else
int main(int argc, const char** argv) {
    return 0;
}
#endif
