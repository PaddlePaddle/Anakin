#include <string>
#include "graph_test.h"
#include "graph_base.h"
#include "graph.h"
#include "scheduler.h"

using namespace anakin;
using namespace anakin::graph;

//std::string model_path = "/home/chaowen/anakin_v2/model_v2/google_net/googlenet.anakin.bin";
std::string model_path = "/home/chaowen/anakin_v2/model_v2/yolo/yolo.anakin.bin";


TEST(GraphTest, graph_load_model) {
    /*Graph<ARM, float, Precision::FP32>* graph = new Graph<ARM, float, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << model_path << " ...";
    // load anakin model files.
    graph->load(model_path);

    DLOG(INFO) << graph->to_string();
    // exec optimization
    graph->Optimize();  */
}

TEST(GraphTest, graph_save_model) {
#ifdef USE_CUDA
    Graph<NV, AK_FLOAT, Precision::FP32>* graph = new Graph<NV, AK_FLOAT, Precision::FP32>();
#endif
#ifdef USE_X86_PLACE
    Graph<X86, AK_FLOAT, Precision::FP32>* graph = new Graph<X86, AK_FLOAT, Precision::FP32>();
#endif
#ifdef USE_ARM_PLACE
    Graph<ARM, AK_FLOAT, Precision::FP32>* graph = new Graph<ARM, AK_FLOAT, Precision::FP32>();
#endif
    // load anakin model files.
    LOG(INFO) << "load anakin model file from " << model_path << " ...";
    graph->load(model_path);

    // regisiter output tensor
    //graph->RegistOut("data_perm",  "data_scale");

    //  exec optimization
    graph->Optimize();

    // save the optimized model to disk.
    std::string save_model_path = model_path + std::string(".saved");
    Status status = graph->save(save_model_path);
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
