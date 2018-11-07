#include <string>
#include "graph_test.h"
#include "graph_base.h"
#include "graph.h"
#include "scheduler.h"

using namespace anakin;
using namespace anakin::graph;

std::string model_path = "/path/to/name.anakin.bin";


TEST(GraphTest, graph_load_model) {
    /*Graph<ARM, Precision::FP32>* graph = new Graph<ARM, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << model_path << " ...";
    // load anakin model files.
    graph->load(model_path);

    DLOG(INFO) << graph->to_string();
    // exec optimization
    graph->Optimize();  */
}

#ifdef USE_CUDA
TEST(GraphTest, nvidia_graph_save_model) {
    Graph<NV, Precision::FP32>* graph = new Graph<NV, Precision::FP32>();
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
#endif

#ifdef USE_X86_PLACE
TEST(GraphTest, x86_graph_save_model) {
    Graph<X86, Precision::FP32>* graph = new Graph<X86, Precision::FP32>();
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
#endif

#ifdef USE_ARM_PLACE
TEST(GraphTest, arm_graph_save_model) {
    Graph<ARM, Precision::FP32>* graph = new Graph<ARM, Precision::FP32>();
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
#endif

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
