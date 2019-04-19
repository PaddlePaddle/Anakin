#include "framework/graph/graph.h"
#include "framework/core/net/calibrator_parse.h"
#include "net_test.h"
int main(int argc, char** argv){

    std::string model_path = "";
    std::string config_name = "model_layout_config";
    if (argc < 2) {
        LOG(ERROR) << "usage: generate_layout_config model config_name";
        LOG(FATAL) << "no model to generate config";
    }
    if (argc < 3) {
        LOG(ERROR) << "no config name, will use default name 'model_layout_config' ";
    }
    if (argc >= 2) {
        model_path = std::string(argv[1]);
    }
    if (argc >= 3) {
        config_name = std::string(argv[2]);
    }
#ifdef USE_CUDA
    Graph<NV, Precision::FP32> graph;
    using Ttype = NV;
#elif defined(USE_X86_PLACE)
    Graph<X86, Precision::FP32> graph;
    using Ttype = X86;
#endif
#if defined USE_CUDA || defined USE_X86_PLACE
    graph.load(model_path);
    std::vector<std::string> edge_names_in_order;
    std::vector<LayoutType>  edge_layouts;

    auto get_edge_names = [&](Edge<Ttype>& edge){
        edge_names_in_order.push_back(edge.name());
        edge_layouts.push_back(edge.layout());
    };
    graph.Scanner->BFS_Edge(get_edge_names);

    CalibratorParser parser;
    parser.auto_config_layout(edge_names_in_order, edge_layouts, config_name);
#endif
    return 0;
}

