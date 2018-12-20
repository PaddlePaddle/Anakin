#include "framework/graph/graph.cpp"
#include "framework/core/net/calibrator_parse.h"
#include "net_test.h"
int main(int argc, char** argv){

    std::string model_path = "";
    std::string config_name = "net_pt_config";
    std::string default_precision = "fp32";
    std::string default_target = "NV";
    if (argc<2){
        LOG(ERROR) << "usage: generate_calibrator_config model config_name config_prec config_target";
        LOG(FATAL) << "no model to generate config";
    }
    if (argc<3){
        LOG(ERROR) << "no config name, will use default name 'net_pt_config' ";
    }
    if (argc<4){
        LOG(ERROR) << "no config precision, will use default precision 'fp32' ";
    }
    if (argc<5){
        LOG(ERROR) << "no config target, will use default target 'NV' ";
    }
    
    if (argc>=2){
        model_path = std::string(argv[1]);
    }
    if (argc>=3){
        config_name = std::string(argv[2]);
    }
    if (argc>=4){
        default_precision = std::string(argv[3]);
    }
    if (argc>=5){
        default_target = std::string(argv[4]);
    }
#ifdef USE_CUDA   
    Graph<NV, Precision::FP32> graph;
#elif defined(USE_X86_PLACE)
    Graph<X86, Precision::FP32> graph;
#endif
#if defined USE_CUDA || defined USE_X86_PLACE
    graph.load(model_path);
    std::vector<std::string> node_names_in_order;
    std::vector<std::string> op_names;

    auto get_node_names = [&](NodePtr& node_ptr){
        node_names_in_order.push_back(node_ptr->name());
        op_names.push_back(node_ptr->get_op_name());
    };
    graph.Scanner->BFS(get_node_names);

    CalibratorParser parser;
    parser.auto_config(node_names_in_order, op_names, config_name, default_precision, default_target);
#endif
    return 0;
}
