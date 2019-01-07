#include "framework/model_parser/parser/model_io.h"
#include "framework/model_parser/parser/parser.h"
#include "test/framework/net/net_test.h"
#include "utils/unit_test/net_info.h"

std::string g_load_path;
std::string g_save_path;

Status read_cmd(int argc, char** argv) {
    Status ret = Status::OK();
    if (argc < 2) {
        LOG(FATAL) << "Usage: ./generate_debug_info load_path save_path";
    }
    if (argc > 1) {
        g_load_path = std::string(argv[1]);
    }
    if (argc > 2) {
        g_save_path = std::string(argv[2]);
    }
    return ret;
}

int main(int argc, char** argv) {

    read_cmd(argc, argv);
    Graph<Target, Precision::FP32>* graph = new Graph<Target, Precision::FP32>;
    auto status = graph->load(g_load_path);
    CHECK_NE(status, 0) << "Graph loading error.";
    graph->Optimize();

    Net<Target, Precision::FP32>* net = new Net<Target, Precision::FP32>(true);
    net->init(*graph);

    NetProto net_proto;
    fill_net_proto(net, graph, net_proto);

    std::fstream o_stream(g_save_path, std::ios::out | std::ios::trunc | std::ios::binary);
    net_proto.SerializeToOstream(&o_stream);

    return 0;
}