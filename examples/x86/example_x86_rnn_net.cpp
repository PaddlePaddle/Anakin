
#include "utils/logger/logger.h"
#include "framework/graph/graph.h"
#include "framework/core/net/net.h"

#ifdef USE_X86_PLACE
/*util to fill tensor*/
#include "saber/core/tensor_op.h"
using namespace anakin;
using namespace anakin::graph;
using namespace anakin::saber;

int main(int argc, const char** argv) {
    /*init graph object, graph is the skeleton of model*/
    logger::init(argv[0]);
    if (argc < 2) {
        LOG(ERROR) << "usage: ./" << argv[0] << " [model path] ";
        return 0;
    }
    const char* model_path = argv[1];
    Graph<X86, Precision::FP32> graph;

    /*load model from file to init the graph*/
    auto status = graph.load(model_path);
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    /*set net input shape and use this shape to optimize the graph(fusion and init operator), shape is n,c,h,w. n=sum of words*/
//    graph.Reshape("input_0", {30, 1, 1, 1});
    graph.Optimize();

    /*net_executer is the executor object of model. use graph to init Net*/
    Net<X86, Precision::FP32> net_executer(graph, true);

    /*use input string to get the input tensor of net. for we use X86 as target, the tensor of net_executer is on host memory*/
    auto d_tensor_in_p = net_executer.get_in_list();
    for (auto& d_tensor : d_tensor_in_p) {
        /*init host tensor by random*/
        fill_tensor_rand(*d_tensor, -1.0f, 1.0f);
    }

    /*run infer*/
    net_executer.prediction();

    LOG(INFO)<<"infer finish";

    auto d_out=net_executer.get_out_list();
    /*get the out put of net, which is a device tensor*/
    for (auto& out : d_out) {
        /*show output content*/
        for(int i = 0; i < out->valid_size(); i++) {
            LOG(INFO) << "out [" << i << "] = " << ((const float*)(out->data()))[i];
        }
    }
}
#else
int main() {
    printf("nothing to do~~\n");
    return 0;
}
#endif