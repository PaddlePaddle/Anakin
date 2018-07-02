
#include "utils/logger/logger.h"
#include "graph.h"
#include "net.h"

#ifdef USE_X86_PLACE
/*util to fill tensor*/
#include "saber/core/tensor_op.h"
using namespace anakin;
using namespace anakin::graph;
using namespace anakin::saber;

int main(int argc, const char** argv) {
    /*init graph object, graph is the skeleton of model*/
    Graph<X86, AK_FLOAT, Precision::FP32> graph;

    /*load model from file to init the graph*/
    auto status = graph.load("language_model.anakin2.bin");
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    /*set net input shape and use this shape to optimize the graph(fusion and init operator), shape is n,c,h,w. n=sum of words*/
    graph.Reshape("input_0", {30, 1, 1, 1});
    graph.Optimize();

    /*net_executer is the executor object of model. use graph to init Net*/
    Net<X86, AK_FLOAT, Precision::FP32> net_executer(graph, true);

    /*use input string to get the input tensor of net. for we use X86 as target, the tensor of net_executer is on host memory*/
    auto h_tensor_in_p = net_executer.get_in("input_0");

    /*init host tensor by continue int*/
    fill_tensor_host_seq(*h_tensor_in_p);

    /*seq offset of tensor means offset of sentence, 0,10,15,30 means sentence0 = 0-9, sentence 1 =  10-14, sentence2 = 15-29*/
    h_tensor_in_p->set_seq_offset({0,10,15,30});


    /*run infer*/
    net_executer.prediction();

    LOG(INFO)<<"infer finash";

    /*get the out put of net, which is a host tensor*/
    auto h_out=net_executer.get_out("fc_1.tmp_2_out");


    /*show some output content*/
    for(int i=0;i<10;i++){
        LOG(INFO)<<"out ["<<i<<"] = "<<h_out->data()[i];
    }
}
#else
int main(){}
#endif