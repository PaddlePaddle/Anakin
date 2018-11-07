
#include "utils/logger/logger.h"
#include "graph.h"
#include "net.h"

#ifdef USE_CUDA
/*util to fill tensor*/
#include "saber/core/tensor_op.h"
using namespace anakin;
using namespace anakin::graph;
using namespace anakin::saber;

int main(int argc, const char** argv) {
    /*init graph object, graph is the skeleton of model*/
    Graph<NV, AK_FLOAT, Precision::FP32> graph;

    /*load model from file to init the graph*/
    auto status = graph.load("Resnet50.anakin.bin");

    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    /*set net input shape and use this shape to optimize the graph(fusion and init operator),shape is n,c,h,w*/
    graph.Reshape("input_0", {1, 3, 224, 224});
    graph.Optimize();

    /*net_executer is the executor object of model. use graph to init Net*/
    Net<NV, AK_FLOAT, Precision::FP32> net_executer(graph, true);

    /*use input string to get the input tensor of net. for we use NV as target, the tensor of net_executer is on GPU memory*/
    auto d_tensor_in_p = net_executer.get_in("input_0");
    auto valid_shape_in = d_tensor_in_p->valid_shape();

    /*create tensor located in host*/
    Tensor4d<X86, AK_FLOAT> h_tensor_in;

    /*alloc for host tensor*/
    h_tensor_in.re_alloc(valid_shape_in);

    /*init host tensor by random*/
    fill_tensor_host_rand(h_tensor_in, -1.0f, 1.0f);

    /*use host tensor to int device tensor which is net input*/
    d_tensor_in_p->copy_from(h_tensor_in);

    /*run infer*/
    net_executer.prediction();

    LOG(INFO) << "infer finash";

    /*get the out put of net, which is a device tensor*/
    auto d_out = net_executer.get_out("prob_out");

    /*create another host tensor, and copy the content of device tensor to host*/
    Tensor4d<X86, AK_FLOAT> h_tensor_out;
    h_tensor_out.re_alloc(d_out->valid_shape());
    h_tensor_out.copy_from(*d_out);

    /*show output content*/
    for (int i = 0; i < h_tensor_out.valid_size(); i++) {
        LOG(INFO) << "out [" << i << "] = " << h_tensor_out.data()[i];
    }
}
#else
int main() {}
#endif