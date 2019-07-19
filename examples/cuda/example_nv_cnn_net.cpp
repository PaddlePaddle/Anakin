
#include "utils/logger/logger.h"
#include "framework/graph/graph.h"
#include "framework/core/net/net.h"

#ifdef USE_CUDA
/*util to fill tensor*/
#include "saber/core/tensor_op.h"
using namespace anakin;
using namespace anakin::graph;
using namespace anakin::saber;

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    if (argc < 2) {
        LOG(ERROR) << "usage: ./" << argv[0] << " [model path] ";
        return 0;
    }
    const char* model_path = argv[1];
    /*init graph object, graph is the skeleton of model*/
    Graph<NV, Precision::FP32> graph;

    /*load model from file to init the graph*/
    auto status = graph.load(model_path);
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    /*set net input shape and use this shape to optimize the graph(fusion and init operator),shape is n,c,h,w*/
//    graph.Reshape("input_0", {1, 3, 224, 224});
    graph.Optimize();

    /*net_executer is the executor object of model. use graph to init Net*/
    Net<NV, Precision::FP32> net_executer(graph, true);

    /*use input string to get the input tensor of net. for we use NV as target, the tensor of net_executer is on GPU memory*/
    auto d_tensor_in_p = net_executer.get_in_list();
    for (auto& d_tensor : d_tensor_in_p) {
        auto valid_shape_in = d_tensor->valid_shape();

        /*create tensor located in host*/
        Tensor4d<X86> h_tensor_in;

        /*alloc for host tensor*/
        h_tensor_in.re_alloc(valid_shape_in);

        /*init host tensor by random*/
        fill_tensor_rand(h_tensor_in, -1.0f, 1.0f);

        /*use host tensor to int device tensor which is net input*/
        d_tensor->copy_from(h_tensor_in);
    }

    /*run infer*/
    net_executer.prediction();

    LOG(INFO)<<"infer finash";

    auto d_out=net_executer.get_out_list();
    /*get the out put of net, which is a device tensor*/
    for (auto& out : d_out) {
        /*create another host tensor, and copy the content of device tensor to host*/
        Tensor4d<X86> h_tensor_out;
        h_tensor_out.re_alloc(out->valid_shape());
        h_tensor_out.copy_from(*out);

        /*show output content*/
        for(int i = 0; i < h_tensor_out.valid_size(); i++) {
            LOG(INFO) << "out [" << i << "] = " << ((const float*)(h_tensor_out.data()))[i];
        }
    }
}
#else
int main(){return 0;}
#endif