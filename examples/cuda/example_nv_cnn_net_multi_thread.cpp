
#include "utils/logger/logger.h"
#include "framework/graph/graph.h"
#include "framework/core/net/net.h"

/*worker is anakin thread pool*/
#include "framework/core/net/worker.h"

/*util to fill tensor*/
#include "saber/core/tensor_op.h"
#ifdef USE_CUDA
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
    /*init works object by model path and thread pool size*/
    Worker<NV, Precision::FP32>  workers(model_path, 10);
    workers.register_inputs({"input_0"});
    workers.register_outputs({"prob_out"});
    /*set input shape*/
//    workers.Reshape("input_0", {1, 3, 224, 224});
    /*start workers*/
    workers.launch();

    /*fill input*/
    std::vector<Tensor4d<target_host<NV>::type>> host_tensor_p_in_list;
    saber::Shape valid_shape_in({1, 3, 224, 224});
    Tensor4d<target_host<NV>::type> h_tensor_in(valid_shape_in);
    float* h_data = static_cast<float*>(h_tensor_in.mutable_data());
    for (int i = 0; i < h_tensor_in.valid_size(); i++) {
        h_data[i] = 1.0f;
    }
    host_tensor_p_in_list.push_back(h_tensor_in);


    /*run infer，send input to worker queue*/
    int epoch = 1000;
    for(int i = 0; i < epoch; i++) {
        auto  d_tensor_p_out_list = workers.sync_prediction(host_tensor_p_in_list);
//        auto d_tensor_p = d_tensor_p_out_list[0];
    }
    LOG(INFO) << "info finish";

}
#else
#include "stdio.h"
int main(){printf("nothing happened -_-!!\n"); return 0;}
#endif