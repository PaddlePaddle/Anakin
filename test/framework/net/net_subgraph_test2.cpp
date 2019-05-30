#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include "debug.h"
#include <fstream>
#if defined(USE_CUDA)
using Target = NV;
using Target_H = X86;
#elif defined(USE_X86_PLACE)
using Target = X86;
using Target_H = X86;
#elif defined(USE_ARM_PLACE)
using Target = ARM;
using Target_H = ARM;
#elif defined(AMD_GPU)
using Target = AMD;
using Target_H = X86;
#elif defined(USE_MLU)
using Target = MLU;
using Target_H = MLUHX86;
#elif defined(USE_BM_PLACE)
using Target = BM;
using Target_H = BMX86;
#endif

std::string g_model_path = "/home/cuichaowen/baidu/Anakin-2.0/buil/not_fuse_before_net_init.bin"; 

TEST(NetTest, net_execute_base_test) {
    Graph<Target, Precision::FP32>* graph = new Graph<Target, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);
    if(!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    LOG(INFO)<<"net_execute_base_test";

#if 0
    graph->Reshape("data", {1, 3, 227, 958});       // right results
#else
    graph->Reshape("data", {1, 3, 1500, 1500});     // wrong results
#endif

    graph->Optimize();

    Net<Target, Precision::FP32> net_executer(true);

    net_executer.init(*graph);

    auto d_tensor_in_p = net_executer.get_in("data");

    d_tensor_in_p->reshape(Shape({1, 3, 227, 958}));
    Tensor4d<Target_H> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i=0; i<valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    float* h_data = (float*)(h_tensor_in.mutable_data());

    for (int i=0; i<h_tensor_in.size(); i++) {
        h_data[i] = 1.0f;
    }

    d_tensor_in_p->copy_from(h_tensor_in);

    net_executer.prediction();

    auto* tensor_out_0_p = net_executer.get_out("detection_output_0.tmp_0662");
    print_tensor_valid(*tensor_out_0_p);


    delete graph;
}



int main(int argc, const char** argv){
	Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
