
#include <string>
#include "graph_base.h"
#include "graph.h"
#include <iostream>
#include "utils/unit_test/aktest.h"
#include "utils/logger/logger.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include "debug.h"
#include <fstream>

#ifdef USE_TENSORRT
#include "rt_net.h"
using namespace anakin;
using ::anakin::test::Test;

using namespace anakin::graph;
std::string g_model_path = "/path/to/your/anakin_model";

std::string model_saved_path = g_model_path + ".saved";
int g_batch_size = 1;
int g_warm_up = 10;
int g_epoch = 1000;
int g_device_id = 0;


void rt_net_test() {
    Graph<X86, Precision::FP32>* graph = new Graph<X86, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);
    if(!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

	graph->ResetBatchSize("input_0", g_batch_size);

    graph->Optimize(true);

    RTNet net_executer(*graph, NULL);

    // get in
    auto d_tensor_in_p = net_executer.get_in("input_0");
    Tensor4d<X86> h_tensor_in;

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


    //int g_epoch = 1000;
    //int g_warm_up=10;
    // do inference
    LOG(WARNING) << "EXECUTER !!!!!!!! ";
	// warm up
	for(int i = 0; i < g_warm_up; i++) {
		net_executer.prediction();
	}

    //auto start = std::chrono::system_clock::now();
    for(int i = 0; i < g_epoch; i++) {
		//DLOG(ERROR) << " g_epoch(" << i << "/" << g_epoch << ") ";
        net_executer.prediction();
    }
    cudaDeviceSynchronize();

    //write_tensorfile(*net_executer.get_out_list()[0],"output.txt");

	LOG(ERROR) << "inner net exe over !";
    for(auto x:net_executer.get_out_list()){
         print_tensor(*x);
    }
    // save the optimized model to disk.
    if (!status ) { 
        LOG(FATAL) << " [ERROR] " << status.info(); 
    }
    if (!graph){
        delete graph;
    }
}
#endif

int main(int argc, const char** argv){
    if (argc < 2){
        LOG(ERROR)<<"no input!!!";
        return;
    }
#ifdef USE_TENSORRT
    if (argc > 1) {
        g_model_path = std::string(argv[1]);
    }
    if (argc > 2) {
        g_batch_size = atoi(argv[2]);
    }
    if (argc > 3) {
        g_warm_up = atoi(argv[3]);
    }
    if (argc > 4) {
        g_epoch = atoi(argv[4]);
    }
    if (argc > 5) {
        g_device_id = atoi(argv[5]);
    }
    cudaSetDevice(g_device_id);
    // initial logger
    logger::init(argv[0]);
    rt_net_test();
#endif
	return 0;
}
