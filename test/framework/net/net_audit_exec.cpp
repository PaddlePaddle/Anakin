#include <string>
#include<random>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include "framework/core/mem_info.h"
#include <chrono>
#include "debug.h"
#include <fstream>
#include <sstream>
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
#endif

std::string g_model_path = "";
std::string g_input_path = "";

int g_batch_size=1;
int g_thread_num=1;
int g_warm_up = 10;
int g_epoch = 1000;

std::string model_saved_path = g_model_path + ".saved";

float Random(float low, float high) {
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(low, high);
  return dist(mt);
}

void fill_with_file(Tensor4d<Target>* d_tensor_in_p) {
    Tensor4d<Target_H> h_tensor_in;
    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i=0; i<valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    }
    h_tensor_in.re_alloc(valid_shape_in);
    float* h_data = (float*)(h_tensor_in.mutable_data());
    std::ifstream file(g_input_path.c_str(), std::ios::binary);
    for (int i=0; i<h_tensor_in.size() && !file.eof(); i++) {
        float tmp;
        file>>tmp;
        h_data[i] = tmp;
    }
    d_tensor_in_p->copy_from(h_tensor_in);
    file.close();
}

void fill_with_random(Tensor4d<Target>* d_tensor_in_p) {
    Tensor4d<Target_H> h_tensor_in;
    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i=0; i<valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    }
    h_tensor_in.re_alloc(valid_shape_in);
    float* h_data = (float*)(h_tensor_in.mutable_data());
    for (int i=0; i<h_tensor_in.size(); i++) {
        h_data[i] = Random(1, 128.0f);;
    }
    d_tensor_in_p->copy_from(h_tensor_in);
}

double InferencePerf(graph::Graph<Target, Precision::FP32>* graph, int thread_idx) {
    LOG(INFO) << "Thread (" << thread_idx << ") processing";
    // constructs the executer net
    Net<Target, Precision::FP32> net_executer(true);

    net_executer.init(*graph);

    // get ins
    for(auto& input_name : graph->get_ins()) {
        auto d_tensor_in_p = net_executer.get_in(input_name);

        if(g_input_path != std::string("")) {
            LOG(INFO) << "Use input file: " << g_input_path;
            fill_with_file(d_tensor_in_p);
        } else {
            fill_with_random(d_tensor_in_p);
        }
    }


    // do inference warm up
	for(int i = 0; i < g_warm_up; i++) {
		net_executer.prediction();
	}

    Context<Target> ctx(0, 0, 0);
    saber::SaberTimer<Target> my_time;
    my_time.start(ctx);
    double count = 0.f;

    for(int i = 0; i < g_epoch; i++) {
        saber::SaberTimer<Target> my_time;
        my_time.start(ctx);
        //auto t0 = std::chrono::high_resolution_clock::now();

        net_executer.prediction();

        //auto t1 = std::chrono::high_resolution_clock::now();
        //count += std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
        my_time.end(ctx);
        //LOG(INFO)<<"immed time : "<<my_time.get_average_ms() ;
        count += my_time.get_average_ms();
		if(i==100){
			double mem_used = anakin::MemoryInfo<Target>::Global().get_used_mem_in_mb();
			LOG(INFO) << "Checking_mem_used: " << mem_used;
		}
    }

    LOG(INFO)<<"InferencePerf aveage time: "<<count/g_epoch << " ms";

    // get out result
    //auto* tensor_out = net_executer.get_out("detection_output_0.tmp_0662");  // face 1
    //auto* tensor_out = net_executer.get_out("scale_0.tmp_0522");  // face 2
    //LOG(WARNING)<< "result : ";
	//test_print(tensor_out);
    //print_tensor_valid(*tensor_out);
    return count/g_epoch;
}

void InferencePerfWithMultiThread() {
    LOG(WARNING) << "Async Runing multi_threads for model: " << g_model_path;
    Graph<Target, Precision::FP32>* graph = new Graph<Target, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);
    if(!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    // reshape the input 's shape for graph model
    //graph->Reshape("data", {1, 3, 195, 758}); // face_box1
    //graph->Reshape("data", {1, 3, 227, 958}); // face_box1 not fusion
    //graph->Reshape("image", {1, 3, 210, 216}); // face_box2

    //anakin graph optimization
    graph->Optimize();

    // launch multi thread
    std::vector<std::thread> work_pool;
    double counter = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<g_thread_num; i++) {
        work_pool.emplace_back(InferencePerf, graph, i);
    }
    for(int i=0; i<g_thread_num; i++) {
        work_pool[i].join();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    counter = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    int QPS = g_epoch * g_thread_num / (counter / 1e6);
    LOG(ERROR) << " QPS : " << QPS;
    delete graph;
}

int main(int argc, const char** argv){
    if(argc < 4){
        LOG(INFO) << "@Anakin@ model audit";
        LOG(INFO) << "usage:";
        LOG(INFO) << "     Param 1:  thread_num      ( thread number )";
        LOG(INFO) << "     Param 2:  batch_size      ( batch size )";
        LOG(INFO) << "     Param 3:  model_path      ( anakin binary model file path )";
        LOG(INFO) << "     Param 4:  input_file_path ( anakin  input_file_path )";
        exit(-1);
    }
    g_thread_num = atoi(argv[1]);
    g_batch_size = atoi(argv[2]);
    g_model_path = argv[3];
    if(argc > 4) {
        g_input_path = argv[4];
    }

    Env<Target>::env_init();

    InferencePerfWithMultiThread();
    // initial logger
    //logger::init(argv[0]);
	//InitTest();
	//RUN_ALL_TESTS(argv[0]);
	return 0;
}
