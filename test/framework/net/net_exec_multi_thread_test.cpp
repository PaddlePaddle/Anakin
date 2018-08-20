#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>

#if defined(USE_CUDA)
using Target = NV;
using Target_H = X86;
#elif defined(USE_X86_PLACE)
using Target = X86;
using Target_H = X86;
#elif defined(USE_ARM_PLACE)
using Target = ARM;
using Target_H = ARM;
#endif
//std::string model_path = "../benchmark/CNN/models/vgg16.anakin.bin";
// std::string model_path = "/home/public/model_from_fluid/beta/demoprogram.anakin2.bin";
//std::string model_path = "benchmark/CNN/mobilenet_v2.anakin.bin";
std::string model_path = "/home/cuichaowen/anakin2/public_model/public-caffe-model/mobilenetv12/mobilenet_v2.anakin.bin";

#ifdef USE_CUDA
#if 0
TEST(NetTest, net_execute_single_test) {
    Graph<NV, AK_FLOAT, Precision::FP32>* graph = new Graph<NV, AK_FLOAT, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << model_path << " ...";
    // load anakin model files.
    auto status = graph->load(model_path);
    if(!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    //anakin graph optimization
    graph->Optimize();

    Net<NV, AK_FLOAT, Precision::FP32> net_executer(*graph, true);

    int epoch = 2;
    // do inference
    Context<NV> ctx(0, 0, 0);
    saber::SaberTimer<NV> my_time;
    LOG(WARNING) << "EXECUTER !!!!!!!! ";

    my_time.start(ctx);


    for(int i=0; i<epoch; i++) {
		LOG(ERROR) << " epoch(" << i << "/" << epoch << ") ";
    	// get in
    	auto d_tensor_in_p = net_executer.get_in("input_0");
    	Tensor4d<Target_H, AK_FLOAT> h_tensor_in;

    	auto valid_shape_in = d_tensor_in_p->valid_shape();
    	for (int i=0; i<valid_shape_in.size(); i++) {
    	    LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    	}

    	h_tensor_in.re_alloc(valid_shape_in);
    	float* h_data = h_tensor_in.mutable_data();

    	for (int i=0; i<h_tensor_in.size(); i++) {
    	    h_data[i] = 1.0f;
    	}

    	d_tensor_in_p->copy_from(h_tensor_in);

        net_executer.prediction();

		auto tensor_out_0_p = net_executer.get_out("prob_out");

		test_print(tensor_out_0_p);		
    }
}   
#endif 

#if 1
TEST(NetTest, net_execute_muti_thread_sync_test) {
    LOG(WARNING) << "Sync Runing multi_threads for model: " << model_path;
    Worker<NV, AK_FLOAT, Precision::FP32>  workers(model_path, 3); 
    workers.register_inputs({"input_0"});
    workers.register_outputs({"prob_out"});    
    workers.Reshape("input_0", {1, 3, 224, 224});

    workers.launch();

    std::vector<Tensor4d<target_host<NV>::type, AK_FLOAT> > host_tensor_in_list;
    // get in
    saber::Shape valid_shape_in({1, 3, 224, 224});
    Tensor4d<target_host<NV>::type, AK_FLOAT> h_tensor_in(valid_shape_in);

    float* h_data = h_tensor_in.mutable_data();
    for (int i=0; i<h_tensor_in.size(); i++) {
        h_data[i] = 1.0f;
    }
    host_tensor_in_list.push_back(h_tensor_in);

    int epoch = 2;
	
    // Running 
    for(int i=0; i<epoch; i++) {
        LOG(ERROR) << "epoch " << i << " / " << epoch;
        auto ret = workers.sync_prediction(host_tensor_in_list);
        auto results = ret.get();

        /*for(int j=0; j<10; j++) {
            LOG(INFO) << "Out data[" << j<<"]: " << results[0].mutable_data()[j];
        }*/
    }
}
#else 
TEST(NetTest, net_execute_muti_thread_async_test) {
    LOG(WARNING) << "Async Runing multi_threads for model: " << model_path;
    Worker<NV, AK_FLOAT, Precision::FP32>  workers(model_path, 10); 
    workers.register_inputs({"input_0"});
    workers.register_outputs({"prob_out"});    
    workers.Reshape("input_0", {1, 3, 224, 224});

    workers.launch();

    std::vector<Tensor4dPtr<target_host<NV>::type, AK_FLOAT> > host_tensor_p_in_list;
    // get in
    saber::Shape valid_shape_in({1, 3, 224, 224});
    Tensor4dPtr<target_host<NV>::type, AK_FLOAT> h_tensor_in = new Tensor4d<target_host<NV>::type, AK_FLOAT>(valid_shape_in);
    float* h_data = h_tensor_in->mutable_data();
    for (int i=0; i<h_tensor_in->size(); i++) {
        h_data[i] = 1.0f;
    }
    host_tensor_p_in_list.push_back(h_tensor_in);

    int epoch = 2000;

	std::thread check([&]() {
        int iterator = epoch;
    	while(iterator) {
            LOG(INFO) << "check work queue: " << iterator << "/" << epoch;
    	    if(!workers.empty()) {
    	        auto d_tensor_p = workers.async_get_result();//[0];
                LOG(WARNING) << "worker consume one";
    	        // get hte data of d_tensor_p
    	        
    	        iterator--;
    	    }
    	} 
	});

    // Running 
    for(int i=0; i<epoch; i++) {
        LOG(ERROR) << "epoch " << i << " / " << epoch;
        workers.async_prediction(host_tensor_p_in_list);
    }

	check.join();
}
#endif 

#endif

int main(int argc, const char** argv){

	Env<Target>::env_init();

    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
