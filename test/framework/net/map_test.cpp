#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>

std::string model_path = "benchmark/CNN/mobilenet_v2.anakin.bin";

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

int main(int argc, const char** argv){
	Env<NV>::env_init();

    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
