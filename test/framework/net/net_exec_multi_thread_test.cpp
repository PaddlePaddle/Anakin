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
#elif defined(AMD_GPU)
using Target = AMD;
using Target_H = X86;
#endif
std::string model_path = "../benchmark/CNN/models/vgg16.anakin.bin";

#ifdef USE_CUDA
#if 0
TEST(NetTest, nv_net_execute_muti_thread_sync_test) {
#if 1 // use host input
    //Env<NV>::env_init(1);
    LOG(WARNING) << "Sync Runing multi_threads for model: " << model_path;
    Worker<NV, AK_FLOAT, Precision::FP32>  workers(model_path, 10); 
    workers.register_inputs({"input_0"});
    workers.register_outputs({"softmax_out"});    
    workers.Reshape("input_0", {1, 384, 960, 3});

    workers.launch();

    std::vector<Tensor4dPtr<target_host<NV>::type> > host_tensor_p_in_list;
    // get in
    saber::Shape valid_shape_in({1, 384, 960, 3});
    Tensor4dPtr<target_host<NV>::type> h_tensor_in = new Tensor4d<target_host<NV>::type, AK_FLOAT>(valid_shape_in);
    float* h_data = h_tensor_in->mutable_data();
    for (int i=0; i<h_tensor_in->size(); i++) {
        h_data[i] = 1.0f;
    }
    host_tensor_p_in_list.push_back(h_tensor_in);

    int epoch = 1000;

    // Running 
    for(int i=0; i<epoch; i++) {
        auto  d_tensor_p_out_list = workers.sync_prediction(host_tensor_p_in_list);

        // get the output
        auto d_tensor_p = d_tensor_p_out_list[0];
    }

    // get exec times
#ifdef ENABLE_OP_TIMER
    auto& times_map = workers.get_task_exe_times_map_of_sync_api();
    for (auto it = times_map.begin(); it!=times_map.end(); it++) {
        LOG(WARNING) << " threadId: " << it->first << " processing " << it->second.size() << " tasks";
        for (auto time_in_ms : it->second) { 
            LOG(INFO) << "      \\__task avg time: " << time_in_ms;
        }
    }
#endif

#endif

#if 0 // use device input
    Env<NV>::env_init(1);
    LOG(WARNING) << "Sync Runing multi_threads for model: " << model_path;
    Worker<NV, AK_FLOAT, Precision::FP32>  workers(model_path, 1); 
    workers.register_inputs({"input_0"});
    workers.register_outputs({"softmax_out"});    
    workers.Reshape("input_0", {1, 384, 960, 3});

    workers.launch();

    std::vector<Tensor4dPtr<target_host<NV>::type> > host_tensor_p_in_list;
    // get in
    saber::Shape valid_shape_in({1, 384, 960, 3});
    Tensor4dPtr<target_host<NV>::type> h_tensor_in = new Tensor4d<target_host<NV>::type, AK_FLOAT>(valid_shape_in);
    float* h_data = h_tensor_in->mutable_data();
    for (int i=0; i<h_tensor_in->size(); i++) {
        h_data[i] = 1.0f;
    }
    host_tensor_p_in_list.push_back(h_tensor_in);

    std::vector<Tensor4dPtr<NV> > device_tensor_p_in_list;
    for (int i=0; i<host_tensor_p_in_list.size(); i++) {
        Tensor4dPtr<NV> d_tensor_in = new Tensor4d<NV, AK_FLOAT>(host_tensor_p_in_list[i]->valid_shape());
        d_tensor_in->copy_from(*(host_tensor_p_in_list[i]));
        device_tensor_p_in_list.push_back(d_tensor_in);
    }

    int epoch = 10;

    // Running 
    for (int i=0; i<epoch; i++) {
        Context<NV> ctx(0, 0, 0);
        saber::SaberTimer<NV> my_time;

        my_time.start(ctx);
        auto  d_tensor_p_out_list = workers.sync_prediction_device(device_tensor_p_in_list);
        my_time.end(ctx);
        LOG(INFO)<<"muti thread single task exec time: "<<my_time.get_average_ms()/epoch << " ms";

        // get the output
        auto d_tensor_p = d_tensor_p_out_list[0];
    }
#endif

}
#endif

#if 1
TEST(NetTest, net_execute_muti_thread_async_test) {
    LOG(WARNING) << "Async Runing multi_threads for model: " << model_path;
    Worker<NV, Precision::FP32>  workers(model_path, 10); 
    //workers.register_inputs({"input_0"});
    //workers.register_outputs({"softmax_out"});    
    //workers.Reshape("input_0", {1, 384, 960, 3});

    workers.launch();

    std::vector<Tensor4dPtr<target_host<NV>::type> > host_tensor_p_in_list;
    // get in
    /*saber::Shape valid_shape_in({1, 384, 960, 3});
    Tensor4dPtr<target_host<NV>::type> h_tensor_in = new Tensor4d<target_host<NV>::type, AK_FLOAT>(valid_shape_in);
    float* h_data = h_tensor_in->mutable_data();
    for (int i=0; i<h_tensor_in->size(); i++) {
        h_data[i] = 1.0f;
    }
    host_tensor_p_in_list.push_back(h_tensor_in);*/

    int epoch = 10000;

    // Running 
    for(int i=0; i<epoch; i++) {
        workers.async_prediction(host_tensor_p_in_list);
    }

    // get the output
    int iterator = epoch;
    while(iterator) {
        if(!workers.empty()) {
            auto d_tensor_p = workers.async_get_result()[0];
            // get hte data of d_tensor_p
            
            iterator--;
        }
    }

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
