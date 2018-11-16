#include <string>
#include "service_test.h"
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

//std::string mobilenet_v2_model_path = "/home/cuichaowen/anakin2/public_model/public-caffe-model/mobilenetv12/mobilenet_v2.anakin.bin";
std::string mobilenet_v2_model_path = "/home/chaowen/public-caffe-model/mobilenetv12/mobilenet_v2.anakin.bin";
int batchsize = 1;

int service_start(int port, int dev_id) {
    // create one server
    brpc::Server server;

    // instance anakin rpc service
    AnakinService<NV, Precision::FP32, ServiceRunPattern::SYNC> rpc_service;
    // device id must be set
    rpc_service.set_device_id(dev_id); 

    // initialize config for mobilenet v2
    rpc_service.initial("mobilenet_v2", mobilenet_v2_model_path, 3);
    rpc_service.register_inputs("mobilenet_v2", {"input_0"});
    rpc_service.Reshape("mobilenet_v2", "input_0", {batchsize, 3, 224, 224});
    rpc_service.register_outputs("mobilenet_v2", {"prob_out"});

    // create moniter for this service
    rpc_service.create_monitor<DEV_ID, DEV_NAME, DEV_TMP, DEV_MEM_FREE, DEV_MEM_USED>(3); // span 30 second

    
    // launch rpc service
    rpc_service.launch();

    // add service to server
    if (server.AddService(&rpc_service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) { 
        LOG(ERROR) << "Fail to add service"; 
        return -1; 
    }
    
    // Start the server
    brpc::ServerOptions options; 
    // Connection will be closed if there is no read/write operations during the time(s)
    options.idle_timeout_sec = 600; 
    // Max number thread of server
    options.num_threads = 10; 
    // Max concurrency request of server
    options.max_concurrency = 300; 

    if (server.Start(port, &options) != 0) { 
        LOG(ERROR) << "Fail to start Server on port "<< port << " device id " << dev_id; 
        return -1; 
    }

    // Wait until Ctrl-C is pressed, then Stop() and Join() the server
    server.RunUntilAskedToQuit();

    // server is stopped, you can release the source here
    return 0;
}

TEST(ServiceTest, Service_base_test) {
    // create anakin service deamon instance
    ServiceDaemon daemon_rpc;
    // launch daemon service for rpc [ on device 0 and port 8000]
    daemon_rpc(service_start, {0}, 8000);
}

int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
