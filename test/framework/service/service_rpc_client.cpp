#include <string>
#include "service_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include <brpc/channel.h>

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

std::string protocol = "baidu_std";
std::string server = "0.0.0.0:8000";

void fill_request(int id, RPCRequest& request) {
    request.set_model("mobilenet_v2");
    request.set_request_id(id);
    int batch_size = 1;
    IO* input = request.add_inputs();
    Data* data = input->mutable_tensor();
    data->add_shape(batch_size);
    data->add_shape(3);
    data->add_shape(224);
    data->add_shape(224);
    float new_tmp_data[batch_size*3*224*224];
    for(int i=0; i < batch_size*3*224*224; i++) {
        new_tmp_data[i] = 1.0f;
    }
    data->mutable_data()->Reserve(batch_size*3*224*224);
    LOG(WARNING) << " client data_p: " << data->mutable_data()->mutable_data() << " o: " 
        << request.inputs(0).tensor().data().data();
    //memcpy(data->mutable_data()->mutable_data(), new_tmp_data, 3*224*224*sizeof(float));
    for(int i=0; i<batch_size*3*224*224; i++) {
        request.inputs(0).tensor().add_data(1.0f);
        //LOG(INFO) << " |-- data[" << i << "]: " << request.inputs(0).tensor().data().data()[i];
    }
}

TEST(ServiceTest, Service_client_base_test) {
    // A Channel represents a communication line to a Server. Notice that 
    // Channel is thread-safe and can be shared by all threads in your program.  
    brpc::Channel channel;
    
    // Initialize the channel, NULL means using default options.
    brpc::ChannelOptions options; 
    // Protocol type. Defined in src/brpc/options.proto
    options.protocol = protocol; 
    // Connection type. Available values: single, pooled, short
    options.connection_type = "single"; 
    // RPC timeout in milliseconds
    options.timeout_ms = 10000 /*milliseconds*/; 
    // Max retries(not including the first RPC)
    options.max_retry = 0; 
    if (channel.Init(server.c_str(), &options) != 0) { 
        LOG(ERROR) << "Fail to initialize channel"; 
        return ; 
    }

    // Normally, you should not call a Channel directly, but instead construct
    // a stub Service wrapping it. stub can be shared by all threads as well.
    RPCService_Stub stub(&channel);

    // Send a request and wait for the response every 1 second
    int log_id = 0;
    while(!brpc::IsAskedToQuit()) {
        RPCRequest request;
        fill_request(log_id, request);
        RPCResponse response;
        brpc::Controller cntl;

        cntl.set_log_id(log_id++);  // set by user
        if(protocol != "http" && protocol != "h2c") {
            // Set attachment which is wired to network directly instead of
            // being serialized into protobuf messages.
            cntl.request_attachment().append("Hi, What's you name?"); // Carry this along with requests
        } else {
            cntl.http_request().set_content_type("application/json"); // Content type of http request
        }

        // Because `done'(last parameter) is NULL, this function waits until
        // the response comes back or error occurs(including timedout).
        stub.evaluate(&cntl, &request, &response, NULL);
        if (!cntl.Failed()) { 
            if (cntl.response_attachment().empty()) {
                LOG(INFO) << "I (" << cntl.local_side() << ") Received response from remote (" << cntl.remote_side() 
                          << "): " << response.info().msg() 
                          << " latency = " << cntl.latency_us() <<" us";
                for(int j =0; j<10; j++) {
                    LOG(WARNING) << "  \\__ get response data[" << j << "]: " 
                        << response.outputs(0).tensor().data(j);
                }
            } else {
                LOG(INFO) << "I (" << cntl.local_side() << ") Received response from remote (" << cntl.remote_side()
                          << "): " << response.info().msg() 
                          << " (attached=" << cntl.response_attachment() << ") latency=" << cntl.latency_us() <<" us";
            }
        } else {
            LOG(WARNING) << "Anakin Client Error==> "<< cntl.ErrorText() <<std::endl;
        }
        usleep(1000 * 1000L); // 1000 * 1000 us = 1 s
    }
    LOG(INFO) << "Client is going to quit.";
}

int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
