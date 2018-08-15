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

void fill_request(int id, RPCRequest& request) {
    request.set_model("mobilenet_v2");
    request.set_request_id(id);
    int batch_size = 10;
    IO* input = request.add_inputs();
    Date* data = input->mutable_tensor();
    data->add_shape(1);
    data->add_shape(3);
    data->add_shape(224);
    data->add_shape(224);
    float new_tmp_data[1*3*224*224] = {1.f};
    input->mutable_data()->Reserve(1*3*224*224);
    memcpy(input->mutable_data()->mutable_data(), new_tmp_data, 3*224*224*sizeof(float));
}

TEST(ServiceTest, Service_client_base_test) {
    // A Channel represents a communication line to a Server. Notice that 
    // Channel is thread-safe and can be shared by all threads in your program.  
    brpc::Channel channel;
    
    // Initialize the channel, NULL means using default options.
    brpc::ChannelOptions options; 
    // Protocol type. Defined in src/brpc/options.proto
    options.protocol = "baidu_std"; 
    // Connection type. Available values: single, pooled, short
    options.connection_type = "single"; 
    // RPC timeout in milliseconds
    options.timeout_ms = 100 /*milliseconds*/; 
    // Max retries(not including the first RPC)
    options.max_retry = 3; 
    if (channel.Init("0.0.0.0:8000", "", &options) != 0) { 
        LOG(ERROR) << "Fail to initialize channel"; 
        return -1; 
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
        if(options.protocol != "http" && options.protocol != "h2c") {
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
                          << "): " << response->msg() 
                          << " latency = " << cntl.latency_us() <<" us";
            } else {
                LOG(INFO) << "I (" << cntl.local_side() << ") Received response from remote (" << cntl.remote_side()
                          << "): " << response->msg() 
                          << " (attached=" << cntl.response_attachment() << ") latency=" << cntl.latency_us() <<" us";
            }
        } else {
            LOG(WARNING) << cntl.ErrorText();
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
