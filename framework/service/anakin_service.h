/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/

#ifndef ANAKIN_SERVICE_H
#define ANAKIN_SERVICE_H 

#include <brpc/server.h>

#include "framework/service/monitor.h"
#include "framework/core/net/worker.h"
#include "framework/service/api/service.pb.h"

namespace anakin {

namespace rpc {

template<typename Ttype, Precision Ptype, ServiceRunPattern RunP = ServiceRunPattern::SYNC>
class AnakinService : public RPCService {
public: 
    void evaluate(::google::protobuf::RpcController* controller_base, 
                  const RPCRequest* request, 
                  RPCResponse* response, 
                  ::google::protobuf::Closure* done) { 
        _evaluate(controller_base, request, response, done, ServiceRunPatternToType<RunP>());      
    }

public:
    void set_device_id(int dev_id);

    void initial(std::string model_name, std::string model_path, int thread_num);

    void launch();

    void Reshape(std::string model_name, std::string in_name, std::vector<int> in_shape);

    void register_inputs(std::string model_name, std::vector<std::string> in_names);

    void register_outputs(std::string model_name, std::vector<std::string>);

    void register_interior_edges(std::string model_name, std::string edge_start, std::string edge_end);

    template<typename functor, typename ...ParamTypes> 
    void register_aux_function(std::string model_name, functor function, ParamTypes ...args) {
        _worker_map[model_name].register_aux_function(function, std::forward<ParamTypes>(args)...);
    }

    template<Info ...infos>
    void create_monitor(int interval_time_in_sec) {
        _monitor.template create_instance<infos...>(_dev_id, interval_time_in_sec);
    }

private:
    void extract_request(const RPCRequest* request, 
                         std::vector<Tensor4d<typename target_host<Ttype>::type> >& inputs);
    void fill_response_data(int request_id, std::string model_name, 
                            RPCResponse* response, 
                            std::vector<Tensor4d<typename target_host<Ttype>::type> >& outputs);
    void fill_response_exec_info(RPCResponse* response);

private:
    void _evaluate(::google::protobuf::RpcController* controller_base, 
                          const RPCRequest* request, 
                          RPCResponse* response, 
                          ::google::protobuf::Closure* done,
                          ServiceRunPatternToType<ServiceRunPattern::SYNC>) {
        // make sure that done will be invoked
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(controller_base);
        // receive remote call from client.
        LOG(INFO) << "Received request[log_id=" << cntl->log_id() << "] from " << cntl->remote_side();
        if (!cntl->request_attachment().empty()) { 
            LOG(INFO) << " |-- (attached=" << cntl->request_attachment() << ")"; 
        }
        std::string model_name = request->model();
        int request_id = request->request_id();
        LOG(INFO) <<" |-- Get model: "<<model_name << " id: " << request_id; 
        std::vector<Tensor4d<typename target_host<Ttype>::type> > inputs;
        extract_request(request, inputs);
        auto ret = _worker_map[model_name]->sync_prediction(inputs);
        auto results = ret.get();
        LOG(ERROR) << "do infer over! thread id: " << std::this_thread::get_id();
        fill_response_data(request_id, model_name, response, results);
        fill_response_exec_info(response);
    }

    void _evaluate(::google::protobuf::RpcController* controller_base, 
                          const RPCRequest* request, 
                          RPCResponse* response, 
                          ::google::protobuf::Closure* done,
                          ServiceRunPatternToType<ServiceRunPattern::ASYNC>) {
        // make sure that done will be invoked
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(controller_base);
    }



private:
    std::unordered_map<std::string, std::shared_ptr<Worker<Ttype, Ptype, OpRunType::ASYNC> > > _worker_map;
    Monitor<Ttype> _monitor;
    int _dev_id;
};

} /* namespace rpc */

} /* namespace anakin */

#endif
