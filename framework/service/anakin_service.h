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

#include "framework/core/net/worker.h"
#include "framework/core/net/worker.h"
#include "framework/service/api/service.pb.h"

namespace anakin {

namespace rpc {

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType = OpRunType::ASYNC>
class AnakinService public: RPCService {
public: 
    void evaluate(::google::protobuf::RpcController* controller_base, 
                  const RPCRequest* request, 
                  RPCResponse* response, 
                  ::google::protobuf::Closure* done) { 
        _evaluate(controller_base, request, response, done, EnumToType<RunType>());      
    }

public:
    inline void set_device_id(int dev_id);

    inline void initial(std::string model_name, std::string model_path, int thread_num);

    inline void Reshape(std::string model_name, std::string in_name, std::vector<int> in_shape);

    inline void register_inputs(std::string model_name, std::vector<std::string> in_names);

    inline void register_outputs(std::string model_name, std::vector<std::string>);

    inline void register_interior_edges(std::string model_name, std::string edge_start, std::string edge_end);

    template<typename functor, typename ...ParamTypes> 
    void register_aux_function(std::string model_name, functor function, ParamTypes ...args) {
        _worker_map[model_name].register_aux_function(function, std::forward<ParamTypes>(args)...);
    }

private:
    inline void _evaluate(::google::protobuf::RpcController* controller_base, 
                          const RPCRequest* request, 
                          RPCResponse* response, 
                          ::google::protobuf::Closure* done) {
        // make sure that done will be invoked
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(controller_base);
    }

private:
    std::unordered_map<std::string, std::shared_ptr<Worker<Ttype, Dtype, Ptype, RunType> > > _worker_map;
};

} /* namespace rpc */

} /* namespace anakin */

#endif
