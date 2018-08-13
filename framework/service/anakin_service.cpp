#include "framework/service/anakin_service.h"

namespace anakin {

namespace rpc {

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void AnakinService<Ttype, Dtype, Ptype, RunType>::set_device_id(int dev_id) { }

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void AnakinService<Ttype, Dtype, Ptype, RunType>::initial(std::string model_name, 
                                                          std::string model_path, 
                                                          int thread_num) { 
    _worker_map[model_name] = std::make_shared<Ttype, Dtype, Ptype, RunType>(model_path, thread_num); 
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void AnakinService<Ttype, Dtype, Ptype, RunType>::register_inputs(std::string model_name, 
                                                                  std::vector<std::string>& in_names) {
    _worker_map[model_name].register_inputs(in_names);
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void AnakinService<Ttype, Dtype, Ptype, RunType>::register_outputs(std::string model_name, 
                                                                   std::vector<std::string>& out_names) {
    _worker_map[model_name].register_outputs(out_names);
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void AnakinService<Ttype, Dtype, Ptype, RunType>::Reshape(std::string model_name, 
                                                          std::string in_name, 
                                                          std::vector<int> in_shape) {
    _worker_map[model_name].Reshape(in_name, in_shape);
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void AnakinService<Ttype, Dtype, Ptype, RunType>::register_interior_edges(std::string model_name, 
                                                                          std::string edge_start, 
                                                                          std::string edge_end) {
    _worker_map[model_name].register_interior_edges(edge_start, edge_end);
}

} /* namespace rpc */

} /* namespace anakin */

