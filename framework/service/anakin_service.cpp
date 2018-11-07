#include "framework/service/anakin_service.h"

namespace anakin {

namespace rpc {

template<typename Ttype, Precision Ptype, ServiceRunPattern RunP>
void AnakinService<Ttype, Ptype, RunP>::set_device_id(int dev_id) {
    _dev_id = dev_id;
    saber::TargetWrapper<Ttype>::set_device(_dev_id);
}

template<typename Ttype, Precision Ptype, ServiceRunPattern RunP>
void AnakinService<Ttype, Ptype, RunP>::initial(std::string model_name,
        std::string model_path,
        int thread_num) {
    _worker_map[model_name] = std::make_shared<Worker<Ttype, Ptype, OpRunType::ASYNC> >(model_path,
                              thread_num);
}

template<typename Ttype, Precision Ptype, ServiceRunPattern RunP>
void AnakinService<Ttype, Ptype, RunP>::launch() {
    for (auto it = _worker_map.begin(); it != _worker_map.end();) {
        it->second->launch();
        it++;
    }
}

template<typename Ttype, Precision Ptype, ServiceRunPattern RunP>
void AnakinService<Ttype, Ptype, RunP>::register_inputs(std::string model_name,
        std::vector<std::string> in_names) {
    _worker_map[model_name]->register_inputs(in_names);
}

template<typename Ttype, Precision Ptype, ServiceRunPattern RunP>
void AnakinService<Ttype, Ptype, RunP>::register_outputs(std::string model_name,
        std::vector<std::string> out_names) {
    _worker_map[model_name]->register_outputs(out_names);
}

template<typename Ttype, Precision Ptype, ServiceRunPattern RunP>
void AnakinService<Ttype, Ptype, RunP>::Reshape(std::string model_name,
        std::string in_name,
        std::vector<int> in_shape) {
    _worker_map[model_name]->Reshape(in_name, in_shape);
}

template<typename Ttype, Precision Ptype, ServiceRunPattern RunP>
void AnakinService<Ttype, Ptype, RunP>::register_interior_edges(std::string model_name,
        std::string edge_start,
        std::string edge_end) {
    _worker_map[model_name]->register_interior_edges(edge_start, edge_end);
}

template<typename Ttype, Precision Ptype, ServiceRunPattern RunP>
inline void AnakinService<Ttype, Ptype, RunP>::extract_request(
    const RPCRequest* request,
    std::vector<Tensor4d<typename target_host<Ttype>::type> >& inputs) {
    for (int i = 0; i < request->inputs_size(); i++) {
        LOG(INFO) << "Get " << i << "input";
        auto& io = request->inputs(i);
        auto& data = io.tensor();
        auto& shape = data.shape();
        saber::Shape tensor_shape({shape[0], shape[1], shape[2], shape[3]});
        Tensor4d<typename target_host<Ttype>::type> h_tensor;
        h_tensor.re_alloc(tensor_shape);
        float* h_data = (float*)(h_tensor.mutable_data());
        DLOG(INFO) << "Check shape: " << shape[0] << " " << shape[1] << " " << shape[2] << " " << shape[3];
        DLOG(INFO) << "h_data: " << h_data << "data_p: " << data.data().data();

        for (int j = 0; j < 10; j++) {
            LOG(INFO) << "  \\__ request data[" << j << "]: " << data.data(j);
        }

        memcpy(h_data, data.data().data(), shape[0]*shape[1]*shape[2]*shape[3]*h_tensor.get_dtype_size());

        for (int j = 0; j < 10; j++) {
            LOG(WARNING) << "  \\__ copy to inputs data[" << j << "]: " << h_data[j];
        }

        inputs.push_back(h_tensor);
    }
}

template<typename Ttype, Precision Ptype, ServiceRunPattern RunP>
inline void AnakinService<Ttype, Ptype, RunP>::fill_response_data(
    int request_id,
    std::string model_name,
    RPCResponse* response,
    std::vector<Tensor4d<typename target_host<Ttype>::type> >& outputs) {
    response->set_model(model_name);
    response->set_request_id(request_id);
    int count = 0;

    for (auto & h_out : outputs) {
        LOG(INFO) << "Get " << count << " output";
        count++;
        // copy to host
        auto shape = h_out.valid_shape();
        // fill response
        IO* output = response->add_outputs();
        Data* data = output->mutable_tensor();
        data->add_shape(shape[0]);
        data->add_shape(shape[1]);
        data->add_shape(shape[2]);
        data->add_shape(shape[3]);
        data->mutable_data()->Reserve(shape[0]*shape[1]*shape[2]*shape[3]);

        for (int j = 0; j < shape[0]*shape[1]*shape[2]*shape[3]; j++) {
            data->add_data(((float*)(h_out.mutable_data()))[j]);
        }

        LOG(INFO) << " output size: " << data->data_size();
    }
}

template<typename Ttype, Precision Ptype, ServiceRunPattern RunP>
inline void AnakinService<Ttype, Ptype, RunP>::fill_response_exec_info(RPCResponse* response) {
    auto* info = response->mutable_info();
    info->set_msg("SUC");
    DeviceStatus* status_p =  info->mutable_device_status();
    status_p->set_id(_monitor.get_id());
    status_p->set_name(_monitor.get_name());
    status_p->set_temp(_monitor.get_temp());
    status_p->set_mem_free(_monitor.get_mem_free());
    status_p->set_mem_used(_monitor.get_mem_used());
    info->set_duration_in_nano_seconds(-1);
}

#ifdef USE_CUDA
template class AnakinService<NV, Precision::FP32, ServiceRunPattern::ASYNC>;
template class AnakinService<NV, Precision::FP16, ServiceRunPattern::ASYNC>;
template class AnakinService<NV, Precision::INT8, ServiceRunPattern::ASYNC>;

template class AnakinService<NV, Precision::FP32, ServiceRunPattern::SYNC>;
template class AnakinService<NV, Precision::FP16, ServiceRunPattern::SYNC>;
template class AnakinService<NV, Precision::INT8, ServiceRunPattern::SYNC>;
#endif

#ifdef USE_X86_PLACE
template class AnakinService<X86, Precision::FP32, ServiceRunPattern::ASYNC>;
template class AnakinService<X86, Precision::FP16, ServiceRunPattern::ASYNC>;
template class AnakinService<X86, Precision::INT8, ServiceRunPattern::ASYNC>;

template class AnakinService<X86, Precision::FP32, ServiceRunPattern::SYNC>;
template class AnakinService<X86, Precision::FP16, ServiceRunPattern::SYNC>;
template class AnakinService<X86, Precision::INT8, ServiceRunPattern::SYNC>;
#endif

#ifdef AMD_GPU
template class AnakinService<AMD, Precision::FP32, ServiceRunPattern::ASYNC>;
template class AnakinService<AMD, Precision::FP16, ServiceRunPattern::ASYNC>;
template class AnakinService<AMD, Precision::INT8, ServiceRunPattern::ASYNC>;

template class AnakinService<AMD, Precision::FP32, ServiceRunPattern::SYNC>;
template class AnakinService<AMD, Precision::FP16, ServiceRunPattern::SYNC>;
template class AnakinService<AMD, Precision::INT8, ServiceRunPattern::SYNC>;
#endif

#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
template class AnakinService<ARM, Precision::FP32, ServiceRunPattern::ASYNC>;
template class AnakinService<ARM, Precision::FP32, ServiceRunPattern::SYNC>;
#endif

#ifdef ANAKIN_TYPE_FP16
template class AnakinService<ARM, Precision::FP16, ServiceRunPattern::ASYNC>;
template class AnakinService<ARM, Precision::FP16, ServiceRunPattern::SYNC>;
#endif

#ifdef ANAKIN_TYPE_INT8
template class AnakinService<ARM, Precision::INT8, ServiceRunPattern::ASYNC>;
template class AnakinService<ARM, Precision::INT8, ServiceRunPattern::SYNC>;
#endif //int8

#endif //arm


} /* namespace rpc */

} /* namespace anakin */

