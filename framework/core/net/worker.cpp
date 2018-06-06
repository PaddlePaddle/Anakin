#include "framework/core/net/worker.h"
#include "saber/funcs/timer.h"

namespace anakin {

//! \brief a model map between thread_id and net model
template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
struct NetGraphWrapper {
    typedef std::thread::id key;

    void initial(std::string model_path, std::unordered_map<std::string, std::vector<int>>& shape_map) EXCLUSIVE_LOCKS_REQUIRED(this->_mut) {
        std::lock_guard<std::mutex> guard(this->_mut);
        if(_graph_map.count(model_path) <= 0) {
            // graph load is thread safe
            _graph_map[model_path].load(model_path);
            for(auto it = shape_map.begin(); it != shape_map.end();) {
                // thread safe
                _graph_map[model_path].Reshape(it->first, it->second);
                ++it;
            }
            // thread safe
            _graph_map[model_path].Optimize();
            {// make sure thread safety
                key id = std::this_thread::get_id();
                LOG(INFO) << "CURRENT thread ID : " << id;
                if(_thread_to_net.find(id) == _thread_to_net.end()) {
                    _thread_to_net[id].init(_graph_map[model_path]);
                }
            }
        } else {
            key id = std::this_thread::get_id(); 
            LOG(INFO) << "CURRENT thread ID : " << id; 
            if (_thread_to_net.find(id) == _thread_to_net.end()) { 
                _thread_to_net[id].init(_graph_map[model_path]); 
            }
        }
    }

    inline Net<Ttype, Dtype, Ptype, RunType>& get_net(key id) {
        if(_thread_to_net.find(id) != _thread_to_net.end()) { 
            return _thread_to_net[id];
        }
        LOG(FATAL) << " target key(thread_id) not found in NetGraphWrapper";
    }
    
private:
    std::unordered_map<std::string, graph::Graph<Ttype, Dtype, Ptype>> _graph_map;
    std::unordered_map<key, Net<Ttype, Dtype, Ptype, RunType>> _thread_to_net GUARDED_BY(this->_mut);
    std::mutex _mut;
};

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
using MultiThreadModel = Singleton<NetGraphWrapper<Ttype, Dtype, Ptype, RunType>>;

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
Worker<Ttype, Dtype, Ptype, RunType>::Worker(std::string model_path, int num_thread) : _model_path(model_path), ThreadPool(num_thread) {}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
Worker<Ttype, Dtype, Ptype, RunType>::~Worker() {}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Dtype, Ptype, RunType>::pause(size_t time) {
    std::function<void(int)> sleep = [](size_t time) {
        std::this_thread::sleep_for(std::chrono::milliseconds(time));
    };
    this->RunSync(sleep, time);
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Dtype, Ptype, RunType>::Reshape(std::string in_name, std::vector<int> new_shape) {
    _in_shapes[in_name] = new_shape;
}


template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Dtype, Ptype, RunType>::register_inputs(std::vector<std::string> input_names) {
    _inputs_in_order = input_names;
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Dtype, Ptype, RunType>::register_outputs(std::vector<std::string> output_names) {
    _outputs_in_order = output_names;
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType> 
void Worker<Ttype, Dtype, Ptype, RunType>::register_interior_edges(std::string bottom, std::string top) {
    graph::Arc<std::string, int> arc(bottom, top);
    _edges_in_order.push_back(arc);
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
std::vector<Tensor4dPtr<Ttype, Dtype> > Worker<Ttype, Dtype, Ptype, RunType>::sync_prediction(std::vector<Tensor4dPtr<typename target_host<Ttype>::type, Dtype> >& net_ins_list) {
    auto task = [&](std::vector<Tensor4dPtr<typename target_host<Ttype>::type, Dtype> >& ins) -> std::vector<Tensor4dPtr<Ttype, Dtype> > {
        auto& net = MultiThreadModel<Ttype, Dtype, Ptype, RunType>::Global().get_net(std::this_thread::get_id()); 
        //fill the graph inputs

        for(int i = 0; i < _inputs_in_order.size(); i++) { 
            auto d_tensor_in_p = net.get_in(_inputs_in_order[i]);
            d_tensor_in_p->reshape(ins[i]->valid_shape());
            d_tensor_in_p->copy_from(*ins[i]);
            d_tensor_in_p->set_seq_offset(ins[i]->get_seq_offset());
        } 
#ifdef ENABLE_OP_TIMER
        Context<Ttype> ctx(0, 0, 0); 
        saber::SaberTimer<Ttype> my_time;
        my_time.start(ctx);
#endif
        net.prediction(); 

#ifdef ENABLE_OP_TIMER
        my_time.end(ctx); 
        {
            std::lock_guard<std::mutex> guard(_mut); 
            _thead_id_to_prediction_times_vec_in_ms[std::this_thread::get_id()].push_back(my_time.get_average_ms());
        }
#endif
        // get outputs of graph
        std::vector<Tensor4dPtr<Ttype, Dtype>> ret;
        for (auto out : _outputs_in_order) {
            auto d_tensor_out_p = net.get_out(out);
            ret.push_back(d_tensor_out_p);
        }

        return ret; 
    };
    return this->RunSync(task, net_ins_list);
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
std::vector<Tensor4dPtr<Ttype, Dtype> > Worker<Ttype, Dtype, Ptype, RunType>::sync_prediction_device(std::vector<Tensor4dPtr<Ttype, Dtype> >& net_ins_list) {
    auto task = [&](std::vector<Tensor4dPtr<Ttype, Dtype> >& ins) -> std::vector<Tensor4dPtr<Ttype, Dtype> > {
        auto& net = MultiThreadModel<Ttype, Dtype, Ptype, RunType>::Global().get_net(std::this_thread::get_id()); 
        //fill the graph inputs 
        for (int i = 0; i < _inputs_in_order.size(); i++) { 
            auto d_tensor_in_p = net.get_in(_inputs_in_order[i]); 
            d_tensor_in_p->copy_from(*ins[i]); 
        } 
        net.prediction(); 
        // get outputs of graph
        std::vector<Tensor4dPtr<Ttype, Dtype>> ret;
        for (auto out : _outputs_in_order) {
            auto d_tensor_out_p = net.get_out(out);
            ret.push_back(d_tensor_out_p);
        }

        return ret; 
    }; 
    return this->RunSync(task, net_ins_list);
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Dtype, Ptype, RunType>::async_prediction(std::vector<Tensor4dPtr<typename target_host<Ttype>::type, Dtype> >& net_ins_list) {
    std::lock_guard<std::mutex> guard(this->_async_que_mut);    
    auto task = [&](std::vector<Tensor4dPtr<typename target_host<Ttype>::type, Dtype> >& ins) -> std::vector<Tensor4dPtr<Ttype, Dtype> > {
            auto& net = MultiThreadModel<Ttype, Dtype, Ptype, RunType>::Global().get_net(std::this_thread::get_id());
            //fill the graph inputs
            for(int i = 0; i < _inputs_in_order.size(); i++) {
                auto d_tensor_in_p = net.get_in(_inputs_in_order[i]);
                d_tensor_in_p->reshape(ins[i]->valid_shape());
                d_tensor_in_p->copy_from(*ins[i]);
                d_tensor_in_p->set_seq_offset(ins[i]->get_seq_offset());
            }

            net.prediction();

            // get outputs of graph
            std::vector<Tensor4dPtr<Ttype, Dtype>> ret;
            for(auto out : _outputs_in_order) {
                auto d_tensor_out_p = net.get_out(out);
                ret.push_back(d_tensor_out_p);
            }

            return ret;
        }; 
    _async_que.push(this->RunAsync(task, net_ins_list)); 
} 

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
std::vector<Tensor4dPtr<Ttype, Dtype> > Worker<Ttype, Dtype, Ptype, RunType>::async_get_result() {
    std::lock_guard<std::mutex> guard(this->_async_que_mut);    
    auto result = std::move(_async_que.front());
    _async_que.pop();
    return result.get();
} 

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Dtype, Ptype, RunType>::init() {
    MultiThreadModel<Ttype, Dtype, Ptype, RunType>::Global().initial(_model_path, _in_shapes);
}

template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Dtype, Ptype, RunType>::auxiliary_funcs() {
    for(auto func : _auxiliary_funcs) {
        func();
    }
}

#ifdef USE_CUDA
template class Worker<NV, AK_FLOAT, Precision::FP32, OpRunType::ASYNC>;
template class Worker<NV, AK_FLOAT, Precision::FP16, OpRunType::ASYNC>;
template class Worker<NV, AK_FLOAT, Precision::INT8, OpRunType::ASYNC>;

template class Worker<NV, AK_FLOAT, Precision::FP32, OpRunType::SYNC>;
template class Worker<NV, AK_FLOAT, Precision::FP16, OpRunType::SYNC>;
template class Worker<NV, AK_FLOAT, Precision::INT8, OpRunType::SYNC>;
#endif

#ifdef USE_X86_PLACE
template class Worker<X86, AK_FLOAT, Precision::FP32, OpRunType::ASYNC>;
template class Worker<X86, AK_FLOAT, Precision::FP16, OpRunType::ASYNC>;
template class Worker<X86, AK_FLOAT, Precision::INT8, OpRunType::ASYNC>;

template class Worker<X86, AK_FLOAT, Precision::FP32, OpRunType::SYNC>;
template class Worker<X86, AK_FLOAT, Precision::FP16, OpRunType::SYNC>;
template class Worker<X86, AK_FLOAT, Precision::INT8, OpRunType::SYNC>;
#endif

#ifdef USE_ARM_PLACE
template class Worker<ARM, AK_FLOAT, Precision::FP32, OpRunType::ASYNC>;
template class Worker<ARM, AK_FLOAT, Precision::FP16, OpRunType::ASYNC>;
template class Worker<ARM, AK_FLOAT, Precision::INT8, OpRunType::ASYNC>;

template class Worker<ARM, AK_FLOAT, Precision::FP32, OpRunType::SYNC>;
template class Worker<ARM, AK_FLOAT, Precision::FP16, OpRunType::SYNC>;
template class Worker<ARM, AK_FLOAT, Precision::INT8, OpRunType::SYNC>;
#endif

#ifdef USE_AMD
template class Worker<AMD, AK_FLOAT, Precision::FP32, OpRunType::ASYNC>;
template class Worker<AMD, AK_FLOAT, Precision::FP16, OpRunType::ASYNC>;
template class Worker<AMD, AK_FLOAT, Precision::INT8, OpRunType::ASYNC>;

template class Worker<AMD, AK_FLOAT, Precision::FP32, OpRunType::SYNC>;
template class Worker<AMD, AK_FLOAT, Precision::FP16, OpRunType::SYNC>;
template class Worker<AMD, AK_FLOAT, Precision::INT8, OpRunType::SYNC>;
#endif
} /* namespace */

