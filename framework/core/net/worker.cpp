#include "framework/core/net/worker.h"

#ifndef USE_SGX
#include "saber/funcs/timer.h"

namespace anakin {

//! \brief a model map between thread_id and net model
template<typename Ttype, Precision Ptype, OpRunType RunType>
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

    inline Net<Ttype, Ptype, RunType>& get_net(key id) {
        if(_thread_to_net.find(id) != _thread_to_net.end()) { 
            return _thread_to_net[id];
        }
        LOG(FATAL) << " target key(thread_id) not found in NetGraphWrapper";
        return _thread_to_net[id];
    }
    
private:
    std::unordered_map<std::string, graph::Graph<Ttype, Ptype>> _graph_map;
    std::unordered_map<key, Net<Ttype, Ptype, RunType>> _thread_to_net GUARDED_BY(this->_mut);
    std::mutex _mut;
};

template<typename Ttype, Precision Ptype, OpRunType RunType>
using MultiThreadModel = Singleton<NetGraphWrapper<Ttype, Ptype, RunType>>;

template<typename Ttype, Precision Ptype, OpRunType RunType>
Worker<Ttype, Ptype, RunType>::Worker(std::string model_path, int num_thread) : _model_path(model_path), ThreadPool(num_thread) {}

template<typename Ttype, Precision Ptype, OpRunType RunType>
Worker<Ttype, Ptype, RunType>::~Worker() {}

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Ptype, RunType>::pause(size_t time) {
    std::function<void(int)> sleep = [](size_t time) {
        std::this_thread::sleep_for(std::chrono::milliseconds(time));
    };
    this->RunSync(sleep, time);
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Ptype, RunType>::Reshape(std::string in_name, std::vector<int> new_shape) {
    _in_shapes[in_name] = new_shape;
}


template<typename Ttype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Ptype, RunType>::register_inputs(std::vector<std::string> input_names) {
    _inputs_in_order = input_names;
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Ptype, RunType>::register_outputs(std::vector<std::string> output_names) {
    _outputs_in_order = output_names;
}

template<typename Ttype, Precision Ptype, OpRunType RunType> 
void Worker<Ttype, Ptype, RunType>::register_interior_edges(std::string bottom, std::string top) {
    graph::Arc<std::string, int> arc(bottom, top);
    _edges_in_order.push_back(arc);
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
std::future<std::vector<Tensor4d<typename target_host<Ttype>::type> > > 
Worker<Ttype, Ptype, RunType>::sync_prediction(std::vector<Tensor4d<typename target_host<Ttype>::type> >& net_ins_list) {
    auto task = [&](std::vector<Tensor4d<typename target_host<Ttype>::type> >& ins) 
                                -> std::vector<Tensor4d<typename target_host<Ttype>::type> > {
        auto& net = MultiThreadModel<Ttype, Ptype, RunType>::Global().get_net(std::this_thread::get_id()); 
        //fill the graph inputs

        for(int i = 0; i < _inputs_in_order.size(); i++) { 
            float* data = (float*)(ins[i].mutable_data());
            for(int j=0; j<10; j++) {
                LOG(INFO) << "------> data " << data[j];;
            }
            auto d_tensor_in_p = net.get_in(_inputs_in_order[i]);
            d_tensor_in_p->reshape(ins[i].valid_shape());
            d_tensor_in_p->copy_from(ins[i]);
            d_tensor_in_p->set_seq_offset(ins[i].get_seq_offset());
        }

//        Context<NV> ctx(0, 0, 0);
//        saber::SaberTimer<NV> my_time;
//        my_time.start(ctx);
#ifdef ENABLE_OP_TIMER
        Context<Ttype> ctx(0, 0, 0);
        saber::SaberTimer<Ttype> my_time;
        my_time.start(ctx);
#endif
        net.prediction();
//
//        my_time.end(ctx);
//        LOG(ERROR) << " exec  << time: " << my_time.get_average_ms() << " ms ";

#ifdef ENABLE_OP_TIMER
        my_time.end(ctx); 
        {
            std::lock_guard<std::mutex> guard(_mut); 
            _thead_id_to_prediction_times_vec_in_ms[std::this_thread::get_id()].push_back(my_time.get_average_ms());
            LOG(ERROR) << " exec  << time: " << my_time.get_average_ms() << " ms ";
        }
#endif
        // get outputs of graph
        std::vector<Tensor4d<typename target_host<Ttype>::type>> ret;
        ret.resize(_outputs_in_order.size());
        for (int out_idx = 0; out_idx <  _outputs_in_order.size(); out_idx++) {
            auto d_tensor_out_p = net.get_out(_outputs_in_order[out_idx]);
            ret[out_idx].re_alloc(d_tensor_out_p->valid_shape());
            ret[out_idx].copy_from(*d_tensor_out_p);
            float* data = (float*)(ret[out_idx].mutable_data());
            LOG(INFO) << "this thread: " << std::this_thread::get_id();
            for(int i=0; i< 10; i++) {
                LOG(INFO) << "????? data " << data[i];
            }
        }

        return ret; 
    };
    return this->RunAsync(task, net_ins_list);
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
std::future<std::vector<Tensor4dPtr<Ttype> > > Worker<Ttype, Ptype, RunType>::sync_prediction_device(std::vector<Tensor4dPtr<Ttype> >& net_ins_list) {
    auto task = [&](std::vector<Tensor4dPtr<Ttype> >& ins) -> std::vector<Tensor4dPtr<Ttype> > {
        auto& net = MultiThreadModel<Ttype, Ptype, RunType>::Global().get_net(std::this_thread::get_id()); 
        //fill the graph inputs 
        for (int i = 0; i < _inputs_in_order.size(); i++) { 
            auto d_tensor_in_p = net.get_in(_inputs_in_order[i]); 
            d_tensor_in_p->copy_from(*ins[i]); 
        } 
        net.prediction(); 
        // get outputs of graph
        std::vector<Tensor4dPtr<Ttype>> ret;
        for (auto out : _outputs_in_order) {
            auto d_tensor_out_p = net.get_out(out);
            ret.push_back(d_tensor_out_p);
        }

        return ret; 
    }; 
    return this->RunAsync(task, net_ins_list);
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Ptype, RunType>::async_prediction(std::vector<Tensor4dPtr<typename target_host<Ttype>::type> >& net_ins_list) {
    std::lock_guard<std::mutex> guard(this->_async_que_mut);    
    auto task = [&](std::vector<Tensor4dPtr<typename target_host<Ttype>::type> >& ins) -> std::vector<Tensor4dPtr<Ttype> > {
            auto& net = MultiThreadModel<Ttype, Ptype, RunType>::Global().get_net(std::this_thread::get_id());
            //fill the graph inputs
            for(int i = 0; i < _inputs_in_order.size(); i++) {
                auto d_tensor_in_p = net.get_in(_inputs_in_order[i]);
                d_tensor_in_p->reshape(ins[i]->valid_shape());
                d_tensor_in_p->copy_from(*ins[i]);
                d_tensor_in_p->set_seq_offset(ins[i]->get_seq_offset());
            }

            net.prediction();

            // get outputs of graph
            std::vector<Tensor4dPtr<Ttype>> ret;
            for(auto out : _outputs_in_order) {
                auto d_tensor_out_p = net.get_out(out);
                ret.push_back(d_tensor_out_p);
            }

            return ret;
        }; 
    _async_que.push(this->RunAsync(task, net_ins_list)); 
} 

template<typename Ttype, Precision Ptype, OpRunType RunType>
std::vector<Tensor4dPtr<Ttype> > Worker<Ttype, Ptype, RunType>::async_get_result() {
    std::lock_guard<std::mutex> guard(this->_async_que_mut);    
    auto result = std::move(_async_que.front());
    _async_que.pop();
    return result.get();
} 

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Ptype, RunType>::init() {
    MultiThreadModel<Ttype, Ptype, RunType>::Global().initial(_model_path, _in_shapes);
}

template<typename Ttype, Precision Ptype, OpRunType RunType>
void Worker<Ttype, Ptype, RunType>::auxiliary_funcs() {
    for(auto func : _auxiliary_funcs) {
        func();
    }
}

#ifdef USE_CUDA
template class Worker<NV, Precision::FP32, OpRunType::ASYNC>;
template class Worker<NV, Precision::FP16, OpRunType::ASYNC>;
template class Worker<NV, Precision::INT8, OpRunType::ASYNC>;

template class Worker<NV, Precision::FP32, OpRunType::SYNC>;
template class Worker<NV, Precision::FP16, OpRunType::SYNC>;
template class Worker<NV, Precision::INT8, OpRunType::SYNC>;
#endif

#ifdef AMD_GPU
template class Worker<AMD, Precision::FP32, OpRunType::ASYNC>;
template class Worker<AMD, Precision::FP16, OpRunType::ASYNC>;
template class Worker<AMD, Precision::INT8, OpRunType::ASYNC>;

template class Worker<AMD, Precision::FP32, OpRunType::SYNC>;
template class Worker<AMD, Precision::FP16, OpRunType::SYNC>;
template class Worker<AMD, Precision::INT8, OpRunType::SYNC>;
#endif

#ifdef USE_X86_PLACE
template class Worker<X86, Precision::FP32, OpRunType::ASYNC>;
template class Worker<X86, Precision::FP16, OpRunType::ASYNC>;
template class Worker<X86, Precision::INT8, OpRunType::ASYNC>;

template class Worker<X86, Precision::FP32, OpRunType::SYNC>;
template class Worker<X86, Precision::FP16, OpRunType::SYNC>;
template class Worker<X86, Precision::INT8, OpRunType::SYNC>;
#endif

#ifdef USE_ARM_PLACE

#ifdef ANAKIN_TYPE_FP32
template class Worker<ARM, Precision::FP32, OpRunType::ASYNC>;
template class Worker<ARM, Precision::FP32, OpRunType::SYNC>;
#endif

#ifdef ANAKIN_TYPE_FP16
template class Worker<ARM, Precision::FP16, OpRunType::ASYNC>;
template class Worker<ARM, Precision::FP16, OpRunType::SYNC>;
#endif

#ifdef ANAKIN_TYPE_INT8
template class Worker<ARM, Precision::INT8, OpRunType::ASYNC>;
template class Worker<ARM, Precision::INT8, OpRunType::SYNC>;
#endif

#endif

} /* namespace */

#endif
