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

#ifndef ANAKIN_WORKER_H
#define ANAKIN_WORKER_H

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <future>
#include <mutex>
#include <condition_variable>
#include "framework/core/thread_safe_macros.h"
#include "framework/core/thread_pool.h"
#include "framework/core/singleton.h"
#include "framework/core/net/operator_func.h"
#include "framework/core/net/net.h"

namespace anakin {

/** 
 *  \brief class Worker for multi-thread anakin inference.
 *  \par Usage: 
 *      - \p [SYNC]
 *          \code
 *          std::string vgg = "/path/to/vgg_net.anakin.bin";
 *          Worker<GPU, float, OpRunType::ASYNC>  worker_for_vgg_net(vgg, 10); // define a worker with 10 threads for 
 *
 *          for(i=0; i< data.lines.size(); i++) {
 *              std::vector<Tensor4dPtr<CPU, float> >& net_ins;
 *              fill(net_ins, data);    // fill the net inputs with data
 *              auto outs = worker_for_vgg_net.sync_prediction(net_ins);
 *          }
 *          \endcode
 *      - \p [ASYNC]
 *          \code
 *          std::string vgg = "/path/to/vgg_net.anakin.bin";
 *          Worker<GPU, float, OpRunType::ASYNC>  worker_for_vgg_net(vgg, 10); // define a worker with 10 threads for
 *          for(i=0; i< data.lines.size(); i++) {
 *              std::vector<Tensor4dPtr<CPU, float> >& net_ins;
 *              fill(net_ins, data);    // fill the net inputs with data
 *              worker_for_vgg_net.async_prediction(net_ins);
 *          }
 *
 *          // in other place
 *          for(i=0; i< data.lines.size(); i++) {
 *              auto outs = worker_for_vgg_net.async_get_result();         
 *          }
 *          \endcode
 *
 */
template<typename Ttype, Precision Ptype, OpRunType RunTyp = OpRunType::ASYNC>
class Worker : public ThreadPool {
public:
    Worker(std::string model_path, int thread_num);
    ~Worker();

	/** 
	 * 	\brief Set the vector of the Output name strings in order. 
	 *  When you set the vector, you will get the target tensor by name that you set.
     *  \param outs_in_order the graph's Output names.
     *  \return void.
     */
	void RegisterOuts(std::vector<std::string> outs_in_order);

public:
    /** 
     *  \brief reshape the input by shape
     */
    void Reshape(std::string, std::vector<int>);

    /** 
     *  \brief register input node names in order
     */
    void register_inputs(std::vector<std::string>); 

    /** 
     *  \brief register output node names in order
     */
    void register_outputs(std::vector<std::string>);

    /** 
     *  \brief register interior edges in order. 
     *  interior edge should be constructed by {first, second}
     *  register_interior_edges can be invoked muti times and the edge's tensor will output in same order
     *
     */
    void register_interior_edges(std::string, std::string);

public:
    /** 
     *  \brief do sync prediction in multi-thread worker useful in sync rpc server. 
     *  \param host net_in_list the inputs of net graph (note: the len of net_in_list should be equal to the net inputs).  
     *  \return the net graph outputs.
     */
    std::future<std::vector<Tensor4d<typename target_host<Ttype>::type> > > sync_prediction(\
        std::vector<Tensor4d<typename target_host<Ttype>::type> >& net_in_list);

    /** 
     *  \brief Do sync prediction in multi-thread worker useful in sync rpc server, this function need 
     *  \param device net_in_list the inputs of net graph (note: the len of net_in_list should be equal to the net inputs).  
     *  \return the net graph outputs.
     */
    std::future<std::vector<Tensor4dPtr<Ttype> > > sync_prediction_device(\
        std::vector<Tensor4dPtr<Ttype> >& net_in_list);

    /** 
     *  \brief do async prediction in multi-thread worker, the result will be save to que 
     *  \param net_in_list the inputs of net graph (note: the len of net_in_list should be equal to the net inputs)  
     *  \return void
     */
    void async_prediction(std::vector<Tensor4dPtr<typename target_host<Ttype>::type> >& net_in_list);
    
    /** 
     *  \brief Judge if the async queue is empty.
     *  \return bool return true if it's empty otherwise false.
     */
    bool empty() { return _async_que.empty(); }

    /** 
     *  \brief async get result of multi-thread worker. 
     *  the return order of results from async_get_result is the same as the order of net_in_list called by async_prediction.
     *  \return the net inference result.
     */
    std::vector<Tensor4dPtr<Ttype> > async_get_result();

public:
    /** 
     *  \biref register auxiliary functions will be lanunched each time when the sync/async is called
     */
    template<typename functor, typename ...ParamTypes>
    void register_aux_function(functor function, ParamTypes ...args);

public:
    /** 
     *  \brief Threads in worker will sleep time(ms) long.
     *  \param time (ms).
     */
    void pause(size_t time);

#ifdef ENABLE_OP_TIMER
    /**
     *  \brief get sync prediction times map
     */
    std::unordered_map<std::thread::id, std::vector<float>>& get_task_exe_times_map_of_sync_api() { return _thead_id_to_prediction_times_vec_in_ms; }
#endif

private:

    /** 
     *  \brief Initial the net resource.
     */
    virtual void init() override;

    virtual void auxiliary_funcs() override;

private:
    std::string _model_path;
    ///< vector of inputs node in order.
    std::vector<std::string> _inputs_in_order;
    ///< vector of outputs node in order.
    std::vector<std::string> _outputs_in_order;
    ///< vector of edges in order.
    std::vector<graph::Arc<std::string, int>> _edges_in_order;
    std::queue< std::future< std::vector<Tensor4dPtr<Ttype> > > > _async_que GUARDED_BY(_async_que_mut);
    std::mutex _async_que_mut;    
    std::vector<std::function<void(void)> > _auxiliary_funcs;
    std::unordered_map<std::string, std::vector<int>> _in_shapes;
#ifdef ENABLE_OP_TIMER
    std::unordered_map<std::thread::id, std::vector<float>> _thead_id_to_prediction_times_vec_in_ms;
    std::mutex _mut;
#endif
};

template<typename Ttype, Precision Ptype, OpRunType RunType>
template<typename functor, typename ...ParamTypes>
void Worker<Ttype, Ptype, RunType>::register_aux_function(functor function, ParamTypes ...args) {
    auto task = std::bind(function, std::forward<ParamTypes>(args)...);
    _auxiliary_funcs.push_back(task);
}


///< global singleton worker
template<typename Ttype, Precision Ptype, OpRunType RunType>
using GlobalWorker = Singleton<Worker<Ttype, Ptype, RunType>>;

} /* namespace */

#endif
