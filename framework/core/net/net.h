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

#ifndef ANAKIN_NET_H
#define ANAKIN_NET_H

#include "framework/graph/graph.h"
#include "framework/core/net/operator_func.h"


namespace anakin {

/** 
 *  \brief Net class used for execution of graph and it is thread safety.
 */
template<typename Ttype, DataType Dtype, Precision Ptype, OpRunType RunTyp = OpRunType::ASYNC>
class Net {
public:
    explicit Net(bool need_summary = false);

    /** 
     *  \brief Construct a net by graph. 
     *  This construction should be use in thread call and make sure thread safety.
     */
    explicit Net(graph::Graph<Ttype, Dtype, Ptype>&, bool need_summary = false);

    /**
     *  \brief Construct a net by graph, init with specified context.
     *  This construction should be use in thread call and make sure thread safety.
     */
    explicit Net(graph::Graph<Ttype, Dtype, Ptype>&, OpContextPtr<Ttype> ctx, bool need_summary = false);

    ~Net();

public:
    /**
     * \brief init execute net from graph, init with specified context.
     *  you can use Net(Graph&) instead.
     */
    void init(graph::Graph<Ttype, Dtype, Ptype>& graph, OpContextPtr<Ttype> ctx);

    /**
     * \brief init execute net from graph.
     *  you can use Net(Graph&) instead.
     */
    void init(graph::Graph<Ttype, Dtype, Ptype>&);
    
    /** 
     * \brief do inference.   
     */
    void prediction();

	/**
	 *  \brief Running model from inputs to target edge
	 *
	 *   We support some api for partly running mode.
	 *   For example, you can execute part of the model by using api
	 *   execute_stop_at_edge(node name), then anakin will run the model 
	 *   in order from input to the node(its computation is not invoked) 
	 *   and other computation is suspended. Beside, anakin supply an api 
	 *   running from target node throughtout end of model.
	 *   NOTE: 
	 *   	Those api should be carefully used, if you want to get edge 
	 *   	tensors after target node you stop at, you need to register 
	 *   	the edges at graph optimizing stage at first.
	 */
	void execute_stop_at_node(std::string node_name);

	/**
	 *  \brief running from edge to end
	 */
	void execute_start_from_node(std::string node_name);

    //! get time for each op;
#ifdef ENABLE_OP_TIMER
    void reset_op_time() {_op_time = std::vector<float>(_exec_funcs.size(), 0.0f);}
    std::vector<float> get_op_time() {return _op_time;}
    std::vector<std::string> get_op_param() {return _op_param;}
    std::vector<OperatorFunc<Ttype, Dtype, Ptype> > get_exec_funcs() {
        return _exec_funcs;
    }
#endif

public:

    /**
     *  \brief Get out by name.
     */
    Tensor4dPtr<Ttype, Dtype> get_out(std::string out_name);
    std::vector<Tensor4dPtr<Ttype, Dtype> > get_out_list();
    
    /**
     *  \brief Get in by name.
     */
    Tensor4dPtr<Ttype, Dtype> get_in(std::string in_name);
    std::vector<Tensor4dPtr<Ttype, Dtype> > get_in_list();

    /**
     *  \brief Get tensor from a given edge.
     */
    Tensor4dPtr<Ttype, Dtype> get_tensor_from_edge(const char* from, const char* to);

private:
    /**
     *  \brief Allocate memory for net.
     */
    Status init_memory();

    /**
     *  \brief Initial context environments.
     */
    Status init_env(graph::Graph<Ttype, Dtype, Ptype>&);

private:
    ///< executor for operators in node.
    std::vector<OperatorFunc<Ttype, Dtype, Ptype> > _exec_funcs;
	///< suspended point is set when you invoke execute_stop_at_node
	int _suspended_point{-1};
	///< start point is set when you invoke execute_start_from_node
	int _start_point{-1};
    ///< The pointer to Context.
    OpContextPtr<Ttype> _ctx_p;
    graph::Graph<Ttype, Dtype, Ptype>* _graph_p{nullptr};
    ///< A list of in tensor.
    std::vector<Tensor4dPtr<Ttype, Dtype> > _in_tensor_list;
    ///< A list of out tensor.
    std::vector<Tensor4dPtr<Ttype, Dtype> > _out_tensor_list;

    bool _need_summary{false};
#ifdef ENABLE_OP_TIMER
    std::vector<float> _op_time;
    std::vector<std::string> _op_param;
#endif
};

} /* namespace */

#endif
