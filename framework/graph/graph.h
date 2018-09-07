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

#ifndef ANAKIN_GRAPH_H
#define ANAKIN_GRAPH_H 

#include "framework/graph/graph_base.h"
#include "framework/graph/node.h"
#include "framework/graph/algorithm.h"
#include "framework/graph/llvm/virtual_graph.h"
#include "framework/core/thread_safe_macros.h"
#include "framework/graph/graph_global_mem.h"

namespace anakin {

namespace graph {

/**
 * \brief Graph class
 * public inherit GraphBase
*/
template<typename Ttype, Precision Ptype>
class Graph : public GraphBase<std::string, 
                               NodePtr, 
                               Tensor4dPtr<Ttype>, 
                               Edge<Ttype> > {
public:
    Graph():GraphBase<std::string, 
                      NodePtr, 
                      Tensor4dPtr<Ttype>, 
                      Edge<Ttype> >() {}
    Graph(size_t size):GraphBase<std::string, 
                                 NodePtr, 
                                 Tensor4dPtr<Ttype>, 
                                 Edge<Ttype> >(size) {}

    ~Graph() {
        if(_vgraph) { 
            delete _vgraph;
            _vgraph = nullptr;
        }
    }

    /// get graph name
    void set_name(std::string name){_name = name;}
    std::string& name() { return _name; }

    /// add i/o
    void add_in(std::string in) { _ins.push_back(in); }
    void add_out(std::string out) { _outs.push_back(out); }
    /// graph io node name
    std::vector<std::string>& get_ins() { return _ins; }
    std::vector<std::string>& get_outs() { return _outs; }

    /// Judge if graph is directed graph, must be override.
    virtual bool directed() final { return true; }

    /// Parsing from model
    Status load(std::string model_path); 
    Status load(const char*  model_path);
    Status save(std::string model_path);
    Status save(const char*  model_path);
    /// Get nodes in execution oroder.
    std::vector<std::string>& get_nodes_in_order();

    /// reshape input by shape
    void Reshape(std::string in_name, std::vector<int> shape);

    void ResetBatchSize(std::string in_name, const int batch_size);

public:
    /** 
     * \brief register out
     *
     * Note: 
     *   The outs is the same as edge weight from  node_bottom_name to node_top_name
     *   When register the out edge, all the fusion pattern that have the edge can't be combined
     *   and maybe have an bad impact on the perfermance
     */
    Status RegistOut(std::string node_bottom_name, std::string node_top_name);
    
    /** 
     * \brief register all outs
     *
     * Note: 
     *   All the outs will be registered.
     *   This api should be used when you test you model and want to test some edge's tensor inside the graph.
     */
    Status RegistAllOut();


    /// optimization for graph
    Status Optimize();
    /// Get virtual graph.
    VGraph& get_vgraph();
    /// Restore real Graph from optimized virtual graph.
    Status restore_from_vgraph(VGraph*);

    /**
     * \biref shallow copy of graph
     * note: only copy parameters and architecture, but not the weights
    */
    Status CopyFrom(Graph<Ttype, Ptype>& graph);

    ///< statistics stand for Statistics info of anakin graph
    Statistics statistics;

private:
    /**
     * \brief clean all the resources(include the graph parameter and weights) used by this graph
     */
    Status Clean();

private:
    ///< _vgraph stand for graph. default nullptr
    VGraph* _vgraph{nullptr};
    ///< _name stand for message
    std::string _name{"default"};
    ///< graph input node name
    std::vector<std::string> _ins; 
    ///< graph output node name     
    std::vector<std::string> _outs;   
    ///< graph node execute list
    std::vector<std::string> _nodes_exec_order;
    ///< node_merges map: target node map to all its fusion node
    std::unordered_map<std::string, std::vector<std::string> > _node_merges;
	///< _node_merges_keep map: target node map to all its fusion node that shouldn't be removed
	std::unordered_map<std::string, std::vector<int> > _node_merges_keep;

    ///< _pattern_name_merges map: target node map to all its fusion pattern node
    std::unordered_map<std::string, std::vector<std::string> > _pattern_name_merges;

    ///< _registed_outs:outs that needs to be exported
    std::vector<std::pair<std::string, std::string>> _registed_outs;


private:
    /// this used to holder the name of target parsed model.
    std::string _model_path{"None"} GUARDED_BY(this->_mut);
    /// this make the graph optimized.
    bool _has_graph_optimized{false}; GUARDED_BY(this->_mut);
    std::mutex _mut;
}; 



} /* graph */

} /* anakin */

#endif /* ANAKIN_GRAPH_H */
