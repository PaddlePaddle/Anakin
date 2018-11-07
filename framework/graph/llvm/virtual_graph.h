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

#ifndef ANAKIN_LLVM_VIRTUAL_GRAPH_H
#define ANAKIN_LLVM_VIRTUAL_GRAPH_H

#include "framework/core/parameter.h"
#include "framework/graph/llvm/base.h"
#include "utils/logger/logger.h"
#include "framework/graph/graph_base.h"

namespace anakin {

namespace graph {

struct io {
    io():share_from(""){}   

    /// io name
    std::string name;

    /// whether io is shared with others.
    /// if shared == true, io 's real memory should refer to other io's memory.
    bool shared{false};

    /// if share_from == iox.name,  then this io should share the memory with iox.
    std::string share_from;

    /// the stream lane the io belongs to. default 0
    int lane{0};

    /** 
    *  \brief get the size of shape
    *  \return size_t the value of shape.size()
    */
    inline size_t size() { return shape.size(); }
    /** 
    *  \brief get the message of whether io is shared
    *  \return std::string the value of message
    */
    std::string ToString();

    /// receive from Graph.
    PTuple<int> shape;

    inline bool operator==(const io& rhs) const { return (name == rhs.name); }
    inline bool operator!=(const io& rhs) const { return (name != rhs.name); }
};

/**
 * \brief nodes of VGraph
 *
 *  note: the nodes may consists of single node or multi-nodes.
 *        you can treat the nodes as an sub-graph.      
 */
struct node {
   ///< name stand for node name same as Graph.
    std::string name;

    ///< opName stand for operator name
    std::string opName;

    ///< functorName stand for operator type of abstract backend.
    OpType functorName;

    ///< property stand for operator property
    OpProperty property;

    ///< mergeNodes stand for sub merged nodes
    std::vector<node> mergeNodes;
	///< save node's index in mergeNodes which shouldn't be removed in reconstructing Graph
	std::vector<int> idx_keep_in_merge_nodes;

    ///<mergeNodeNames stand for sub merged node names from pattern
    std::vector<std::string> mergeNodeNames;

    ///< lane stand for the stream of lane the node operator occurs. default 0
    int lane{0};
    ///<need_wait stand forwhether it needs wait .default false
    bool need_wait{false};
    
    std::string ToString();

    inline bool operator==(const node& rhs) { return name == rhs.name; }
    inline bool operator!=(const node& rhs) { return (*this) == (rhs); }

    node() {}
    ~node() {}

    /// copy and assign
    node(const node& rhs) {
        name = rhs.name;
        opName = rhs.opName;
        //functorName = rhs.functorName;
        //property = rhs.property;
        mergeNodeNames.clear();
        for (auto& node_name : rhs.mergeNodeNames) {
            mergeNodeNames.push_back(node_name);
        }
        mergeNodes.clear();
        for (auto& node_tmp : rhs.mergeNodes) {
            mergeNodes.push_back(node_tmp);
        }
    }

    inline node& operator=(const node& rhs) {
        name = rhs.name;
        opName = rhs.opName;
        //functorName = rhs.functorName;
        //property = rhs.property;
        mergeNodeNames.clear();
        for (auto& node_name : rhs.mergeNodeNames) {
            mergeNodeNames.push_back(node_name);
        }
        mergeNodes.clear();
        for (auto& node_tmp : rhs.mergeNodes) { 
            mergeNodes.push_back(node_tmp); 
        }
        return *this;
    }

    /// += operator for fusion
    inline node& operator+=(const node& rhs) {
        this->mergeNodes.push_back(rhs);
        return *this;
    }

	// register node index should keep
	inline void register_keep(int idx) {
		idx_keep_in_merge_nodes.push_back(idx);
	}
};

/**
 * \brief abstract virtual graph class
 */
class VGraph : public GraphBase<std::string, node, io> {
public:
    VGraph():GraphBase<std::string, node, io>(){}

    virtual bool directed() { return true; }

    /**
    * \brief Match graph
    * search the vgraph and find the matched vgraph_pattern, 
    * then replace it with fusion_op defined in class Pattern
    * \param vgraph_pattern matched graph
    */
    void Match(VGraph*);

    /// check if the arc is aceessable for fusion 
    bool check_pass(std::string, std::string);

    ///check if the the node is accessible to another
    bool check_accessible(std::string, std::string);

    ///make vgraph node index
    std::map<std::pair<std::string, std::string>, int> connect_table();

    /// register the arc outs 
    void register_outs(std::string, std::string);

    std::vector<std::pair<std::string, std::string>>& get_registed_outs() { return _registed_outs; }

	bool has_exec_order() { return _nodes_exec_order.size() == 0 ? false : true; }

	void set_exec_order(std::vector<std::string>& exe_order) { _nodes_exec_order = exe_order; }

	std::vector<std::string>& get_exec_order() { return _nodes_exec_order; }

private:
    ///< _registed_outs :outs that needs to be exported
    std::vector<std::pair<std::string, std::string>> _registed_outs;
	///< node execute order
	std::vector<std::string> _nodes_exec_order;
};


} /* namespace graph */

} /* namespace anakin */

#endif
