/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

#ifndef ANAKIN_FRAMEWORK_LITE_CODE_GEN_BASE_H
#define ANAKIN_FRAMEWORK_LITE_CODE_GEN_BASE_H

#include <string>
#include <vector>
#include <unordered_map>

#include "framework/graph/graph.h"

namespace anakin {

namespace lite {

/**
 * \brief Node information for generating executor
 */
struct NodeInfo {
    std::string name;				// node name
    std::string op_name;			// op name
    std::vector<std::string> ins;	// input edge name
    std::vector<std::string> outs;	// output edge name
    DataType dtype;
};


/**
 * \brief Edge information for generating edge tensors.
 */
struct EdgeInfo {
    std::string name;	 			// edge name
    std::vector<int> valid_shape; 	// edge valid shape
    std::vector<int> real_shape;	// edge real shape
    bool is_shared{false}; 			// if the edge is shared by others
    std::string share_from{""}; 	// if the edge is_shared(true), share_from will hold the target edge name.
    std::string in_node;
    std::string out_node;
    std::vector<float> scale;
    DataType dtype;
};

/**
 *  \brief class for target language code generator.
 *
 *  The class CodeGenBase hold base information for running model.
 *  There exists several base info:
 *  	1. Operatoin name in execution order.
 *  	2. All the tensor model needs and share info between those tensors.
 *  	3. Model weights
 */
template<typename Ttype, Precision Ptype>
class CodeGenBase {
public:
	CodeGenBase() {}
	virtual ~CodeGenBase(){}

	/**
	 *  \biref extract graph msg
	 */
	bool extract_graph(const std::string& model_path, const int batch_size = 1);

	/**
	 * \brief generate all source files
	 */
	virtual void gen_files(const bool debug_mode) = 0;


private:
	/**
	 * \brief analyse the memory reuse info
	 */
	bool init_memory_info();


    /**
     * \brief change graph edge and node name to match the standard of c variable name
     */
    void change_name(graph::Graph<Ttype, Ptype>&);

	/**
	 * \brief generate ops of graph
	 */
	virtual void gen_ops() = 0;

protected:
	graph::Graph<Ttype, Ptype> _graph;
	std::vector<std::string> _exec_node_order; /// running order of operation's name
	std::vector<std::string> _ins;	/// graph ins
	std::vector<std::string> _outs; /// graph outs
	std::unordered_map<std::string, NodeInfo> _graph_node_map;
	/// graph base arch
	std::unordered_map<std::string, EdgeInfo> _tensor_map;
};

} /* namespace lite */

} /* namespace anakin */

#endif

