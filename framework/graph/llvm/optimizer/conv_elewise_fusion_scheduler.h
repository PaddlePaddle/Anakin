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

#ifndef ANAKIN_LLVM_SCHEDULER_CONV_ELEWISE_FUSION_H
#define ANAKIN_LLVM_SCHEDULER_CONV_ELEWISE_FUSION_H

#include <unordered_map>

#include "utils/logger/logger.h"
#include "framework/graph/llvm/schedule_base.h"
#include "framework/graph/llvm/virtual_graph.h"
#include "framework/graph/llvm/scheduler.h"

namespace anakin {

namespace graph {

/**
 *  \brief ConvElsFusionScheduler helper class
 */
struct ConvElsFusionHelper {
private:
	std::vector<std::string> ops {
		"ConvBatchnormScale",
        "Convolution"
	};
	struct conv_eltwise_pair {
		std::string conv_name;
		std::string eltwise_name;
		inline bool operator==(const conv_eltwise_pair& pair_other) {
			return (conv_name == pair_other.conv_name) && (eltwise_name == pair_other.eltwise_name);
		}
	};
	std::vector<conv_eltwise_pair> _pairs;

	std::vector<std::string> _node_need_to_wait;

public:
	/**
	 * \brief judge if meet target op
	 */
	inline bool has_node(node& node_arg) {
		for(auto& op : ops) {
			if(op == node_arg.opName) {
				return true;
			}
		}
		return false;
	}

	bool need_wait(std::string& node_name) {
		auto ret = std::find(_node_need_to_wait.begin(), _node_need_to_wait.end(), node_name);
		if(ret != _node_need_to_wait.end()) {
			return true;
		}
		return false;
	}

	void push_wait(std::string& node_name) {
		if(!need_wait(node_name)) {
			_node_need_to_wait.push_back(node_name);
		}
	}

	void release(std::string& node_name) {
		int index = -1;
		for(int i=0; i<_node_need_to_wait.size();i++) {
			if(_node_need_to_wait[i] == node_name) {
				index = i;
			}
		}
		if(index != -1) {
			_node_need_to_wait.erase(_node_need_to_wait.begin()+index);
		}
	}

	/*void set_holder(std::vector<io>& io_vec, VGraph* graph) {
		for(auto io : io_vec) {
			io.holder = true;
		}
		for (auto& io_res : io_vec) { 
			auto replace_arc = [&](Arc<std::string, io>& arc) { 
				if (arc.weight() == io_res) { 
					auto& io_tmp = arc.weight(); 
					io_tmp = io_res; 
					return Status::EXIT(" Find the matched target arc io. "); 
				} 
				return Status::OK(); 
			}; 
			graph->Scanner->BFS_Edge(replace_arc); 
		}
	}*/

	void register_pair(std::string& conv_name, std::string& eltwise_name) {
		conv_eltwise_pair tmp_pair;
		tmp_pair.conv_name = conv_name;
		tmp_pair.eltwise_name = eltwise_name;
		auto ret = std::find(_pairs.begin(), _pairs.end(), tmp_pair);
		if(ret == _pairs.end()) {
			_pairs.push_back(tmp_pair);
		}
	}

	std::vector<conv_eltwise_pair>& get_replace_pairs() {
		return _pairs;
	}
};

/**
 *  \brief Dependency scheduler for analysing the possibility of conv+eltwise fusion in graph
 */
class ConvElsFusionScheduler : public Scheduler {
public:
    ConvElsFusionScheduler() {}
    virtual ~ConvElsFusionScheduler() {}

	/// decide if the target node's op is callable
    virtual bool callable(node&);

	/// run scheduler
    virtual void Run();

    virtual std::vector<std::string> get_exec_node_in_order();


private:
	ConvElsFusionHelper _helper;
	std::unordered_map<std::string, node> _force_order;
};


} /* namespace graph */

} /* namespace anakin */

#endif 
