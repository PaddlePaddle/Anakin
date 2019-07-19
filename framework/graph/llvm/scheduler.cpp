/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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

#include "framework/graph/llvm/scheduler.h"

#include <algorithm>
#include <set>
#include <map>

namespace anakin {

namespace graph {

void Scheduler::RegIOResource(VGraph* vgraph) {
    auto register_io_f = [this](Arc<std::string, io>& arc) {
        auto& tmp_io = arc.weight();
        this->lock(tmp_io);
        this->RegResource(tmp_io);
        return 0;
    };
    // register io resources.
    vgraph->Scanner->BFS_Edge(register_io_f);

	/*if(vgraph->has_exec_order()) {
		auto node_exec_order = vgraph->get_exec_order();
		for(auto& node_name : node_exec_order) {
			this->wait_push((*vgraph)[node_name]);
		}
	} else {*/
    	auto push_wait_que_f = [this](node & node_arg) {
        	this->wait_push(node_arg);
        	return 0;
    	};
    	// push all node op to wait que and disable the out resources.
    	vgraph->Scanner->BFS(push_wait_que_f);
	//}

    // scheduler add fix arc io
    auto& regist_outs = vgraph->get_registed_outs();

    for (auto tmp_pair : regist_outs) {
        Arc<std::string, io> arc(tmp_pair.first, tmp_pair.second);
        auto& tmp_io = arc.weight();
        tmp_io.name = arc.name();
        _fix_io_res.push_back(tmp_io);
    }

    // holds the virtual graph
    _vgraph = vgraph;
}

bool Scheduler::callable(node& node_arg) {
    auto& node_arc_in_its = _vgraph->get_in_arc_its(node_arg.name);
    std::vector<io> io_in;

    for (auto& arc_it : node_arc_in_its) {
        io_in.push_back(arc_it->weight());
    }

    return this->check_access(io_in);
}

void Scheduler::launch(node& node_arg) {
    this->exe_push(node_arg);
    auto& node_arc_out_its = _vgraph->get_out_arc_its(node_arg.name);
    std::vector<io> io_out;

    for (auto& arc_it : node_arc_out_its) {
        io_out.push_back(arc_it->weight());
    }

    this->free(io_out);
}

void Scheduler::Run() {
    for (std::size_t wait_que_size = this->_wait_que.size()
            ; !(this->_wait_que.empty()); ) {
        // lanuch the acessible op and remove it from wait que.
        for (auto op_it = this->_wait_que.begin(); op_it != this->_wait_que.end();) {
            if (callable(*op_it)) {
                launch(*op_it);
                op_it = this->_wait_que.erase(op_it);
            } else {
                ++op_it;
            }
        }

        // if _wait_que.size not change, Scheduler::Run won't stop
        // check and fatal out
        if (wait_que_size == _wait_que.size()) {
            // check loop
            std::map<std::string, std::set<std::string>> op_inputs;
            std::map<std::string, std::set<std::string>> op_outputs;

            // get op io
            for (const auto& op : this->_wait_que) {
                for (auto& arc_in : _vgraph->get_in_arc_its(op.name)) {
                    io& in = arc_in->weight();
                    if (this->check_access(in)) {
                        continue;
                    }

                    op_inputs[op.name].insert(in.name);
                }
                for (auto& arc_out : _vgraph->get_out_arc_its(op.name)) {
                    io& out = arc_out->weight();
                    if (this->check_access(out)) {
                        continue;
                    }

                    op_outputs[op.name].insert(out.name);
                }
            }

            // debug info
            for (const auto& op : this->_wait_que) {
                LOG(INFO) << "op.name=" << op.name;
                int i = 0;
                for (const auto& in : op_inputs[op.name]) {
                    LOG(INFO) << "input[" << i++ << "]=" << in;
                }
                i = 0;
                for (const auto& out : op_outputs[op.name]) {
                    LOG(INFO) << "output[" << i++ << "]=" << out;
                }
            }

            // check if some op nerver get feed
            for (const auto& op : op_inputs) {
                const auto& op_name = op.first;
                const auto& inputs = op.second;

                std::set<std::string> others_output;
                for (const auto& p : op_outputs) {
                    if (op_name == p.first) {
                        continue;
                    }
                    for (auto& x : p.second) {
                        others_output.insert(x);
                    }
                }

                std::set<std::string> lack_of;
                std::set_difference(
                    inputs.begin(), inputs.end()
                    , others_output.begin(), others_output.end()
                    ,  std::inserter(lack_of, lack_of.begin()));
                if (lack_of.size() != 0) {
                    LOG(INFO) << "lack of:";
                    for (auto x : lack_of) {
                        LOG(INFO) << x;
                    }
                    LOG(FATAL) << "Failed with lack of data provision";
                }
            }

            LOG(FATAL) << "unkown topo problem";
        }
        wait_que_size = _wait_que.size();
    }
    auto exec_node_order = this->get_exec_node_in_order();
    _vgraph->set_exec_order(exec_node_order);
}

bool Scheduler::is_fixed(io& io_arg) {
    auto it = std::find(_fix_io_res.begin(), _fix_io_res.end(), io_arg);
    if (it != _fix_io_res.end()) {
        return true;
    }

    return false;
}

bool Scheduler::is_target_fixed(io& io_arg) {
    io target_io = io_arg;
    auto search_target = [&](Arc<std::string, io>& arc) {
        auto share_from = target_io.share_from;
        if(arc.weight().name == share_from) {
            target_io = arc.weight();
            return Status::EXIT(" Find the matched target arc io. ");
        }
        return Status::OK();
    };
    _vgraph->Scanner->BFS_Edge(search_target);
    if(is_fixed(target_io)) {
        return true;
    }
    return false;
}

std::vector<std::string> Scheduler::get_exec_node_in_order() {
    auto& exec_node_in_order = this->get_exec_que();
    std::vector<std::string> ret;

    for (auto& tmp_node : exec_node_in_order) {
        ret.push_back(tmp_node.name);
    }

    return ret;
}


} /* namespace graph */

} /* namespace anakin */


