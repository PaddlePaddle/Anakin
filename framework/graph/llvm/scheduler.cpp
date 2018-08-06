#include "framework/graph/llvm/scheduler.h"

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
    while (!(this->_wait_que.empty())) {
        // lanuch the acessible op and remove it from wait que.
        for (auto op_it = this->_wait_que.begin(); op_it != this->_wait_que.end();) {
            if (callable(*op_it)) {
                launch(*op_it);
                op_it = this->_wait_que.erase(op_it);
            } else {
                ++op_it;
            }
        }
    }
}

bool Scheduler::is_fixed(io& io_arg) {
    auto it = std::find(_fix_io_res.begin(), _fix_io_res.end(), io_arg);

    if (it != _fix_io_res.end()) {
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


