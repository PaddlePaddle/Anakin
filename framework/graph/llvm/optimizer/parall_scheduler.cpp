#include "framework/graph/llvm/optimizer/parall_scheduler.h"

namespace anakin {

namespace graph {

void SyncFlagController::init(VGraph* vgraph) {
    auto graph_in_names = vgraph->get_graph_ins();
    int lane = 0;

    for (auto & name : graph_in_names) {
        (*vgraph)[name].lane = lane;
        auto& arc_out_its_of_graph_in = vgraph->get_out_arc_its(name);
        CHECK_EQ(arc_out_its_of_graph_in.size(),
                 1) << "Parall analysis: graph input node should have one out arc.";
        _map_io_to_lane[arc_out_its_of_graph_in[0]->weight()] = lane;
        map_io_to_vgraph(arc_out_its_of_graph_in[0]->weight(), vgraph);
        (*vgraph)[name].lane = lane;
        (*vgraph)[name].need_wait = false;
        lane++;
    }
}

void SyncFlagController::node_sync_flags(node& node_arg, VGraph* vgraph) {
    std::vector<int> lanes;
    auto& node_arc_in_its = vgraph->get_in_arc_its(node_arg.name);
    auto has_lane = [&](int lane) {
        for (auto & tmp_lane : lanes) {
            if (tmp_lane == lane) {
                return true;
            }
        }

        return false;
    };

    if (node_arc_in_its.size()) {
        for (auto & arc_it : node_arc_in_its) {
            auto& io_in = arc_it->weight();

            if (!has_lane(_map_io_to_lane[io_in])) {
                lanes.push_back(_map_io_to_lane[io_in]);
            }
        }

        if (lanes.size() >= 2) {
            (*vgraph)[node_arg.name].need_wait = true;
        }

        (*vgraph)[node_arg.name].lane = lanes[0];
    }
}

void SyncFlagController::io_sync_flags(node& node_arg, VGraph* vgraph) {
    int lane = (*vgraph)[node_arg.name].lane;
    auto& node_arc_out_its = vgraph->get_out_arc_its(node_arg.name);

    for (auto & arc_it : node_arc_out_its) {
        auto& io_out = arc_it->weight();
        _map_io_to_lane[io_out] = lane;
        map_io_to_vgraph(io_out, vgraph);
    }
}

void SyncFlagController::map_io_to_vgraph(io& io_arg, VGraph* vgraph) {
    auto map_io = [&, this](Arc<std::string, io>& arc) -> Status {
        auto& tmp_io = arc.weight();

        if (tmp_io == io_arg) {
            tmp_io.lane = _map_io_to_lane[io_arg];
            return Status::EXIT(" Find the matched target arc io. ");
        }

        return Status::OK();
    };
    vgraph->Scanner->BFS_Edge(map_io);
}

void ParallScheduler::Run() {
    _sync_flag_ctl.init(_vgraph);

    while (!(this->_wait_que.empty())) {
        // lanuch the acessible op and remove it from wait que.
        for (auto op_it = this->_wait_que.begin(); op_it != this->_wait_que.end();) {
            if (callable(*op_it)) {
                _sync_flag_ctl.node_sync_flags(*op_it, _vgraph); // set in node  sync flags
                launch(*op_it);
                _sync_flag_ctl.io_sync_flags(*op_it, _vgraph); // set out arc sync flags
                op_it = this->_wait_que.erase(op_it);
            } else {
                ++op_it;
            }
        }
    }
}

} /* namespace graph */

} /* namespace anakin */


