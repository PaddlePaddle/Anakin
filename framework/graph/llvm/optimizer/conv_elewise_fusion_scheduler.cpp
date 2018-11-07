#include "framework/graph/llvm/optimizer/conv_elewise_fusion_scheduler.h"

namespace anakin {

namespace graph {

bool ConvElsFusionScheduler::callable(node& node_arg) {
    if (_helper.has_node(node_arg)) {
        auto& node_arc_out_its = _vgraph->get_out_arc_its(node_arg.name);
        auto& node_arc_in_its = _vgraph->get_in_arc_its(node_arg.name);
        CHECK_EQ(node_arc_out_its.size(),
                 1) << "Conv+eltwise analysis: Convolution like op should have only one output.";
        auto& node_next = (*_vgraph)[node_arc_out_its[0]->top()];

        if (node_next.opName == "EltwiseRelu" || node_next.opName == "Eltwise") {
            auto& elt_node_in_its = _vgraph->get_in_arc_its(node_next.name);

            for (auto & it : elt_node_in_its) {
                if (it->bottom() != node_arg.name) {
                    if (!_helper.need_wait(it->bottom())) {
                        _helper.push_wait(node_arg.name);

                        if (!this->have_launched((*_vgraph)[it->bottom()])) {
                            /*std::vector<io> io_in;
                            for (auto& arc_it : node_arc_in_its) {
                                io_in.push_back(arc_it->weight());
                            }
                            _helper.set_holder(io_in, _vgraph);*/
                            //_helper.register_pair(node_arg.name, node_next.name);
                            if ((*_vgraph)[it->bottom()].opName == "Split") {
                                _helper.register_pair(node_arg.name, node_next.name);
                            } else {
                                _helper.register_pair(it->bottom(), node_next.name);
                            }

                            //_helper.register_pair(it->bottom(), node_next.name);
                            return false;
                        } else {
                            _helper.release(node_arg.name);
                        }

                        break;
                    }
                }
            }
        }
    }

    // original code
    auto& node_arc_in_its = _vgraph->get_in_arc_its(node_arg.name);
    std::vector<io> io_in;

    for (auto & arc_it : node_arc_in_its) {
        io_in.push_back(arc_it->weight());
    }

    return this->check_access(io_in);
}

void ConvElsFusionScheduler::Run() {
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

    // complete fusion replacement for conv+eltwise
    auto& pairs = _helper.get_replace_pairs();

    for (auto & tmp_pair : pairs) {
        auto& node_conv = (*_vgraph)[tmp_pair.conv_name];
        auto& node_eltwise = (*_vgraph)[tmp_pair.eltwise_name];
        node_conv += node_eltwise; // merge node parameter
        node_conv.register_keep(node_conv.mergeNodes.size() - 1); // keep eltwise node in reconstruction
        node_conv.mergeNodeNames.push_back("merge"); // eltwise op's pattern name is equal to its original attr's name
        node_conv.opName = "ConvEltwise";

        node_eltwise.opName = "Gather"; // change eltwise op to Gather op
    }

    // set exec order for vgraph
    auto exec_node_order = this->get_exec_node_in_order();
    _vgraph->set_exec_order(exec_node_order);
}


} /* namespace graph */

} /* namespace anakin */


