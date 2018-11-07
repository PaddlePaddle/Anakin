#include "framework/graph/llvm/virtual_graph.h"
#include "framework/graph/llvm/fusion/graph_pattern.h"

namespace anakin {

namespace graph {

std::string io::ToString() {
    std::string msg;

    if (shared) {
        msg = name + " shared_from: " + share_from;
    } else {
        msg = name + " not_shared";
    }

    return msg;
}

std::string node::ToString() {
    std::ostringstream msg;

    if (mergeNodes.size()) {
        msg << name << " : op(" << opName << ") merge(";

        for (auto & tmp_node : mergeNodes) {
            msg << tmp_node.name << " : " << tmp_node.opName << ",";
        }

        msg << ") lane(" << lane << ") need_wait(" << need_wait << ")";
    } else {
        msg << name << " : op(" << opName << ") lane(" << lane << ") need_wait(" << need_wait << ")";
    }

    return msg.str();
}

void VGraph::Match(VGraph* vgraph_pattern) {
    if (Pattern* pattern = dynamic_cast<Pattern*>(vgraph_pattern)) {
        switch (pattern->type()) {
        case IN_ORDER: {
            FusionSniffer[IN_ORDER](this, pattern);
        }
        break;

        case IN_PARELLEL: {
        } break;

        case GRAPH: {
        } break;

        default :
            break;
        }
    } else {
        LOG(FATAL) << " the input vgraph pattern must be pointer of the class Pattern";
    }
}

bool VGraph::check_pass(std::string bottom, std::string top) {
    std::pair<std::string, std::string> arc_tmp(bottom, top);
    auto it = std::find(_registed_outs.begin(), _registed_outs.end(), arc_tmp);

    if (it != _registed_outs.end()) {
        return false;
    }

    return true;
}

//check if bottom connect to top
bool VGraph::check_accessible(std::string bottom, std::string top) {
    LOG(INFO) << "running";

    if (!check_pass(bottom, top) || bottom == top) {
        return true;
    }

    auto bottom_out_arc_its = this -> get_out_arc_its(bottom);

    for (int i = 0; i < bottom_out_arc_its.size(); ++i) {
        std::string mid_name = (*this)[bottom_out_arc_its[i] -> top()].name;

        if (check_accessible(mid_name, top)) {
            return true;
        }
    }

    return false;
}

std::map<std::pair<std::string, std::string>, int> VGraph::connect_table() {
    std::map<std::pair<std::string, std::string>, int> table_map;

    for (auto node0 = this -> begin(); node0 != this -> end(); ++node0) {
        for (auto node1 = this -> begin(); node1 != this -> end(); ++node1) {
            table_map[ {node0->first, node1->first}] = 0;
        }
    }

    for (auto gnode = this -> begin(); gnode != this -> end(); ++gnode) {

        std::stack<std::string> stk;
        auto out_arc_its = this -> get_out_arc_its(gnode->first);
        table_map[ {gnode->first, gnode->first}] = 1;


        for (auto arc : out_arc_its) {
            stk.push(arc->top());
        }

        std::vector<std::string> flag;

        while (!stk.empty()) {
            std::string topname = stk.top();
            stk.pop();

            if (std::find(flag.begin(), flag.end(), topname) != flag.end()) {
                continue;
            }

            table_map[ {gnode->first, topname}] = 1;
            //if (gnode->first=="conv_4e_3x3")
            // LOG(INFO)<<"add node:"<<topname;

            auto out_arc_its = this -> get_out_arc_its(topname);

            for (auto arc : out_arc_its) {
                stk.push(arc->top());
            }

            flag.push_back(topname);
        }
    }

    return table_map;
}

void VGraph::register_outs(std::string bottom, std::string top) {
    std::pair<std::string, std::string> arc_tmp(bottom, top);

    if (check_pass(bottom, top)) {
        _registed_outs.push_back(arc_tmp);
    }
}


} /* namespace graph */

} /* namespace anakin */


