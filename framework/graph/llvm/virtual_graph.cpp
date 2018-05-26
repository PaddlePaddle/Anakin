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

        for (auto& tmp_node : mergeNodes) {
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

void VGraph::register_outs(std::string bottom, std::string top) {
    std::pair<std::string, std::string> arc_tmp(bottom, top);

    if (check_pass(bottom, top)) {
        _registed_outs.push_back(arc_tmp);
    }
}


} /* namespace graph */

} /* namespace anakin */


