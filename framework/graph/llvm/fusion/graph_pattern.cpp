#include "framework/graph/llvm/fusion/graph_pattern.h"

namespace anakin {

namespace graph {

const std::unordered_map<Fusion, std::function<int(VGraph*, Pattern*)>, FusionHash> FusionSniffer
= {
    {
        IN_ORDER,
        [](VGraph * vgraph, Pattern * pattern) -> int {
            auto search_vgraph = [&](node & param_node, Pattern * param_pattern) {
                // for reset the virtual graph fusion node name
                // which is useful for define the parameter of fusion op
                std::vector<std::string> pattern_node_name_saves;
                node node_merge = param_node;
                auto ins = param_pattern->get_graph_ins();
                CHECK_EQ(ins.size(), 1) << " The IN_ORDER pattern graph should only have one input";
                auto& pt_node = (*param_pattern)[ins[0]];

                if (pt_node.opName == param_node.opName) { // match first
                    auto pattern_arc_out_its = param_pattern->get_out_arc_its(pt_node.name);
                    auto vgraph_arc_out_its = vgraph->get_out_arc_its(param_node.name);

                    if (pattern_arc_out_its.size() != 1 or vgraph_arc_out_its.size() != 1) {
                        return -1;
                    }

                    auto pattern_next_node = (*param_pattern)[pattern_arc_out_its[0]->top()];
                    auto vgraph_next_node = (*vgraph)[vgraph_arc_out_its[0]->top()];

                    while (pattern_arc_out_its.size() && vgraph_arc_out_its.size()) {
                        if (pattern_arc_out_its.size() != 1 or vgraph_arc_out_its.size() != 1) {
                            return -1;
                        }

                        if (!(vgraph->check_pass(vgraph_arc_out_its[0]->bottom(), vgraph_arc_out_its[0]->top()))) {
                            return -1;
                        }

                        pattern_next_node = (*param_pattern)[pattern_arc_out_its[0]->top()];
                        vgraph_next_node = (*vgraph)[vgraph_arc_out_its[0]->top()];

                        if (pattern_next_node.opName != vgraph_next_node.opName) {
                            return -1;
                        }

                        pattern_arc_out_its = param_pattern->get_out_arc_its(pattern_next_node.name);
                        vgraph_arc_out_its = vgraph->get_out_arc_its(vgraph_next_node.name);
                        node_merge += vgraph_next_node;
                        pattern_node_name_saves.push_back(pattern_next_node.name);
                    }

                    // need to replace
                    node_merge.opName = param_pattern->fusion_op_name();
                    // pattern ins and outs in original vgraph
                    //auto ori_arc_in_its = vgraph->get_in_arc_its(param_node.name);
                    auto ori_arc_out_its = vgraph->get_out_arc_its(vgraph_next_node.name);
                    std::vector<std::string> pattern_tops;

                    for (auto out_it : ori_arc_out_its) {
                        pattern_tops.push_back(out_it->top());
                    }

                    //vgraph->remove(node_merge.name);
                    for (auto& node_temp : node_merge.mergeNodes) {
                        vgraph->remove(node_temp.name);
                    }

                    node_merge.mergeNodeNames = pattern_node_name_saves;
                    param_node = node_merge;
                    //vgraph->add_vertex(node_merge.name, node_merge);

                    /*for (auto& in_arc_it : ori_arc_in_its) {
                        Arc<std::string, io> arc(in_arc_it->bottom(), node_merge.name);
                        auto& io_tmp = arc.weight();
                        io_tmp.name = arc.name();
                        vgraph->add_in_arc(arc); // set io name for future analysis
                        vgraph->add_out_arc(arc);
                    }*/
                    for (auto& top : pattern_tops) {
                        Arc<std::string, io> arc(node_merge.name, top);
                        auto& io_tmp = arc.weight();
                        io_tmp.name = arc.name(); // set io name for future analysis
                        vgraph->add_out_arc(arc);
                        vgraph->add_in_arc(arc);
                    }

                    return 0;
                } else {
                    return 0; // continue searching
                }
            };
            vgraph->Scanner->BFS(search_vgraph, pattern);
        }
    },
    {
        IN_PARELLEL,
        [](VGraph * vgraph, Pattern * pattern) ->int {
        }
    },
    {
        GRAPH,
        [](VGraph * vgraph, Pattern * pattern) ->int {
        }
    },
    { None, [](VGraph*, Pattern*) ->int {} }
};

Pattern& Pattern::name(std::string fusion_op_name) {
    _fusion_op_name = fusion_op_name;
    return *this;
}

Pattern& Pattern::AddOpNode(std::string node_name, std::string op_name) {
    _level++;
    node tmp_node; 
    tmp_node.name = node_name; 
    tmp_node.opName = op_name;
    this->add_vertex(node_name, tmp_node);
    return *this;
}

Pattern& Pattern::AddConnect(std::string node_name_btm, std::string node_name_top) {
    Arc<std::string, io> arc(node_name_btm, node_name_top);
    auto& io_tmp = arc.weight();
    io_tmp.name = arc.name();
    this->add_in_arc(arc);
    this->add_out_arc(arc);
    return *this;
}

Pattern& Pattern::CreatePattern(std::function<void(VGraph*)> create_pattern) {
    _pattern_create = create_pattern;
    return *this;
}

Pattern& Pattern::Type(Fusion fusion_type) {
    _type = fusion_type;
    return *this;
}

std::vector<std::string> OpFusionPatternObjectRegister::get_list_op_name_of(Fusion pattern) {
    std::vector<std::string> ret_vec;
    auto& op_name_list = this->get_list_op_name();

    for (auto& op_name : op_name_list) {
        if (FusionOpRegister::Global()[op_name]->type() == pattern) {
            ret_vec.push_back(op_name);
        }
    }

    return ret_vec;
}

std::vector<std::string> OpFusionPatternObjectRegister::get_list_op_name_in_fusion_order_of(Fusion pattern) {
    std::vector<std::string> ret_vec;
    auto pattern_op_name_list = this->get_list_op_name_of(pattern);
    struct PatternLevel {
        std::string op_name;
        int level{0};
        inline bool operator>(const PatternLevel& rhs) { return level > rhs.level ? true:false; }
    };
    std::vector<PatternLevel> ready_to_sort;
    // order the  pattern_op_name_list by level
    for(auto& op_name : pattern_op_name_list) {
        auto* pattern_p = FusionOpRegister::Global()[op_name];
        PatternLevel p_temp;
        p_temp.op_name = op_name;
        p_temp.level = pattern_p->level();
        ready_to_sort.push_back(p_temp);
    }
    // sort
    std::sort(ready_to_sort.begin(), ready_to_sort.end(), std::greater<PatternLevel>());
    for(auto& item : ready_to_sort) {
        ret_vec.push_back(item.op_name);
    }
    return ret_vec;
}


std::vector<std::string>& OpFusionPatternObjectRegister::get_list_op_name() {
    return this->get_list_name();
}

Pattern* OpFusionPatternObjectRegister::operator[](const std::string op_name) {
    return ObjectRegister<Pattern>::operator[](op_name);
}

void OpFusionPatternObjectRegister::add_alias(const std::string& ori_op_name,
        const std::string& op_name_alias) {
    this->__alias__(ori_op_name, op_name_alias);
}

} /* namespace graph */

} /* namespace anakin */
