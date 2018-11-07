#include "framework/graph/llvm/fusion/graph_pattern.h"

namespace anakin {

namespace graph {

std::unordered_map<Fusion, std::function<int(VGraph*, Pattern*)>, FusionHash> FusionSniffer = {
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
                    auto ori_arc_out_its = vgraph->get_out_arc_its(vgraph_next_node.name);
                    std::vector<std::string> pattern_tops;

                    for (auto out_it : ori_arc_out_its) {
                        pattern_tops.push_back(out_it->top());
                    }

                    // update arcs
                    for (int tops_idx = 0; tops_idx < pattern_tops.size(); tops_idx++) {
                        Arc<std::string, io> arc(node_merge.name, pattern_tops[tops_idx]);
                        auto arc_in_its = vgraph->get_in_arc_its(pattern_tops[tops_idx]);

                        for (int in_arc_idx = 0; in_arc_idx < arc_in_its.size(); in_arc_idx++) {
                            if (arc_in_its[in_arc_idx]->bottom() == vgraph_next_node.name) {
                                // update in arc of node next pattern
                                vgraph->update_in_arc(arc, in_arc_idx);
                                arc_in_its[in_arc_idx]->weight().name = arc.name();
                                break;
                            }
                        }
                    }

                    for (auto & node_temp : node_merge.mergeNodes) {
                        vgraph->remove(node_temp.name);
                    }

                    for (int tops_idx = 0; tops_idx < pattern_tops.size(); tops_idx++) {
                        Arc<std::string, io> arc(node_merge.name, pattern_tops[tops_idx]);
                        auto& io_tmp = arc.weight();
                        io_tmp.name = arc.name();
                        vgraph->add_out_arc(arc);
                    }

                    node_merge.mergeNodeNames = pattern_node_name_saves;
                    param_node = node_merge;


                    return 0;
                } else {
                    return 0; // continue searching
                }
            };
            vgraph->Scanner->BFS(search_vgraph, pattern);
            return 0;
        }
    },
    {
        IN_PARELLEL,
        [](VGraph * vgraph, Pattern * pattern) ->int {
            //function: check accessible between node and nodelist
            auto table = vgraph -> connect_table();
            //auto table = std::map<std::pair<std::string, std::string>, int>();
            auto check_accessible_node_and_nodelist = [&](std::string name, std::vector<std::string> namelist) -> bool {
                for (std::string another_name : namelist) {
                    if (table[ {name, another_name}] || table[ {another_name, name}]) {
                        return true;
                    }
                }

                return false;
            };
            //function: match in graph begin with one node by pattern
            auto search_vgraph = [&](node & param_node, Pattern * param_pattern) {

                std::vector<std::string> pattern_in_name_vec = param_pattern->get_graph_ins();

                if (pattern_in_name_vec.size() <= 1) {
                    LOG(WARNING) << "pattern node <= 1, fusion searching will use search_vgraph_from_one_node";
                    return 0;
                }

                std::vector<std::string> pattern_in_opname_vec;

                for (std::string name : pattern_in_name_vec) {
                    auto in_node = (*param_pattern)[name];
                    pattern_in_opname_vec.push_back(in_node.opName);
                }

                //
                //\biref first match all in_node
                //
                std::string node_name = param_node.name;
                std::string node_op_name = param_node.opName;

                if (std::count(pattern_in_opname_vec.begin(), pattern_in_opname_vec.end(), node_op_name) <= 0) {
                    return 0;
                }

                std::vector<node> merge_node_list {param_node};
                std::vector<std::string> merge_node_name_list {param_node.name};
                //make follow name list of pattern,which has same order with merge_node_name_list
                std::vector<std::string> pattern_follow_name_list;
                int index = std::find(pattern_in_opname_vec.begin(), pattern_in_opname_vec.end(), node_op_name) -
                            pattern_in_opname_vec.begin();
                pattern_follow_name_list.push_back(pattern_in_name_vec[index]);
                //erase op_name
                pattern_in_opname_vec.erase(pattern_in_opname_vec.begin() + index);
                pattern_in_name_vec.erase(pattern_in_name_vec.begin() + index);

                //find suitable node by node
                for (auto another_node = vgraph -> begin(); another_node != vgraph -> end(); ++another_node) {
                    std::string another_name = another_node -> first;
                    std::string another_op_name = another_node -> second.opName;

                    if (std::count(pattern_in_opname_vec.begin(), pattern_in_opname_vec.end(), another_op_name) > 0) {
                        if (!check_accessible_node_and_nodelist(another_name, merge_node_name_list)) {
                            merge_node_list.push_back(another_node -> second);
                            merge_node_name_list.push_back(another_name);
                            int index = std::find(pattern_in_opname_vec.begin(), pattern_in_opname_vec.end(), another_op_name) -
                                        pattern_in_opname_vec.begin();
                            pattern_follow_name_list.push_back(pattern_in_name_vec[index]);
                            pattern_in_opname_vec.erase(pattern_in_opname_vec.begin() + index);
                            pattern_in_name_vec.erase(pattern_in_name_vec.begin() + index);
                        }
                    }
                }

                if (pattern_in_opname_vec.size() > 0) {
                    return 0;
                }

                std::vector<std::string> graph_name_list = merge_node_name_list;

                CHECK_EQ(graph_name_list.size(), pattern_follow_name_list.size()) << "unmatched pattern size";

                std::vector<std::string> pattern_node_name_saves = pattern_follow_name_list;
                //
                //\biref match next nodes with each in node
                //
                std::vector<std::string> graph_top_name;

                for (int i = 0; i < graph_name_list.size(); ++i) {

                    std::string graph_node_name = graph_name_list[i];
                    std::string pattern_node_name = pattern_follow_name_list[i];
                    auto pattern_arc_its = param_pattern -> get_out_arc_its(pattern_node_name);

                    //save top node with each in_node in graph
                    if (pattern_arc_its.size() == 0) {
                        graph_top_name.push_back(graph_node_name);
                    }

                    while (pattern_arc_its.size()) {
                        CHECK_EQ(pattern_arc_its.size(), 1) << "unsupport pattern !!";
                        auto graph_arc_its = vgraph -> get_out_arc_its(graph_node_name);

                        if (graph_arc_its.size() != 1) {
                            return 0;
                        }

                        graph_node_name = graph_arc_its[0] -> top();
                        pattern_node_name = pattern_arc_its[0] -> top();
                        auto graph_node = (*vgraph)[graph_arc_its[0] -> top()];
                        auto pattern_node = (*param_pattern)[pattern_arc_its[0] -> top()];
                        std::string graph_node_opname = graph_node.opName;
                        std::string pattern_node_opname = pattern_node.opName;

                        if (graph_node_opname != pattern_node_opname) {
                            return 0;
                        }

                        //add merge node and name
                        merge_node_list.push_back(graph_node);
                        merge_node_name_list.push_back(graph_node_name);
                        pattern_node_name_saves.push_back(pattern_node_name);
                        //check next node
                        pattern_arc_its = param_pattern -> get_out_arc_its(pattern_node_name);

                    }//while pattern...
                }//for i

                //
                //\brief match finished, update
                //
                //merge node
                node merge_node = merge_node_list[0];

                //merge_node.mergeNodes.clear();
                //LOG(INFO)<<"======find pattern can fusion of "<<merge_node.name;
                for (int i = 1; i < merge_node_list.size(); ++i) {
                    //LOG(INFO)<<"list:"<<merge_node_list[i].name;
                    merge_node += merge_node_list[i];
                }

                //update arcs
                std::vector<std::string> top_node_names;
                std::vector<std::string> in_node_names;

                //update out arcs
                for (int i = 0; i < graph_top_name.size(); ++i) {
                    //LOG(INFO) << "graph_top_name:" << graph_top_name[i];
                    if (graph_top_name[i] == merge_node.name) {
                        continue;
                    }

                    std::string topname = graph_top_name[i];
                    auto out_arc_its = vgraph -> get_out_arc_its(topname);

                    for (int j = 0; j < out_arc_its.size(); ++j) {
                        std::string toptop_name = out_arc_its[j] -> top();
                        Arc<std::string, io> arc(merge_node.name, toptop_name);

                        if (std::find(top_node_names.begin(), top_node_names.end(), toptop_name) == top_node_names.end()) {
                            top_node_names.push_back(toptop_name);
                        }

                        auto toptop_in_arcs = vgraph -> get_in_arc_its(toptop_name);
                        bool has_arc = false;

                        for (int arc_idx = 0; arc_idx < toptop_in_arcs.size(); ++arc_idx) {
                            if (toptop_in_arcs[arc_idx] -> bottom() == merge_node.name) {
                                has_arc = true;
                                break;
                            }
                        }

                        if (has_arc) {
                            continue;
                        }

                        for (int arc_idx = 0; arc_idx < toptop_in_arcs.size(); ++arc_idx) {
                            if (toptop_in_arcs[arc_idx] -> bottom() == topname) {
                                vgraph -> update_in_arc(arc, arc_idx);
                                toptop_in_arcs[arc_idx]->weight().name  = arc.name();
                                break;
                            }
                        }
                    }
                }

                //update in arcs
                for (int i = 1; i < graph_name_list.size(); ++i) {
                    std::string in_name = graph_name_list[i];
                    auto in_arc_its = vgraph -> get_in_arc_its(in_name);

                    for (int j = 0; j < in_arc_its.size(); ++j) {
                        std::string inin_name = in_arc_its[j] -> bottom();
                        Arc<std::string, io> arc(inin_name, merge_node.name);

                        if (std::find(in_node_names.begin(), in_node_names.end(), inin_name) == in_node_names.end()) {
                            in_node_names.push_back(inin_name);
                        }

                        auto inin_out_arc_its = vgraph -> get_out_arc_its(inin_name);
                        bool has_arc = false;

                        for (int arc_idx = 0; arc_idx < inin_out_arc_its.size(); ++arc_idx) {
                            if (inin_out_arc_its[arc_idx] -> top() == merge_node.name) {
                                has_arc = true;
                                break;
                            }
                        }

                        if (has_arc) {
                            continue;
                        }

                        for (int arc_idx = 0; arc_idx < inin_out_arc_its.size(); ++arc_idx) {
                            if (inin_out_arc_its[arc_idx] -> top() == in_name) {
                                vgraph -> update_out_arc(arc, arc_idx);
                                inin_out_arc_its[arc_idx]->weight().name  = arc.name();
                                break;
                            }
                        }
                    }
                }

                //delete useless vertexs
                for (int i = 1; i < merge_node_name_list.size(); ++i) {
                    vgraph -> remove(merge_node_name_list[i]);
                    merge_node.mergeNodeNames.push_back(merge_node_name_list[i]);
                }

                //add arcs
                for (auto name : top_node_names) {
                    //LOG(INFO)<<"add out name:"<<name;
                    Arc<std::string, io> arc(merge_node.name, name);
                    auto& io_tmp = arc.weight();
                    io_tmp.name = arc.name();
                    vgraph -> add_out_arc(arc);
                }

                for (auto name : in_node_names) {
                    //LOG(INFO)<<"add in name"<<name;
                    Arc<std::string, io> arc(name, merge_node.name);
                    auto& io_tmp = arc.weight();
                    io_tmp.name = arc.name();
                    vgraph -> add_in_arc(arc);
                }

                //update mergenode info
                merge_node.opName = param_pattern -> fusion_op_name();
                param_node = merge_node;
                merge_node.mergeNodeNames = pattern_node_name_saves;

                return 0;

            };

            auto search_vgraph_from_onenode = [&](node & param_node, Pattern * param_pattern) {
                int node_count = 0;

                for (auto it = param_pattern->begin(); it != param_pattern->end(); ++it) {
                    ++node_count;

                    if (node_count > 1) {
                        LOG(WARNING) << "pattern node > 1, fusion searching will use search_vgraph";
                        return 0;
                    }
                }

                std::string merge_op_name = param_pattern->begin()->second.opName;
                std::string merge_name = param_pattern->begin()->first;


                std::string node_name = param_node.name;
                std::string node_op_name = param_node.opName;
                std::vector<node> merge_node_list;

                auto node_out_arc_its = vgraph -> get_out_arc_its(node_name);

                for (int i = 0; i < node_out_arc_its.size(); ++i) {
                    std::string name = node_out_arc_its[i] -> top();
                    auto top_node = (*vgraph)[name];
                    std::string opname = top_node.opName;

                    if (opname == merge_op_name) {
                        merge_node_list.push_back(top_node);
                    }
                }

                if (merge_node_list.size() > 1) {
                    node node_merge = merge_node_list[0];
                    node_merge.mergeNodes.clear();
                    std::vector<std::string> pattern_node_name_saves;//{node_merge.name};

                    //LOG(INFO)<<"======find pattern can fusion of "<<node_merge.name;
                    for (int i = 1; i < merge_node_list.size(); ++i) {
                        LOG(INFO) << "list:" << merge_node_list[i].name;
                        node_merge += merge_node_list[i];
                        pattern_node_name_saves.push_back(merge_name + "_" + (char)(i + 0x30));
                    }

                    //update arcs
                    std::vector<std::string> top_node_names;
                    std::vector<std::string> in_node_names;

                    //update out arc
                    for (auto & merge_node : merge_node_list) {
                        if (merge_node.name == node_merge.name) {
                            continue;
                        }

                        auto node_arc_out_its = vgraph->get_out_arc_its(merge_node.name);

                        for (int i = 0; i < node_arc_out_its.size(); ++i) {
                            Arc<std::string, io> arc(node_merge.name, node_arc_out_its[i]->top());

                            if (std::find(top_node_names.begin(), top_node_names.end(),
                                          node_arc_out_its[i]->top()) == top_node_names.end()) {
                                top_node_names.push_back(node_arc_out_its[i]->top());
                            }

                            auto arc_in_its =  vgraph->get_in_arc_its(node_arc_out_its[i]->top());
                            bool has_arc = false;

                            for (int arc_idx = 0; arc_idx < arc_in_its.size(); ++arc_idx) {
                                if (arc_in_its[arc_idx] -> bottom() == node_merge.name) {
                                    has_arc = true;
                                    break;
                                }
                            }

                            if (has_arc) {
                                continue;
                            }

                            for (int in_arc_idx = 0; in_arc_idx < arc_in_its.size(); in_arc_idx++) {
                                if (arc_in_its[in_arc_idx] -> bottom() == merge_node.name) {
                                    // update in arc of node next pattern
                                    vgraph -> update_in_arc(arc, in_arc_idx);
                                    //vgraph -> add_out_arc(arc);
                                    arc_in_its[in_arc_idx] -> weight().name  = arc.name();//anything todo
                                    break;
                                }
                            }
                        }
                    }

                    //update in arcs
                    for (auto & merge_node : merge_node_list) {
                        if (merge_node.name == node_merge.name) {
                            continue;
                        }

                        auto node_arc_in_its = vgraph->get_in_arc_its(merge_node.name);

                        for (int i = 0; i < node_arc_in_its.size(); ++i) {
                            Arc<std::string, io> arc(node_arc_in_its[i]->bottom(), node_merge.name);

                            if (std::find(in_node_names.begin(), in_node_names.end(),
                                          node_arc_in_its[i]->bottom()) == in_node_names.end()) {
                                in_node_names.push_back(node_arc_in_its[i]->bottom());
                            }

                            auto arc_out_its =  vgraph->get_out_arc_its(node_arc_in_its[i]->bottom());
                            bool has_arc = false;

                            for (int arc_idx = 0; arc_idx < arc_out_its.size(); ++arc_idx) {
                                if (arc_out_its[arc_idx] -> top() == node_merge.name) {
                                    has_arc = true;
                                    break;
                                }
                            }

                            if (has_arc) {
                                continue;
                            }

                            for (int out_arc_idx = 0; out_arc_idx < arc_out_its.size(); out_arc_idx++) {
                                if (arc_out_its[out_arc_idx] -> top() == merge_node.name) {
                                    // update in arc of node next pattern
                                    vgraph -> update_out_arc(arc, out_arc_idx);
                                    //vgraph -> add_in_arc(arc);
                                    arc_out_its[out_arc_idx] -> weight().name  = arc.name();//anything todo
                                    break;
                                }
                            }
                        }
                    }

                    //delete other nodes
                    for (int i = 1; i < merge_node_list.size(); ++i) {
                        vgraph -> remove(merge_node_list[i].name);
                        //node_merge.mergeNodeNames.push_back(merge_node_list[i].name);
                    }

                    //add arcs
                    for (auto name : top_node_names) {
                        //LOG(INFO)<<"add out name:"<<name;
                        Arc<std::string, io> arc(node_merge.name, name);
                        auto& io_tmp = arc.weight();
                        io_tmp.name = arc.name();
                        vgraph -> add_out_arc(arc);

                    }

                    for (auto name : in_node_names) {
                        //LOG(INFO)<<"add in name"<<name;
                        Arc<std::string, io> arc(name, node_merge.name);
                        auto& io_tmp = arc.weight();
                        io_tmp.name = arc.name();
                        vgraph -> add_in_arc(arc);
                    }

                    //update mergenode
                    node_merge.opName = param_pattern -> fusion_op_name();
                    /*
                    for (int i=0; i<pattern_node_name_saves.size(); ++i){
                        node_merge.mergeNodeNames.push_back(pattern_node_name_saves[i]);
                    }
                     */
                    node_merge.mergeNodeNames = pattern_node_name_saves;
                    //LOG(INFO)<<node_merge.opName;


                    (*vgraph)[node_merge.name] = node_merge;

                    return 0;
                } else {
                    return 0;
                }
            };

            vgraph->Scanner->DFS(search_vgraph_from_onenode, pattern);
            //vgraph->Scanner->DFS(search_vgraph, pattern);

            //test in and out arc
            std::vector<std::pair<std::string, std::string>> out_arc_vec;
            std::vector<std::pair<std::string, std::string>> in_arc_vec;

            for (auto vert = vgraph->begin(); vert != vgraph->end(); ++vert) {
                auto out_arc = vgraph -> get_out_arc_its(vert->first);

                for (auto arc : out_arc) {
                    out_arc_vec.push_back({vert->first, arc->top()});
                }

                auto in_arc = vgraph -> get_in_arc_its(vert->first);

                for (auto arc : in_arc) {
                    in_arc_vec.push_back({arc->bottom(), vert->first});
                }
            }

            for (int i = 0; i < in_arc_vec.size(); ++i) {
                //LOG(INFO)<< "in_arcs:"<<"{"<<in_arc_vec[i].first<<","<<in_arc_vec[i].second<<"}";
                if (std::find(out_arc_vec.begin(), out_arc_vec.end(), in_arc_vec[i]) == out_arc_vec.end()) {
                    LOG(INFO) << "not in out_arcs:" << "{" << in_arc_vec[i].first << "," << in_arc_vec[i].second << "}";
                }
            }

            for (int i = 0; i < out_arc_vec.size(); ++i) {
                //LOG(INFO)<< "out_arcs:"<<"{"<<out_arc_vec[i].first<<","<<out_arc_vec[i].second<<"}";
                if (std::find(in_arc_vec.begin(), in_arc_vec.end(), out_arc_vec[i]) == in_arc_vec.end()) {
                    LOG(INFO) << "not in in_arcs:" << "{" << out_arc_vec[i].first << "," << out_arc_vec[i].second <<
                              "}";
                }
            }

            return 0;
        }
    },
    {
        GRAPH,
        [](VGraph * vgraph, Pattern * pattern) ->int {
            return 0;
        }
    },
    { None, [](VGraph*, Pattern*) ->int { return 0;} }
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

    for (auto & op_name : op_name_list) {
        if (FusionOpRegister::Global()[op_name]->type() == pattern) {
            ret_vec.push_back(op_name);
        }
    }

    return ret_vec;
}

std::vector<std::string> OpFusionPatternObjectRegister::get_list_op_name_in_fusion_order_of(
    Fusion pattern) {
    std::vector<std::string> ret_vec;
    auto pattern_op_name_list = this->get_list_op_name_of(pattern);
    struct PatternLevel {
        std::string op_name;
        int level {0};
        inline bool operator>(const PatternLevel& rhs) const {
            return level > rhs.level ? true : false;
        }
    };
    std::vector<PatternLevel> ready_to_sort;

    // order the  pattern_op_name_list by level
    for (auto & op_name : pattern_op_name_list) {
        auto* pattern_p = FusionOpRegister::Global()[op_name];
        PatternLevel p_temp;
        p_temp.op_name = op_name;
        p_temp.level = pattern_p->level();
        ready_to_sort.push_back(p_temp);
    }

    // sort
    std::sort(ready_to_sort.begin(), ready_to_sort.end(), std::greater<PatternLevel>());

    for (auto & item : ready_to_sort) {
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
