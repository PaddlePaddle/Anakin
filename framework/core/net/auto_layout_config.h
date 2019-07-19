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

#ifndef ANAKIN_FRAMEWORK_CORE_NET_AUTO_LAYOUT_CONFIG_H
#define ANAKIN_FRAMEWORK_CORE_NET_AUTO_LAYOUT_CONFIG_H

#include "framework/graph/graph.h"
#include "framework/core/net/operator_func.h"
#include "framework/core/net/calibrator_factory.h"
namespace anakin {
template<typename Ttype, Precision Ptype>
class AutoLayoutConfigHelper {
public:
    AutoLayoutConfigHelper() {
        init();
    }
    bool check_merge(graph::Graph<Ttype, Ptype>& graph);
    void print_layout();
    void scane_dfs_from_input(graph::Graph<Ttype, Ptype>& graph);
    std::unordered_map<std::string, std::string> get_config_layout(){
        return _layout_map_bynode;
    };
    std::unordered_map<std::string, std::string> auto_config_int8_edge_layout(graph::Graph<Ttype, Ptype>& graph);
    std::unordered_map<std::string, std::string> auto_config_node_dtype(graph::Graph<Ttype, Ptype>& graph);

private:
    void init();
    std::vector<std::string> get_node_out_layout(std::string node_type, std::string in_layout);
    std::vector<graph::NodePtr> get_node_output_nodes(graph::Graph<Ttype, Ptype>& graph, graph::NodePtr& node);
    std::vector<graph::Edge<Ttype>> get_node_output_arcs(graph::Graph<Ttype, Ptype>& graph,
                                 graph::NodePtr& node);

    bool scane_dfs(graph::Graph<Ttype, Ptype>& graph, graph::Edge<Ttype>& edge,
                   std::string suggest_layout, bool frozen_layout,
                   std::unordered_map<std::string, std::string>* return_layout_map = nullptr);

    void scane_dfs_int8_node(graph::Graph<Ttype, Ptype>& graph, graph::NodePtr& node, std::string last_node_dtype);


    std::unordered_map<std::string, std::string> _lock_node_out_edge_map;
    std::unordered_map<std::string, std::string> _lock_node_in_edge_map;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>>
    _node_layout_hint;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>>
    _node_layout_hint_reverse;
    std::unordered_map<std::string, std::string> _edge_done_map;
    std::unordered_map<std::string, std::string> _layout_map_bynode;
};
}
#endif //ANAKIN_AUTO_LAYOUT_CONFIG_H
