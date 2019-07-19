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

#include "framework/core/net/auto_layout_config.h"
#include <unordered_set>
#include "framework/graph/node.h"
namespace anakin {

template<typename Ttype, Precision Ptype>
void AutoLayoutConfigHelper<Ttype, Ptype>::init() {
    _node_layout_hint["Input"]["nchw"] = {"nchw"};
    _node_layout_hint["Convolution"]["nchw"] = {"nchw_c8r", "nchw"};
    _node_layout_hint["Convolution"]["nchw_c8r"] = {"nchw_c8r", "nchw"};
    _node_layout_hint["ConvRelu"]["nchw"] = {"nchw_c8r", "nchw"};
    _node_layout_hint["ConvRelu"]["nchw_c8r"] = {"nchw_c8r", "nchw"};
    _node_layout_hint["ConvBatchnormScaleRelu"]["nchw"] = {"nchw_c8r", "nchw"};
    _node_layout_hint["ConvBatchnormScaleRelu"]["nchw_c8r"] = {"nchw_c8r", "nchw"};
    _node_layout_hint["ConvBatchnormScale"]["nchw"] = {"nchw_c8r", "nchw"};
    _node_layout_hint["ConvBatchnormScale"]["nchw_c8r"] = {"nchw_c8r", "nchw"};
    _node_layout_hint["Pooling"]["nchw"] = {"nchw"};
    _node_layout_hint["Pooling"]["nchw_c8r"] = {"nchw_c8r", "nchw"};
    _node_layout_hint["Dense"]["nchw_c8r"] = {"nchw"};
    _node_layout_hint["Dense"]["nchw"] = {"nchw"};
    _node_layout_hint["ReLU"]["nchw_c8r"] = {"nchw_c8r"};
    _node_layout_hint["ReLU"]["nchw"] = {"nchw"};
    _node_layout_hint["Activation"]["nchw_c8r"] = {"nchw_c8r"};
    _node_layout_hint["Activation"]["nchw"] = {"nchw"};
    _node_layout_hint["Softmax"]["nchw"] = {"nchw"};
    _node_layout_hint["Split"]["nchw_c8r"] = {"nchw_c8r"};
    _node_layout_hint["Split"]["nchw"] = {"nchw"};
    _node_layout_hint["Gather"]["nchw_c8r"] = {"nchw_c8r"};
    _node_layout_hint["Gather"]["nchw"] = {"nchw"};
    _node_layout_hint["ConvEltwise"]["nchw_c8r"] = {"nchw_c8r"};
    _node_layout_hint["ConvEltwise"]["nchw"] = {"nchw"};
    _node_layout_hint["Eltwise"]["nchw_c8r"] = {"nchw_c8r"};
    _node_layout_hint["Eltwise"]["nchw"] = {"nchw"};
    _node_layout_hint["EltwiseRelu"]["nchw_c8r"] = {"nchw_c8r"};
    _node_layout_hint["EltwiseRelu"]["nchw"] = {"nchw"};
    _node_layout_hint["Concat"]["nchw_c8r"] = {"nchw_c8r"};
    _node_layout_hint["Concat"]["nchw"] = {"nchw"};
    _node_layout_hint["Pad"]["nchw_c8r"] = {"nchw_c8r"};
    _node_layout_hint["Pad"]["nchw"] = {"nchw"};
    _node_layout_hint["Scale"]["nchw_c8r"] = {"nchw_c8r"};
    _node_layout_hint["Scale"]["nchw"] = {"nchw"};

    _node_layout_hint["Resize"]["nchw"] = {"nchw"};
    _node_layout_hint["Slice"]["nchw"] = {"nchw"};
    _node_layout_hint["Reshape"]["nchw"] = {"nchw"};
    _node_layout_hint["PriorBox"]["nchw"] = {"nchw"};
    _node_layout_hint["DetectionOutput"]["nchw"] = {"nchw"};
    _node_layout_hint["Permute"]["nchw"] = {"nchw"};
    _node_layout_hint["Flatten"]["nchw"] = {"nchw"};

    for (auto node : _node_layout_hint) {
        std::string node_name = node.first;
        auto in_out_map = node.second;
        std::unordered_map<std::string, std::vector<std::string> >out_in_map;

        for (auto in_layout_obj : in_out_map) {
            std::string in_layout = in_layout_obj.first;
            auto out_layout_vec = in_layout_obj.second;

            for (auto out_layout : out_layout_vec) {
                if (std::count(out_in_map[out_layout].begin(), out_in_map[out_layout].end(), in_layout) == 0) {
                    out_in_map[out_layout].push_back(in_layout);
                }
            }
        }

        _node_layout_hint_reverse[node_name] = out_in_map;
    }
}

template<typename Ttype, Precision Ptype>
std::unordered_map<std::string, std::string> AutoLayoutConfigHelper<Ttype, Ptype>::
auto_config_int8_edge_layout(graph::Graph<Ttype, Ptype>& graph) {
    std::unordered_map<std::string, std::string> result;
    std::unordered_set<std::string> relu_op = {"ConvBatchnormScaleRelu", "ConvRelu"};
    auto int8_edge_config = [&, this](graph::Edge<Ttype>& edge) {
        auto bottom_node = graph[edge.bottom()];
        bottom_node->bit_type();
        auto edge_name = edge.name();

        if (edge.scale().size() > 0 || relu_op.count(bottom_node->get_op_name()) > 0) {
            result[edge.name()] = "nhwc";
        } else {
            result[edge.name()] = "nchw";
        }
    };
    graph.Scanner->BFS_Edge(int8_edge_config);
    return result;
};

template<typename Ttype, Precision Ptype>
std::unordered_map<std::string, std::string> AutoLayoutConfigHelper<Ttype, Ptype>::
auto_config_node_dtype(graph::Graph<Ttype, Ptype>& graph) {

    std::unordered_map<std::string, std::string> result;
    std::unordered_set<std::string> relu_op = {"ConvBatchnormScaleRelu", "ConvRelu", "ConvEltwise"};
    auto uint8_node_config = [&, this](graph::NodePtr target_node) {
        if (target_node->bit_type() == AK_INT8) {
            if (relu_op.count(target_node->get_op_name()) > 0) {
                if (target_node->get_op_name() == "ConvEltwise") {

                    for (auto k : target_node->attr()) {
                        LOG(INFO) << "ConvEltwise attr :" << k.first;
                    }

                }

                result[target_node->name()] = "uint8";
                return;
            } else {
                result[target_node->name()] = "int8";
                return;
            }
        }
    };
    graph.Scanner->BFS(uint8_node_config);
    return result;
};

template<typename Ttype, Precision Ptype>
void AutoLayoutConfigHelper<Ttype, Ptype>::scane_dfs_int8_node(graph::Graph<Ttype, Ptype>& graph,
        graph::NodePtr& node,
        std::string last_node_dtype) {
    LOG(FATAL) << "not impl";
}

template<typename Ttype, Precision Ptype>
std::vector<std::string> AutoLayoutConfigHelper<Ttype, Ptype>::get_node_out_layout(
    std::string node_type,
    std::string in_layout) {
    if (_node_layout_hint.count(node_type) > 0) {
        return _node_layout_hint[node_type][in_layout];
    } else {
        LOG(INFO) << "not find op prefer layout " << node_type;

        if (in_layout == "nchw") {
            return {"nchw"};
        } else {
            return {};
        }
    }
}

template<typename Ttype, Precision Ptype>
std::vector<graph::Edge<Ttype>> AutoLayoutConfigHelper<Ttype, Ptype>::get_node_output_arcs(
graph::Graph<Ttype, Ptype>& graph, graph::NodePtr& node) {
    std::vector<graph::Edge<Ttype>> result;

    for (auto out_edge : graph.get_out_arc_its(node->name())) {
        result.push_back(*out_edge);
    }

    return result;
}

template<typename Ttype, Precision Ptype>
std::vector<graph::NodePtr> AutoLayoutConfigHelper<Ttype, Ptype>::get_node_output_nodes(
    graph::Graph<Ttype, Ptype>& graph, graph::NodePtr& node) {
    std::vector<graph::NodePtr> result;

    for (auto out_edge : graph.get_out_arc_its(node->name())) {
        result.push_back(graph[out_edge->top()]);
    }

    return result;
}
template<typename Ttype, Precision Ptype>
void AutoLayoutConfigHelper<Ttype, Ptype>::scane_dfs_from_input(graph::Graph<Ttype, Ptype>& graph) {
    for (auto out_name : graph.get_outs()) {
        for (auto next_arc : graph.get_in_arc_its(out_name)) {
            _layout_map_bynode[next_arc->name()] = "nchw";
            _edge_done_map[out_name] = "nchw";
        }

    }

    for (auto in_name : graph.get_ins()) {
        for (auto next_arc : graph.get_out_arc_its(in_name)) {
            scane_dfs(graph, *next_arc, "nchw", true);
        }
    }
}

template<typename Ttype, Precision Ptype>
bool AutoLayoutConfigHelper<Ttype, Ptype>::scane_dfs(graph::Graph<Ttype, Ptype>& graph,
        graph::Edge<Ttype>& edge,
        std::string suggest_layout, bool frozen_layout,
        std::unordered_map<std::string, std::string>* return_layout_map) {
    if (_layout_map_bynode.count(edge.name()) > 0) {
        return _layout_map_bynode[edge.name()] == suggest_layout;
    }

    auto node = graph[edge.top()];

    auto layout_prefer_vec = get_node_out_layout(node->get_op_name(), suggest_layout);

    if (layout_prefer_vec.size() > 0) {
        std::unordered_map<std::string, std::string> retire_layout_map;

        for (auto layout_prefer : layout_prefer_vec) {
            bool accept = true;
            bool multi_output = get_node_output_arcs(graph, node).size() > 1;

            for (auto next_arc : get_node_output_arcs(graph, node)) {
                std::string next_node_name = graph[next_arc.top()]->name();

                bool ck = false;

                if (multi_output) {
                    if (return_layout_map == nullptr) {
                        ck = scane_dfs(graph, next_arc, layout_prefer, false, &retire_layout_map);
                    } else {
                        ck = scane_dfs(graph, next_arc, layout_prefer, false, return_layout_map);
                    }
                } else {
                    ck = scane_dfs(graph, next_arc, layout_prefer, true);
                }

                accept = accept && ck;

                if (!accept) {
                    break;
                }
            }

            if (accept) {
                if (frozen_layout) {
                    _layout_map_bynode[edge.name()] = suggest_layout;

                    if (multi_output) {
                        if (return_layout_map == nullptr) {
                            for (auto next_arc : retire_layout_map) {
                                _layout_map_bynode[next_arc.first] = next_arc.second;
                            }
                        }
                    }
                } else {
                    (*return_layout_map)[edge.name()] = suggest_layout;
                }

                return true;
            }
        }

    }

    return false;

}
template<typename Ttype, Precision Ptype>
bool AutoLayoutConfigHelper<Ttype, Ptype>::check_merge(graph::Graph<Ttype, Ptype>& graph) {
    bool result = true;
    auto check_merge = [&, this](graph::Edge<Ttype>& edge) {
        auto node = graph[edge.top()];
        auto layout = _layout_map_bynode[edge.name()];

        if (layout == "") {
            LOG(ERROR) << "layout for " << edge.name() << " is empty, auto layout config failed";
            result = false;
            return;
        }

        if (graph.get_in_arc_its(node->name()).size() > 1) {
            for (auto in_edge : graph.get_in_arc_its(node->name())) {
                if (_layout_map_bynode[(*in_edge).name()] != layout) {
                    result = false;
                    LOG(ERROR) << "layout not equal " << (*in_edge).name() << "," << node->name() <<
                               _layout_map_bynode[(*in_edge).name()] << "!= " << layout;
                    return;
                }
            }
        }
    };
    graph.Scanner->BFS_Edge(check_merge);
    return result;
}
template<typename Ttype, Precision Ptype>
void AutoLayoutConfigHelper<Ttype, Ptype>::print_layout() {
    for (auto k : _layout_map_bynode) {
        LOG(INFO) << "layout " << k.first << " = " << k.second;
    }
}


#ifdef USE_CUDA
template class AutoLayoutConfigHelper<NV, Precision::FP32>;
template class AutoLayoutConfigHelper<NV, Precision::FP16>;
template class AutoLayoutConfigHelper<NV, Precision::INT8>;

#endif

#ifdef USE_X86_PLACE
template class AutoLayoutConfigHelper<X86, Precision::FP32>;
template class AutoLayoutConfigHelper<X86, Precision::FP16>;
template class AutoLayoutConfigHelper<X86, Precision::INT8>;
#endif

#ifdef AMD_GPU
template class AutoLayoutConfigHelper<AMD, Precision::FP32>;
template class AutoLayoutConfigHelper<AMD, Precision::FP16>;
template class AutoLayoutConfigHelper<AMD, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
template class AutoLayoutConfigHelper<ARM, Precision::FP32>;
#endif

#ifdef ANAKIN_TYPE_FP16
template class AutoLayoutConfigHelper<ARM, Precision::FP16>;
#endif

#ifdef ANAKIN_TYPE_INT8
template class AutoLayoutConfigHelper<ARM, Precision::INT8>;
#endif //int8

#endif //arm
}
