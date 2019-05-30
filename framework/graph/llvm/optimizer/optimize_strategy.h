#ifndef ANAKIN_FRAMEWORK_OPTIMIZER_STRATEGY_H
#define ANAKIN_FRAMEWORK_OPTIMIZER_STRATEGY_H
#include "framework/graph/graph.h"
#include "framework/core/parameter.h"
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>

namespace anakin {

namespace graph {

struct conv_samll_param{
    PTuple<int> padding;
    PTuple<int> strides;
    PTuple<int> kernel_size;
    PTuple<int> dilation_rate;
    bool with_relu{false};
    int ref_count{0};
    bool operator==(const conv_samll_param& right){
        return (padding == right.padding && strides == right.strides &&
            kernel_size == right.kernel_size && dilation_rate == right.dilation_rate && 
            with_relu == right.with_relu);
    }
};

template <typename Ttype, Precision Ptype>
class graph_strategy{
public:
    typedef Graph<Ttype, Ptype>                  graph_t;
    typedef std::vector<std::string>             node_list_t;
public:
	void apply_horizontal_combine(graph_t* graph, int iter = 1){
		node_list_t marked_nodes = _mark_horizontal_like_conv(graph);
		for (int i = 0; i < marked_nodes.size(); ++i){
                    DLOG(ERROR) << "merging" << marked_nodes[i];
		    node_list_t matched_nodes = _mark_horizontal_single_node(graph, marked_nodes[i]);
			_merge_nodes_horizontal(graph, matched_nodes, marked_nodes[i]);
		}
		//add slice param
		_add_fusion_slice_info(graph, "ConvFusion");
	}
    void apply_stride_up(graph_t* graph, int iter = 1){
        node_list_t marked_nodes = _mark_1x1_sn(graph);
        for (int i = 0; i < marked_nodes.size(); ++i){
            DLOG(ERROR) << "stride uping" << marked_nodes[i];
            _stride_up(graph, marked_nodes[i]);
        }
    }

private:
    //
    node_list_t stride_op_names{node_list_t{"Convolution", 
                                   "ConvBatchnormScale",
                                   "ConvBatchnorm",
                                   "ConvScale",
                                   "ConvBatchnormScaleRelu",
                                   "ConvRelu"}};
    node_list_t can_change_stride_names{node_list_t{"Convolution", 
                                   "ConvBatchnormScale",
                                   "ConvBatchnorm",
                                   "ConvScale",
                                   "ConvBatchnormScaleRelu",
                                   "ConvRelu"}};
    node_list_t like_conv_op_names{node_list_t{"Convolution", 
                                   "ConvBatchnormScale",
                                   "ConvBatchnorm",
                                   "ConvScale",
                                   "ConvBatchnormScaleRelu",
                                   "ConvRelu"}};
    node_list_t like_concat_op_names{node_list_t{"Concat",
                                     "Eltwise",
                                     "Gather",
                                     "EltwiseRelu",
                                     "Split"}};

    bool _is_in(std::string name, node_list_t list){
        if (std::find(list.begin(), list.end(), name) != list.end()){
            return true;
        } else {
            return false;
        }
    }

	//horizontal fusion
	node_list_t _mark_horizontal_like_conv(graph_t* graph);
	node_list_t _mark_horizontal_single_node(graph_t* graph, std::string name);
	bool _merge_nodes_horizontal(graph_t* graph, node_list_t& nodes, std::string node);
	node_list_t _add_fusion_slice_info(graph_t* graph, std::string op_name);
    //stride
    node_list_t _mark_1x1_sn(graph_t* graph);
    int _get_conv_stride(graph_t* graph, std::string name){
        PTuple<int> strides;
        auto node_p = (*graph)[name];
        strides = node_p->template get_attr<PTuple<int>>("strides");
        if (strides.size() < 2 || strides[0] != strides[1]){
            return -1;
        }
        return strides[0];
    }
    int _get_conv_kernel(graph_t* graph, std::string name){
        PTuple<int> kernel_size;
        auto node_p = (*graph)[name];
        kernel_size = node_p->template get_attr<PTuple<int>>("kernel_size");
        if (kernel_size[0] != kernel_size[1]){
            return -1;
        }
        return kernel_size[0];   
    }
    bool _set_conv_stride(graph_t* graph, std::string name, int stride){
        PTuple<int> new_strides;
        new_strides.push_back(stride);
        new_strides.push_back(stride);
        auto node_p = (*graph)[name];
        node_p->template set_attr<PTuple<int>>("strides", new_strides);
        return true;
    }
    bool _change_stride(graph_t* graph, std::string name, int stride){
        auto node_p = (*graph)[name];
        if (!_is_in(node_p->get_op_name(), can_change_stride_names)){
            return false;
        }
        int old_stride = _get_conv_stride(graph, name);
        if (old_stride == -1){
            return false;
        }
        int new_stride = stride * old_stride;
        _set_conv_stride(graph, name, new_stride);
        return true;
    }
    bool _stride_up_like_concat(graph_t* graph, std::string name, int stride = -1);
    bool _stride_up(graph_t* graph, std::string name);
    bool _check_stride_conv(graph_t* graph, std::string name, int stride = -1){
        auto node_p = (*graph)[name];
        if (!_is_in(node_p->get_op_name(), stride_op_names)){
            return false;
        }
        if (_get_conv_kernel(graph, name) != 1){
            return false;
        }
        auto node = (*graph)[name];
        if (stride != -1){
           int node_stride = _get_conv_stride(graph, name);
           return node_stride == stride; 
        }

        return true;
    }
};

template <typename Ttype, Precision Ptype>
typename graph_strategy<Ttype, Ptype>::node_list_t 
graph_strategy<Ttype, Ptype>::_mark_1x1_sn(graph_t* graph){
    
    node_list_t node_list;
    auto mark_nodes = [&](const NodePtr node_p){
        std::string name = node_p->get_op_name();
        if (_is_in(name, like_conv_op_names)){
                PTuple<int> kernel_size;
                PTuple<int> strides;
                kernel_size = node_p->template get_attr<PTuple<int>>("kernel_size");
                strides = node_p->template get_attr<PTuple<int>>("strides");
                if (kernel_size[0] == 1 && kernel_size[1] == 1 && 
                    strides[0] > 1 && strides[1] > 1 && strides[0] == strides[1]){
                    node_list.push_back(node_p->name());
                }
        }
    };
    graph->Scanner->BFS(mark_nodes);

    return node_list;
}
template <typename Ttype, Precision Ptype>
bool  
graph_strategy<Ttype, Ptype>::_stride_up_like_concat(graph_t* graph, std::string name, int stride){
    if (stride == -1){    
        auto out_arcs = graph->get_out_arc_its(name);
        if (out_arcs.size() == 0){
            return false;
        }
        if (_check_stride_conv(graph,out_arcs[0]->second())){
            stride = _get_conv_stride(graph, out_arcs[0]->second());
            if (stride == -1){
                return false;
            }
            for (int i = 1; i < out_arcs.size(); ++i){
                if (!_check_stride_conv(graph, out_arcs[i]->second(), stride)){
                    return false;
                };
            }    
        } else {
            return false;
        }
        for (int i = 0; i < out_arcs.size(); ++i){
            _set_conv_stride(graph, out_arcs[i]->second(), 1);
        }
    }
    
    
    //if go here, we make sure that all out stride is same and kernel is all 1x1
    bool has_can_change_stride_node = true;
    auto in_arcs = graph->get_in_arc_its(name);
    if (in_arcs.size() == 1 && _is_in((*graph)[in_arcs[0]->first()]->get_op_name(),
        like_concat_op_names)){
        return _stride_up_like_concat(graph, in_arcs[0]->first(), stride);
    }
    /*
    for (int i = 0; i < in_arcs.size(); ++i){
        if (_is_in(in_arcs[i]->first(), can_change_stride_names)){
            has_can_change_stride_node = true; 
            break;
        }
    }*/
    if (has_can_change_stride_node){
    //if go here, it means that we can deliver stride by this name
        for (int i = 0; i < in_arcs.size(); ++i){
            auto t_node = (*graph)[in_arcs[i]->first()];
            if (_change_stride(graph, in_arcs[i]->first(), stride)){
                _stride_up(graph, in_arcs[i]->first());
            } else {  
                //add pool op
                std::string op_name = in_arcs[i]->first() + "_pool";
                //insert pooling op
                NodePtr node_p = std::make_shared<graph::Node>();
                node_p->set_name(op_name);
                node_p->get_op_name() = "Pooling";
                graph->add_vertex(op_name, node_p);
                Edge<Ttype> out_edge(in_arcs[i]->first(), op_name);
                Edge<Ttype> in_edge(op_name, name);

                graph->add_in_arc(out_edge);
                graph->add_out_arc(out_edge);
                graph->add_out_arc(in_edge);
                graph->add_in_arc(in_edge);
                graph->remove_byio(in_arcs[i]->first(), name);
                /////////////////////////////////
                
                //kernel 1x1
                PTuple<int> kernel_size = {1,1};
                //strides nxn
                PTuple<int> strides = {stride, stride};
                //pad
                PTuple<int> padding = {0, 0};
                //other param
                std::string method = "MAX";
                bool floor = true;
                bool global_pooling = false;

                graph->AddOpAttr(op_name, "pool_size", kernel_size);
                graph->AddOpAttr(op_name, "strides", strides);
                graph->AddOpAttr(op_name, "padding", padding);
                graph->AddOpAttr(op_name, "method", method);
                graph->AddOpAttr(op_name, "cmp_out_shape_floor_as_conv", floor);
                graph->AddOpAttr(op_name, "global_pooling", global_pooling);


            }
        }
        
    }
    return true;

}


template <typename Ttype, Precision Ptype>
bool  
graph_strategy<Ttype, Ptype>::_stride_up(graph_t* graph, std::string name){
    auto node = (*graph)[name];
    int stride = _get_conv_stride(graph, name);
    int kernel_size = _get_conv_kernel(graph, name);
    if (stride <= 1 || kernel_size != 1){
        return false;
    }
    auto in_arcs = graph->get_in_arc_its(name);
    if (in_arcs.size()>1){
        LOG(WARNING) << "graph has conv with more than 2 in_arcs";
        return false;
    }
    
    std::string last_name = in_arcs[0]->first();
    auto last_node = (*graph)[last_name];
    if (_is_in(last_node->get_op_name(), can_change_stride_names)){
        if (!_change_stride(graph, last_name, stride)){
            return false;
        }
        _set_conv_stride(graph, name, 1);
        return _stride_up(graph, last_name);    
    } else if (_is_in(last_node->get_op_name(), like_concat_op_names)){
        return _stride_up_like_concat(graph, last_name);
    } else {
        return false;
    }
}

template <typename Ttype, Precision Ptype>
bool  
graph_strategy<Ttype, Ptype>::_merge_nodes_horizontal(graph_t* graph, node_list_t& node_names, std::string node){
	if (node_names.empty()){
		return false;
	}
	//check if any node has more than 1 in arc
    for (int i = 0; i < node_names.size(); ++i){
    	auto in_arcs = graph->get_in_arc_its(node_names[i]);
    	if (in_arcs.size() > 1){
    		return false;
    	}
    }
    //merge
    //update out arc
    node_list_t top_node_names;
    node_list_t in_order_names;
    auto out_arcs = graph->get_out_arc_its(node);
    for (int i = 0; i< out_arcs.size(); ++i){
        std::string temp_name = out_arcs[i]->second();
        if (_is_in(temp_name, node_names)){
            in_order_names.push_back(temp_name);
        }
    }
    node_names = in_order_names;

    std::string merged_name = node_names[0];
    auto merged_node = (*graph)[merged_name];
    for (auto cur_name : node_names){
        if (cur_name == merged_name){
            continue;
        }
        auto cur_node_outs = graph->get_out_arc_its(cur_name);
        for (int i = 0; i < cur_node_outs.size(); ++i){
            Edge<Ttype> edge(merged_name, cur_node_outs[i]->second());
            if (!_is_in(cur_node_outs[i]->second(), top_node_names)){
                top_node_names.push_back(cur_node_outs[i]->second());
            }
                            
            auto second_arc_ins = graph->get_in_arc_its(cur_node_outs[i]->second());
            bool has_arc = false;
            for (int arc_idx=0; arc_idx<second_arc_ins.size(); ++arc_idx){
                if (second_arc_ins[arc_idx] -> first() == merged_name){
                    has_arc = true;
                    break;
                }
            }
            if (has_arc){
                continue;
            }
            for (int in_arc_idx = 0; in_arc_idx < second_arc_ins.size(); in_arc_idx++) {
                if (second_arc_ins[in_arc_idx] -> first() == cur_name) {
                    // update in arc of node next pattern
                    graph -> update_in_arc(edge, in_arc_idx);
                    //vgraph -> add_out_arc(arc);
                    //second_arc_ins[in_arc_idx] -> set_weight_name();//anything todo
                    break;
                }
            }
        }
    }
    //update in arc
    node_list_t in_node_names;
    for (auto cur_name : node_names){
        if (cur_name == merged_name){
            continue;
        }
        auto cur_node_ins = graph->get_in_arc_its(cur_name);
        for (int i=0; i < cur_node_ins.size(); ++i){
            Edge<Ttype> edge(cur_node_ins[i]->first(), merged_name);
            if (!_is_in(cur_node_ins[i]->first(), in_node_names)){
                in_node_names.push_back(cur_node_ins[i]->first());
            }           
            auto first_node_outs = graph->get_out_arc_its(cur_node_ins[i]->first());
            bool has_arc = false;
            for (int arc_idx=0; arc_idx<first_node_outs.size(); ++arc_idx){
                if (first_node_outs[arc_idx] -> second() == merged_name){
                    has_arc = true;
                    break;
                }
            }
            if (has_arc){
                continue;
            }
                            
            for (int out_arc_idx = 0; out_arc_idx < first_node_outs.size(); out_arc_idx++) {
                if (first_node_outs[out_arc_idx] -> second() == cur_name) {
                    // update in arc of node next pattern
                    graph -> update_out_arc(edge, out_arc_idx);
                    //vgraph -> add_in_arc(arc);
                    //first_node_outs[out_arc_idx] -> set_weight_name();//anything todo
                    break;
                }
            }
        }
    }
    //add new arc

    for (auto name : top_node_names){
        //LOG(INFO)<<"add out name:"<<name;
        Edge<Ttype> edge(merged_name, name);
        //edge.set_weight_name();
        graph -> add_out_arc(edge);
    }
    for (auto name : in_node_names){
        //LOG(INFO)<<"add in name"<<name;
        Edge<Ttype> edge(name, merged_name);
        //edge.set_weight_name();
        graph -> add_in_arc(edge);
    }
    //
    merged_node->set_op_name("ConvFusion");
    //merge op attributes
    //now,we support most 10 nodes
    std::vector<std::string> name_str{"0","1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"};
    int str_ind = 1;
    for (int i = 0; i < node_names.size(); ++i){
        if (node_names[i] != merged_name){
            auto cur_node = (*graph)[node_names[i]];
            merged_node->Merge(*cur_node, "conv_"+name_str[str_ind]);
            ++str_ind;    
        }
    	
    }
    merged_node->template set_attr<int>("fusion_count", node_names.size());

    //delete other nodes
    for (int i=0; i < node_names.size(); ++i){
        if (node_names[i] != merged_name){
            graph -> remove(node_names[i]);
        }
    }
    return true;
}
template <typename Ttype, Precision Ptype>
typename graph_strategy<Ttype, Ptype>::node_list_t 
graph_strategy<Ttype, Ptype>::_mark_horizontal_like_conv(graph_t* graph){
    node_list_t node_list;
    node_list_t can_merge_from_nodes{"Split", "Gather"};
    auto mark_nodes = [&](const NodePtr node_p){
        if (!_is_in(node_p->get_op_name(), can_merge_from_nodes)){
            return;
        }
        auto node_arc_out_its = graph->get_out_arc_its(node_p->name());
        int node_count = 0;
        for (auto out_arc: node_arc_out_its){
        	//LOG(ERROR) << node_p->get_op_name();
        	auto top_node = (*graph)[out_arc->second()];
            if (_is_in(top_node->get_op_name(), like_conv_op_names)){
                ++node_count;
    	    }	
        }
        if (node_count > 1){
        	node_list.push_back(node_p->name());
        }
    };
    graph->Scanner->BFS(mark_nodes);

    return node_list;     
}
template <typename Ttype, Precision Ptype>
typename graph_strategy<Ttype, Ptype>::node_list_t 
graph_strategy<Ttype, Ptype>::
_mark_horizontal_single_node(graph_t* graph, std::string name){
    node_list_t node_list;
    node_list_t conv_lists;
    auto node_arc_out_its = graph->get_out_arc_its(name);
    std::unordered_map<std::string, conv_samll_param> conv_params;

    for (auto out_arc: node_arc_out_its){
    	auto top_node = (*graph)[out_arc->second()];
        if (_is_in(top_node->get_op_name(), like_conv_op_names)){
                if (graph->get_in_arc_its(out_arc->second()).size() > 1){
                	continue;
                }

                conv_samll_param param;
                auto group = top_node->template get_attr<int>("group");
                param.padding = top_node->template get_attr<PTuple<int>>("padding");
                param.strides = top_node->template get_attr<PTuple<int>>("strides");
                param.dilation_rate = top_node->template get_attr<PTuple<int>>("dilation_rate");
                param.kernel_size = top_node->template get_attr<PTuple<int>>("kernel_size");
                param.with_relu = top_node->inspect_attr("relu_0_alpha");
                if (group == 1){
                	conv_params[out_arc->second()] = param;
                }
    	}	
    }
    std::vector<conv_samll_param> param_count;
    for (auto it = conv_params.begin(); it != conv_params.end(); ++it) {
    	int j = 0;
        for (j = 0; j < param_count.size(); ++j){
        	if (it->second == param_count[j]){
        		param_count[j].ref_count += 1;
        		break;
        	}
        }
        if (j == param_count.size()){
        	param_count.push_back(it->second);
        }
    }

    for (int i = 0; i < param_count.size(); ++i){
    	//LOG(INFO) << param_count[i].ref_count;
    	if (param_count[i].ref_count > 0){
    		for (auto it = conv_params.begin(); it != conv_params.end(); ++it){
    			if (it->second == param_count[i]){
    				node_list.push_back(it->first);
    			}
    		}
    	}
    }
    return node_list;     
}

template <typename Ttype, Precision Ptype>
typename graph_strategy<Ttype, Ptype>::node_list_t 
graph_strategy<Ttype, Ptype>::_add_fusion_slice_info(graph_t* graph, std::string op_name){
    node_list_t node_list;
    auto add_slice_info = [&](NodePtr node_p){
        if (node_p->get_op_name() != op_name){
    		return;
    	}
    	auto node_out_arcs = graph->get_out_arc_its(node_p->name());
    	PTuple<int> slice_info;
    	int slice_count = 0;
    	for (int i = 0; i < node_out_arcs.size(); ++i){
            auto top_node = (*graph)[node_out_arcs[i]->second()];

            if (!top_node->inspect_attr("fusion_count")){
            	slice_info.push_back(1);
            	slice_count += 1;
            } else {
            	int fusion_count = top_node->template get_attr<int>("fusion_count");
            	slice_info.push_back(fusion_count);
            	slice_count += fusion_count;
            }
    	}
    	node_p->template set_attr<PTuple<int>>("slice_info", slice_info);
    	node_list.push_back(node_p->name());
    };

    graph->Scanner->BFS(add_slice_info);
    return node_list;
}

}//namespace graph
}//namespace anakin 


#endif
