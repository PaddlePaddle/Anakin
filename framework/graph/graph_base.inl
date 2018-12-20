#include "framework/graph/node.h"
#include "saber/core/tensor.h"
namespace anakin {

namespace graph {

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
GraphBase<VertexNameType, VertexType, WeightType, ArcType>::GraphBase() {
    // init graph algorithm
    Scanner = new Algorithm<VertexNameType, VertexType, WeightType, ArcType>(this); 
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
GraphBase<VertexNameType, VertexType, WeightType, ArcType>::GraphBase(size_t size) {
    // TODO ...
    LOG(WARNING) <<"Not Impl Yet";
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
GraphBase<VertexNameType, VertexType, WeightType, ArcType>::~GraphBase() {
    all_clear();
    delete Scanner;
    Scanner = nullptr;
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
void GraphBase<VertexNameType, VertexType, WeightType, ArcType>::arcs_clear() {
    _arcs.clear();
    _graph_out_arcs.clear();
    _graph_in_arcs.clear();
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
void GraphBase<VertexNameType, VertexType, WeightType, ArcType>::vertices_clear() {
    _vertices.clear();
}

template <>
inline  void GraphBase<std::string, std::shared_ptr<Node>, Tensor4dPtr<X86>, Edge<X86>>::vertices_clear(){
    for (auto iter=_vertices.begin(); iter != _vertices.end(); iter++){
        if(iter->second.use_count()>1){
            LOG(INFO)<<"force destory node "<<iter->first<<",count = "<<iter->second.use_count();
//            delete iter->second.get();
        }
    }
    _vertices.clear();
};
template <>
inline  void GraphBase<std::string, std::shared_ptr<Node>, Tensor4dPtr<NV>, Edge<NV>>::vertices_clear(){
    for (auto iter=_vertices.begin(); iter != _vertices.end();iter++){
        if(iter->second.use_count()>1){
            LOG(INFO)<<"force destory node "<<iter->first<<",count = "<<iter->second.use_count();
//            delete iter->second.get();
        }
    }
    _vertices.clear();
};

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
void GraphBase<VertexNameType, VertexType, WeightType, ArcType>::all_clear() {
    arcs_clear();
    vertices_clear();
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
void GraphBase<VertexNameType, VertexType, WeightType, ArcType>::add_vertex(VertexNameType vertexName, VertexType vertex) {
    if(!this->has_vertex(vertexName)) {
        _vertices[vertexName] = vertex;
        // initial out/in map
        _graph_out_arcs[vertexName].resize(0); 
        _graph_in_arcs[vertexName].resize(0);
    }
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
void GraphBase<VertexNameType, VertexType, WeightType, ArcType>::add_alias(VertexNameType vertexNameOri, VertexNameType vertexNameAlias) {
	if(vertexNameOri == vertexNameAlias) {
		return;
	}
	if(!this->has_vertex(vertexNameOri)) {
		LOG(FATAL) << "The graph doesn't have vertext " << vertexNameOri;
		return;
	}
	if(this->has_vertex(vertexNameAlias)) {
		LOG(FATAL) <<"The graph shouldn't have alias vertex("
				   <<vertexNameAlias<<") for vertex(" << vertexNameOri
				   <<"), which already exists.";
	}
	_vertices[vertexNameAlias] = _vertices[vertexNameOri];
	_graph_out_arcs[vertexNameAlias] = _graph_out_arcs[vertexNameOri];
	_graph_in_arcs[vertexNameAlias] = _graph_in_arcs[vertexNameOri];
}


template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
void GraphBase<VertexNameType, VertexType, WeightType, ArcType>::add_in_arc(ArcType& arc) {
    if(!this->has_arc(arc)){
        _arcs.push_back(arc);
        CHECK(this->has_vertex(arc.bottom()) && this->has_vertex(arc.top())) << " The arc's top or bottom is not vertex! ";
    }     
    Arc_iterator<VertexNameType, WeightType, ArcType> arc_iterator = find(arc.bottom(), arc.top()); 
    auto top_in_arcs = _graph_in_arcs[arc.top()];
    if (std::find(top_in_arcs.begin(), top_in_arcs.end(), arc_iterator) == top_in_arcs.end()){
        _graph_in_arcs[arc.top()].push_back(arc_iterator);
    }
    //_graph_in_arcs[arc.top()].push_back(arc_iterator);
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
void GraphBase<VertexNameType, VertexType, WeightType, ArcType>::update_in_arc(ArcType& arc, size_t index_of_in_arc) {
    if(!this->has_vertex(arc.top()) || !this->has_vertex(arc.bottom())) {
        LOG(FATAL) << "The graph doesn't have vertext " << arc.top() << " or " << arc.bottom();
        return;
    }
    auto& in_arc_it_list = this->get_in_arc_its(arc.top());
    CHECK(index_of_in_arc >=0 && index_of_in_arc < in_arc_it_list.size())
            << " The in arc's index should be between 0 and in arc's sizes";
    for(int i=0; i < in_arc_it_list.size(); i++) {
        if(i == index_of_in_arc) {
            in_arc_it_list[i]->bottom() = arc.bottom();
            break;
        }
    }
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
void GraphBase<VertexNameType, VertexType, WeightType, ArcType>::add_out_arc(ArcType& arc) {
    if(!this->has_arc(arc)){
        _arcs.push_back(arc);
        CHECK(this->has_vertex(arc.bottom()) && this->has_vertex(arc.top())) << " The arc's top or bottom is not vertex! ";
    }     
    Arc_iterator<VertexNameType, WeightType, ArcType> arc_iterator = find(arc.bottom(), arc.top());
    auto bottom_out_arcs = _graph_out_arcs[arc.bottom()];
    if (std::find(bottom_out_arcs.begin(), bottom_out_arcs.end(), arc_iterator) == bottom_out_arcs.end()){
        _graph_out_arcs[arc.bottom()].push_back(arc_iterator);
    }
    //_graph_out_arcs[arc.bottom()].push_back(arc_iterator);
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
void GraphBase<VertexNameType, VertexType, WeightType, ArcType>::update_out_arc(ArcType& arc, size_t index_of_out_arc) {
    if(!this->has_vertex(arc.bottom()) || !this->has_vertex(arc.top())) {
        LOG(FATAL) << "The graph doesn't have vertext " << arc.top() << " or " << arc.bottom();
        return;
    }
    auto& out_arc_it_list = this->get_out_arc_its(arc.bottom());
    CHECK(index_of_out_arc >=0 && index_of_out_arc < out_arc_it_list.size())
            << " The in arc's index should be between 0 and in arc's sizes";
    for(int i=0; i < out_arc_it_list.size(); i++) {
        if(i == index_of_out_arc) {
            out_arc_it_list[i]->top() = arc.top();
            break;
        }
    }
}


template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
void GraphBase<VertexNameType, VertexType, WeightType, ArcType>::remove(VertexNameType vertexName) {
    if(!has_vertex(vertexName)) { 
        LOG(WARNING) << " [Can't remove vertex] Target vertex:  " << vertexName;
    } else {
        iterator it = find(vertexName);
        _vertices.erase(it.origin());
        // remove corresponding arc which has vertexName
        for(iterator it = begin(); it!=end(); it++) { 
            //vertices_ss << " |-- [v_" << index << ": " << it->first << "] \n"; 
            for(auto arc_it = _graph_in_arcs[it->first].begin(); arc_it != _graph_in_arcs[it->first].end();) {
                if((*arc_it)->bottom() == vertexName || (*arc_it)->top() == vertexName) {
                    arc_it = _graph_in_arcs[it->first].erase(arc_it);
                } else {
                    ++arc_it;
                }
            }
            for(auto arc_it = _graph_out_arcs[it->first].begin(); arc_it != _graph_out_arcs[it->first].end();) {
                if((*arc_it)->bottom() == vertexName || (*arc_it)->top() == vertexName) {
                    arc_it = _graph_out_arcs[it->first].erase(arc_it);
                } else {
                    ++arc_it;
                }
            }
        }
        // remove corresponding arc in _arcs 
        for(auto& in_arc_it : _graph_in_arcs[vertexName]) {
            if(in_arc_it->top() == vertexName) {
                remove(*in_arc_it);
            }
        }
        for(auto& out_arc_it : _graph_out_arcs[vertexName]) {
            if(out_arc_it->bottom() == vertexName) {
                remove(*out_arc_it);
            }
        }

        // clear corresponding in/out arc in graph in/out map.
        auto it_out = _graph_out_arcs.find(vertexName);
        if(it_out != _graph_out_arcs.end()) {
            _graph_out_arcs.erase(it_out);
        }
        auto it_in = _graph_in_arcs.find(vertexName);
        if(it_in != _graph_in_arcs.end()) {
            _graph_in_arcs.erase(it_in);
        }
    }
}


template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
void GraphBase<VertexNameType, VertexType, WeightType, ArcType>::remove(ArcType& arc) {
    if(has_arc(arc)) {
        Arc_iterator<VertexNameType, WeightType, ArcType> it = find(arc.bottom(), arc.top());
        _arcs.erase(it.origin());
    }
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
void GraphBase<VertexNameType, VertexType, WeightType, ArcType>::remove(VertexNameType vertex_name_0, VertexNameType vertex_name_1) {
    ArcType arc(vertex_name_0, vertex_name_1);
    remove(arc);
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
bool GraphBase<VertexNameType, VertexType, WeightType, ArcType>::has_vertex(VertexNameType vertex_name) {
    iterator it_end = _vertices.end();
    iterator it = find(vertex_name);
    if(it != it_end) {
        return true;
    }
    return false;
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
bool GraphBase<VertexNameType, VertexType, WeightType, ArcType>::has_arc(VertexNameType vertex_name_0, VertexNameType vertex_name_1) {
    ArcType arc(vertex_name_0, vertex_name_1);
    return has_arc(arc);
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
bool GraphBase<VertexNameType, VertexType, WeightType, ArcType>::has_arc(ArcType& arc) {
    Arc_iterator<VertexNameType, WeightType, ArcType> it_end = _arcs.end();
    Arc_iterator<VertexNameType, WeightType, ArcType> it = std::find(_arcs.begin(), _arcs.end(), arc);
    if(it != it_end) {
        return true;
    }
    return false;
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
typename GraphBase<VertexNameType, VertexType, WeightType, ArcType>::iterator GraphBase<VertexNameType, VertexType, WeightType, ArcType>::find(VertexNameType vertex_name) {
    return _vertices.find(vertex_name);
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
Arc_iterator<VertexNameType, WeightType, ArcType> GraphBase<VertexNameType, VertexType, WeightType, ArcType>::find(VertexNameType vertex_name_0, VertexNameType vertex_name_1) {
    ArcType arc(vertex_name_0, vertex_name_1);
    Arc_iterator<VertexNameType, WeightType, ArcType> it  = std::find(_arcs.begin(), _arcs.end(), arc);
    return it;
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
typename GraphBase<VertexNameType, VertexType, WeightType, ArcType>::ArcsIteratorList& GraphBase<VertexNameType, VertexType, WeightType, ArcType>::get_out_arc_its(VertexNameType vertex_name) {
    if(has_vertex(vertex_name)) {
        return _graph_out_arcs[vertex_name];
    } else {
        LOG(FATAL) << " Graph doesn't have vertex: " << vertex_name;
        ArcsIteratorList* vec_ret_holder = new ArcsIteratorList(); // this will not be invoke
        return *vec_ret_holder;
    }
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
typename GraphBase<VertexNameType, VertexType, WeightType, ArcType>::ArcsIteratorList& GraphBase<VertexNameType, VertexType, WeightType, ArcType>::get_in_arc_its(VertexNameType vertex_name) {
    if(has_vertex(vertex_name)) {
        return _graph_in_arcs[vertex_name];
    } else {
        LOG(FATAL) << " Graph doesn't have vertex: " << vertex_name;
        ArcsIteratorList* vec_ret_holder = new ArcsIteratorList(); // this will not be invoke
        return *vec_ret_holder;
    }
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
ArcType& GraphBase<VertexNameType, VertexType, WeightType, ArcType>::get_arc(VertexNameType vertex_name_from, VertexNameType vertex_name_to) {
    if(!has_arc(vertex_name_from, vertex_name_to)) {
        LOG(FATAL) << "graph base doesn't have arc(" << vertex_name_from << " --> " << vertex_name_to;
    }
    Arc_iterator<VertexNameType, WeightType, ArcType> it_end = _arcs.end(); 
    Arc_iterator<VertexNameType, WeightType, ArcType> it = find(vertex_name_from, vertex_name_to); 
//    if(it != it_end) {
//        return *it;
//    }
    return *it;
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
std::vector<VertexNameType> GraphBase<VertexNameType, VertexType, WeightType, ArcType>::get_graph_ins() {
    std::vector<VertexNameType> vertex_names_vec;
    /*
     for (auto vertex_pair : _vertices){
        VertexNameType vertex_name = vertex_pair.first;
        auto vertex_in_arc_its = get_in_arc_its(vertex_name);
        if (vertex_in_arc_its.size() == 0){
            vertex_names_vec.push_back(vertex_name);
        }
    }
    return vertex_names_vec;
    */
    
    Arc_iterator<VertexNameType, WeightType, ArcType> it = _arcs.begin();
    Arc_iterator<VertexNameType, WeightType, ArcType> it_end = _arcs.end();
    for(; it != it_end; ++it) {
        VertexNameType vertex_name_btm = it->bottom();
        if(has_vertex(vertex_name_btm)) {
            auto& in_arcs_it = get_in_arc_its(vertex_name_btm);
            if(in_arcs_it.size() == 0) {
                vertex_names_vec.push_back(vertex_name_btm);
            }
        }
    }
    return vertex_names_vec;
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
std::vector<VertexNameType> GraphBase<VertexNameType, VertexType, WeightType, ArcType>::get_graph_outs() {
    std::vector<VertexNameType> vertex_names_vec;
    /*
    for (auto vertex_pair : _vertices){
        VertexNameType vertex_name = vertex_pair.first;
        auto vertex_out_arc_its = get_out_arc_its(vertex_name);
        if (vertex_out_arc_its.size() == 0){
            vertex_names_vec.push_back(vertex_name);
        }
    }
    return vertex_names_vec;
    */
    Arc_iterator<VertexNameType, WeightType, ArcType> it = _arcs.begin();
    Arc_iterator<VertexNameType, WeightType, ArcType> it_end = _arcs.end();
    for(; it != it_end; ++it) {
        VertexNameType vertex_name_top = it->top();
        if(!has_vertex(vertex_name_top)) {
            vertex_names_vec.push_back(vertex_name_top);
        }
    }
    return vertex_names_vec;
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
VertexType& GraphBase<VertexNameType, VertexType, WeightType, ArcType>::operator[](VertexNameType vertex_name) {
    CHECK(has_vertex(vertex_name)) << " The graph hasn't the target vertex: " << vertex_name;
    return _vertices[vertex_name];
}

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
inline std::string GraphBase<VertexNameType, VertexType, WeightType, ArcType>::to_string() {
#ifdef USE_SGX
    return "GrahBase.to_string() not implemented in SGX mode";
#else
    std::ostringstream vertices_ss;
    vertices_ss << "Graph infrastructure: \n-- Vertices: (sum  " << size() << ") \n";
    int index = 0;
    for(iterator it = begin(); it!=end(); it++, index++) {
        vertices_ss << " |-- [v_" << index << ": " << it->first << "] \n";
    }

    std::ostringstream arcs_ss;
    arcs_ss << "-- Arcs: (sum " << _arcs.size() << ") \n";
    Arc_iterator<VertexNameType, WeightType, ArcType> it = _arcs.begin();
    Arc_iterator<VertexNameType, WeightType, ArcType> it_end = _arcs.end();
    for(; it!=it_end; it++) {
        arcs_ss << " |-- (arc: " << it->bottom() << " --> " << it->top() << ") \n";
    }
    return vertices_ss.str() + arcs_ss.str();
#endif
}
 

} /* namespace graph */

} /* namespace anakin */

