/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

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

#ifndef ANAKIN_GRAPH_BASE_H
#define ANAKIN_GRAPH_BASE_H 

#include "framework/graph/arc.h"
#include "framework/graph/algorithm.h"

namespace anakin {

namespace graph {

/** 
 * \brief  Template parameter:
 * \param VertexNameType  type of vertex's name (Usually string)
 * \param VertexType      type of vertex (Maybe Node class)
 * \param WeightType      type of weights of arc
*/
template<typename VertexNameType, 
         typename VertexType, 
         typename WeightType, 
         typename ArcType = Arc<VertexNameType, WeightType> >
/**
* \brief Base Graph container class.
*/
class GraphBase {
    ///< ArcsList stand for the list of arc
    typedef std::list<ArcType> ArcsList;
    ///< ArcsIteratorList stand for the list of Iterator arc 
    typedef std::vector<Arc_iterator<VertexNameType, WeightType, ArcType>> ArcsIteratorList;
public:
    GraphBase();
    GraphBase(size_t size);

    virtual ~GraphBase();

    virtual inline size_t size() { return _vertices.size(); }
    
    /// graph arc clear operations
    virtual inline void arcs_clear();
    /// graph vertices clear operations
    virtual inline void vertices_clear();
    /// graph all clear operations
    virtual inline void all_clear();

	/// add vertex to graph
	virtual void add_vertex(VertexNameType vertexName, VertexType vertex);
	virtual void add_alias(VertexNameType vertexNameOri, VertexNameType vertexNameAlias);

    /// add in/out arc to graph, if you in/out arc need order
    virtual void add_in_arc(ArcType& arc);
    virtual void update_in_arc(ArcType& arc, size_t index_of_in_arc);
    virtual void add_out_arc(ArcType& arc);
    virtual void update_out_arc(ArcType& arc, size_t index_of_out_arc);


    /// remove vertex from graph
	virtual void remove(VertexNameType vertexName);
	/// remove arc from graph
	virtual void remove(ArcType& arc);
    virtual void remove(VertexNameType vertex_name_0, VertexNameType vertex_name_1);

    /// has arc	
 	virtual bool has_arc(ArcType& arc);
    virtual bool has_arc(VertexNameType vertex_name_0, VertexNameType vertex_name_1);
 	/// has vertex
 	virtual bool has_vertex(VertexNameType vertex_name);

	/// judge if graph is directed graph, must be override.
	virtual bool directed() = 0;

    class iterator;
    /// get iterator begin
    virtual inline iterator begin() { return _vertices.begin(); }
    /// get iterator end 
    virtual inline iterator end() { return _vertices.end(); }
    /// find iterator of vertex
    virtual iterator find(VertexNameType vertex_name);

    /// find iterator of arc
    virtual Arc_iterator<VertexNameType, WeightType, ArcType> find(VertexNameType vertex_name_0, VertexNameType vertex_name_1);
	/// get out arcs of given vertex.
	virtual ArcsIteratorList& get_out_arc_its(VertexNameType vertex_name);
	/// get in arcs of given vertex.
	virtual ArcsIteratorList& get_in_arc_its(VertexNameType vertex_name);

    /// get arc from given arc define(from --> to)
    virtual ArcType& get_arc(VertexNameType vertex_name_from, VertexNameType vertex_name_to);

    /// detect and get the i/o
    virtual std::vector<VertexNameType> get_graph_ins();
    virtual std::vector<VertexNameType> get_graph_outs();
    
    /// get vertex by vertex name
    virtual VertexType& operator[](VertexNameType vertex_name);

    virtual inline std::string to_string();

    /**
    * \brief iterator class of graph base
    */
    class iterator {
    public:
        iterator(typename std::unordered_map<VertexNameType, VertexType>::iterator& it):_vertex_it(it) {}
        iterator(typename std::unordered_map<VertexNameType, VertexType>::iterator&& it):_vertex_it(it) {}

        /// copy
        iterator& operator=(iterator& rhs) {
            _vertex_it = rhs._vertex_it;
            return *(this); 
        }
        /// Prefix ++
	    iterator& operator++() { _vertex_it++; return *(this); }
	    /// Postfix ++
	    iterator operator++(int) { 
            iterator tmp(*this); 
            operator++(); 
            return tmp; 
        }

        inline typename std::unordered_map<VertexNameType, VertexType>::iterator& origin() { return _vertex_it; }
	
	    bool operator==(const iterator& rhs) const { return _vertex_it==rhs._vertex_it; }
        bool operator!=(const iterator& rhs) const { return _vertex_it!=rhs._vertex_it; }
        std::pair<const VertexNameType, VertexType>& operator*() { return *_vertex_it; }
        std::pair<const VertexNameType, VertexType>* operator->() { return &(*_vertex_it); }

    private:
        typename std::unordered_map<VertexNameType, VertexType>::iterator _vertex_it;
    };

    /// algorithm for graph base.
    Algorithm<VertexNameType, VertexType, WeightType, ArcType> *Scanner{nullptr};
private:
    ///<  _vertices stand for set of vertices 
    std::unordered_map<VertexNameType, VertexType> _vertices;
    ///<  _arcs stand for set of arcs
    ArcsList _arcs;

    ///< _graph_out_arcs : map from vertex's name to it's out arc list
    std::unordered_map<VertexNameType, ArcsIteratorList> _graph_out_arcs;
    ///< _graph_in_arcs : map from vertex's name to it's in arc list
    std::unordered_map<VertexNameType, ArcsIteratorList> _graph_in_arcs;
};

} /* namespace graph */

} /* namespac anakin */

#include "graph_base.inl"

#endif /* ANAKIN_GRAPH_BASE_H */
