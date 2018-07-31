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

#ifndef ANAKIN_ALGO_H
#define ANAKIN_ALGO_H 

#include <queue>
#include "utils/logger/logger.h"
#include "framework/core/base.h"
#include "framework/core/type_traits_extend.h"

namespace anakin {

namespace graph {

template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
class GraphBase;

/**
* \brief Algorithm class.
*/
template<typename VertexNameType, typename VertexType, typename WeightType, typename ArcType>
class Algorithm {
public:
    explicit Algorithm(GraphBase<VertexNameType, VertexType, WeightType, ArcType>* graph):_graph(graph) {}

    ~Algorithm() {}

   /** base search algorithm & function for edge
    * note: 
    *     the functor must return Status type
    */
	template<typename functor, typename ...ParamTypes>
    Algorithm<VertexNameType, VertexType, WeightType, ArcType>& DFS_Edge(functor& func, ParamTypes&& ...args);
	template<typename functor, typename ...ParamTypes>
    Algorithm<VertexNameType, VertexType, WeightType, ArcType>& BFS_Edge(functor& func, ParamTypes&& ...args);

	/// base search algorithm & function for node
	template<typename functor, typename ...ParamTypes>
    Algorithm<VertexNameType, VertexType, WeightType, ArcType>& DFS(functor& func, ParamTypes&& ...args);
    template<typename functor, typename ...ParamTypes>
    Algorithm<VertexNameType, VertexType, WeightType, ArcType>& BFS(functor& func, ParamTypes&& ...args);

    friend class GraphBase<VertexNameType, VertexType, WeightType, ArcType>;
private:
    template<typename functor, typename ...ParamTypes>
    Algorithm<VertexNameType, VertexType, WeightType, ArcType>& _DFS_Edge(Bool2Type<true>, functor& func, ParamTypes&& ...args);
	template<typename functor, typename ...ParamTypes>
    Algorithm<VertexNameType, VertexType, WeightType, ArcType>& _BFS_Edge(Bool2Type<true>, functor& func, ParamTypes&& ...args);

	//! base search algorithm & function for node
	template<typename functor, typename ...ParamTypes>
    Algorithm<VertexNameType, VertexType, WeightType, ArcType>& _DFS(Bool2Type<true>, functor& func, ParamTypes&& ...args);
    template<typename functor, typename ...ParamTypes>
    Algorithm<VertexNameType, VertexType, WeightType, ArcType>& _BFS(Bool2Type<true>, functor& func, ParamTypes&& ...args);

    template<typename functor, typename ...ParamTypes>
    Algorithm<VertexNameType, VertexType, WeightType, ArcType>& _DFS_Edge(Bool2Type<false>, functor& func, ParamTypes&& ...args);
	template<typename functor, typename ...ParamTypes>
    Algorithm<VertexNameType, VertexType, WeightType, ArcType>& _BFS_Edge(Bool2Type<false>, functor& func, ParamTypes&& ...args);

	//! base search algorithm & function for node
	template<typename functor, typename ...ParamTypes>
    Algorithm<VertexNameType, VertexType, WeightType, ArcType>& _DFS(Bool2Type<false>, functor& func, ParamTypes&& ...args);
    template<typename functor, typename ...ParamTypes>
    Algorithm<VertexNameType, VertexType, WeightType, ArcType>& _BFS(Bool2Type<false>, functor& func, ParamTypes&& ...args);

private:
    GraphBase<VertexNameType, VertexType, WeightType, ArcType>* _graph;
};

template<typename VertexNameType, 
         typename VertexType, 
         typename WeightType, 
         typename ArcType>
template<typename functor, typename ...ParamTypes>
Algorithm<VertexNameType, VertexType, WeightType, ArcType>& 
Algorithm<VertexNameType, VertexType, WeightType, ArcType>::DFS_Edge(functor& func, ParamTypes&& ...args) {
    return _DFS_Edge(Bool2Type<is_status_function<functor>::value>(), func, std::forward<ParamTypes>(args)...);
} 

template<typename VertexNameType, 
         typename VertexType, 
         typename WeightType, 
         typename ArcType>
template<typename functor, typename ...ParamTypes>
Algorithm<VertexNameType, VertexType, WeightType, ArcType>& Algorithm<VertexNameType, VertexType, WeightType, ArcType>::BFS_Edge(functor& func, ParamTypes&& ...args) {
    return _BFS_Edge(Bool2Type<is_status_function<functor>::value>(), func, std::forward<ParamTypes>(args)...);
} 

template<typename VertexNameType, 
         typename VertexType, 
         typename WeightType, 
         typename ArcType>
template<typename functor, typename ...ParamTypes>
Algorithm<VertexNameType, VertexType, WeightType, ArcType>& Algorithm<VertexNameType, VertexType, WeightType, ArcType>::DFS(functor& func, ParamTypes&& ...args) {
    return _DFS(Bool2Type<is_status_function<functor>::value>(), func, std::forward<ParamTypes>(args)...);
} 

template<typename VertexNameType, 
         typename VertexType, 
         typename WeightType, 
         typename ArcType>
template<typename functor, typename ...ParamTypes>
Algorithm<VertexNameType, VertexType, WeightType, ArcType>& Algorithm<VertexNameType, VertexType, WeightType, ArcType>::BFS(functor& func, ParamTypes&& ...args) {
    return _BFS(Bool2Type<is_status_function<functor>::value>(), func, std::forward<ParamTypes>(args)...);
} 

} /* namespace graph */

} /* namespace anakin */

#include "algorithm.inl"

#endif /* ANAKIN_ALGO_H */
