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

#ifndef ANAKIN_MODEL_PARSER_H
#define ANAKIN_MODEL_PARSER_H

#include "framework/graph/graph.h"
#include "framework/graph/node.h"
#include "framework/graph/algorithm.h"
#include <limits>

#define ProtoReadBytesLimit std::numeric_limits<int>::max() 

namespace anakin {

namespace parser {

//! parse data of external model_path file into graph.
template<typename Ttype, Precision Ptype>
Status load(graph::Graph<Ttype, Ptype>* graph, std::string& model_path);
template<typename Ttype, Precision Ptype>
Status load(graph::Graph<Ttype, Ptype>* graph, const char* model_path);

template<typename Ttype, Precision Ptype>
Status load(graph::Graph<Ttype, Ptype>* graph, const char* buffer, size_t len);

//! save graph to disk. use to save improved Graph.
template<typename Ttype, Precision Ptype>
Status save(graph::Graph<Ttype, Ptype>* graph, std::string& model_path);
template<typename Ttype, Precision Ptype>
Status save(graph::Graph<Ttype, Ptype>* graph, const char* model_path);

} /* parser */

} /* anakin */

#endif /* ANAKIN_GRAPH_H */
