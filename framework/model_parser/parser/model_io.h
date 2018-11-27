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

#ifndef ANAKIN_MODEL_IO_H
#define ANAKIN_MODEL_IO_H

#include<queue>
#include "framework/graph/graph.h"
#include "framework/graph/graph_global_mem.h"
#include "framework/graph/node.h"
#include "framework/graph/algorithm.h"
#include "framework/model_parser/parser/parser.h"
#include "framework/model_parser/proto/graph.pb.h"
#include "framework/model_parser/proto/node.pb.h"
#include "framework/model_parser/proto/operator.pb.h"
#include "framework/model_parser/proto/tensor.pb.h"

namespace anakin {

namespace parser {


template<typename Ttype, Precision Ptype>
class NodeIO {
public:
    NodeIO() {}
    ~NodeIO() {}

    size_t size() { return _que.size(); }
    bool empty() { return _que.empty(); }

    // read NodeProto
    NodeIO& operator>>(const NodeProto& node_proto);
    // read Node 
    NodeIO& operator>>(const graph::NodePtr& node_p);

    // output to Graph
    Status operator<<(graph::Graph<Ttype, Ptype>& graph);

    // output to GraphProto
    Status operator<<(GraphProto& graph);

    // get que node name in order
    std::vector<std::string>& get_node_name_in_order() { return _que_node_name_in_order; }

private:
    std::queue<graph::NodePtr> _que;
    std::vector<std::string> _que_node_name_in_order;
};

} /* parser */

} /* anakin */

#endif /* ANAKIN_GRAPH_H */
