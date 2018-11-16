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

#ifndef ANAKIN_LLVM_FUSION_GRAPH_PATTERN_H
#define ANAKIN_LLVM_FUSION_GRAPH_PATTERN_H

#include <vector>
#include "framework/core/factory.h"
#include "framework/core/common_macros.h"
#include "framework/graph/llvm/virtual_graph.h"

namespace anakin {

namespace graph {

/**
*   \brief The fusion pattern in graph
*   GRAPH fusion pattern may has different match principle compared to that of IN_ORDER/IN_PARALLEL
*   searching methods of IN_PARALLEL and IN_ORDER may be very simple.
*/

enum Fusion: int {
    None = -1,
    IN_ORDER = 0,       ///< e.g. conv+relu
    IN_PARELLEL,        ///< e.g. batched fullconnected operator in parellel
    GRAPH,              ///< e.g. replace the target pattern sub-graph in vgraph by inception
};

struct FusionHash {
    size_t operator()(Fusion pattern) const { 
        return static_cast<size_t>(pattern);
    }
};

class Pattern : public VGraph {
public:
    Pattern() {} 
    ~Pattern() {}

    /**
     *  \brief Get the fusion operation's name.
     */
    inline std::string& fusion_op_name() { return _fusion_op_name; }

    /**
     *  \brief Get the fusion operation's type.
     */
    inline Fusion& type() { return _type; }

    inline int level() { return _level; }
    
    /**
     *  \brief Set _fusion_op_name and return the current object.
     *  \param string the value will be assigned to _fusion_op_name.
     *  \return Pattern& return the object after set its _fusion_op_name.
     */
    Pattern& name(std::string);

    /**
     *  \brief Creat a pattern which will set _pattern_create.
     *  \param std::function<void(VGraph*)> The value will be assigned to _pattern_create.
     *  \return Pattern& return the object after set its _pattern_create.
     */
    Pattern& CreatePattern(std::function<void(VGraph*)>);
    
    /**
     *  \brief Add a new node.
     *  \param string which is node name.
     *  \param string which is op name.
     *  \return Pattern& return the current object.
     */
    Pattern& AddOpNode(std::string, std::string);
    
    /**
     *  \brief Add a new arc.
     *  \param string which is one side of arc(edge).
     *  \param string which is the other side of arc.
     *  \return Pattern& return the current object.
     */
    Pattern& AddConnect(std::string, std::string);

    /**
     *  \brief Set _type and return the current object.
     *  \param fusion_type The value will be assigned to _type.
     *  \return Pattern& return the object.
     */
    Pattern& Type(Fusion fusion_type);

private:
    std::string _fusion_op_name;
    Fusion _type;
    ///< set fusion level for this pattern used to prioritize the fusion order (from high to low)
    int _level{0};     
    std::function<void(VGraph*)> _pattern_create;
};

class OpFusionPatternObjectRegister : public ObjectRegister<Pattern> {
public:

    /**
     *  \brief Get list op name of target fusion pattern.
     */
    std::vector<std::string> get_list_op_name_of(Fusion);

    /**
     *  \brief Get list op name of target fusion pattern with fusion order.
     */
    std::vector<std::string> get_list_op_name_in_fusion_order_of(Fusion);


    /**
     *  \brief Get list op name.
     */
    virtual std::vector<std::string>& get_list_op_name();
     
    /**
     *  \brief Get object pointer by op_name.
     */
    virtual Pattern* operator[](const std::string op_name);
  
    /**
     *  \brief Add another alias to the type_id.
     */
    virtual void add_alias(const std::string& ori_op_name, const std::string& op_name_alias);
};

typedef Singleton<OpFusionPatternObjectRegister> FusionOpRegister;

extern std::unordered_map<Fusion, std::function<int(VGraph*, Pattern*)>, FusionHash> FusionSniffer;


#define REGISTER_GRAPH_FUSION_PATTERN(OpName) \
        static AK_ATTRIBUTE_UNUSED Pattern& AK_MAKE_UNIQ_OPERATOR_NAME(OpName) = \
                   FusionOpRegister::Global().Register(#OpName).name(#OpName)

} // namespace graph

} // namespace anakin

#endif
