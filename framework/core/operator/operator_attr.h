/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

#ifndef ANAKIN_OPERATOR_ATTR_H
#define ANAKIN_OPERATOR_ATTR_H

#include <unordered_map>
#include "framework/core/parameter.h"
#include "framework/core/common_macros.h"
#include "framework/graph/graph.h"

namespace anakin {

/**
 *  \brief Argument class for Operator
 */
struct Argument {
    std::string name; ///< arg name.
    std::string type; ///< arg type_identifier.
    std::string doc;  ///< human readable doc for arg.
};

/**
 *  \brief OperatorAttr class. 
 */
struct OperatorAttr {
public:
    std::string name; ///< operator name. 
    std::string doc;  ///< operator doc.
    size_t num_in;    ///< io number of operator.
    size_t num_out;

    bool is_commutative{true}; ///< true default.
                               ///< Judge if the operation is commutative.

    ///< Operator paremeter map:  parameter name ---> arguments.
    std::unordered_map<std::string, Argument> Args_map;
};

/** 
 *  \brief Class OpAttrWarpper used for prcessing operation's parameter.
 */
class OpAttrWarpper {
public:
    OpAttrWarpper() {}

    /** 
     *  \brief Set origin op name (opAttr_ = op_name)
     */
    OpAttrWarpper& name(std::string op_name);

    /// set alias name of Operator.
    template<typename Ttype, DataType Dtype, Precision Ptype>
    OpAttrWarpper& __alias__(std::string);
    /// set description doc for target op.
    OpAttrWarpper& Doc( std::string );
    /// set and get number input and output.
    OpAttrWarpper& num_in(size_t);
    OpAttrWarpper& num_out(size_t);
    /// set commutative.
    OpAttrWarpper& commutative(bool);
    /** 
     *  \brief Set arguments of target op.
     *  e.g.
     *   .Args<int>("axis", " axis of concat .... ") or
     *   .Args<tensor>("weight", " the weight of full connect operator...")
     *   .Args<float>("scale", "...")
     *   .Args<bool>("bias_term", " if the full connect contain the bias parameter. ")
     * 
     *   param arg_name  : the name of argument.
     *   param arg_doc   : the doc for argument [default = ""].
     */
    template<typename T>
    OpAttrWarpper& Args(std::string arg_name, std::string arg_doc = "") {
        Argument arg;        
        arg.name = arg_name;
        arg.type = type_id<T>().type_info();
        arg.doc = arg_doc;
        if (!this->has_arg(arg_name)) {
            this->opAttr_.Args_map[arg_name] = arg;
        } else {
            LOG(ERROR) << " you have set the argument: " << arg_name << " , so it's igrored by anakin";
        }
        return *(this);
    }


    /** 
     *  \brief Get arg value from attributes info (AttrInfo) in node.
     */
    template<typename T>
    T& GetArg(std::string arg_name, graph::AttrInfo& info);

    /** 
     *  \brief Get name of operator.
     */
    std::string name() { return opAttr_.name; }
    /** 
     *  \brief Get doc description of op.
     */
    std::string doc() { return opAttr_.doc; }
    
    /// Get num input.
    size_t num_in() { return opAttr_.num_in; }
    /** 
     *  \brief Get num output.
     */
    size_t num_out() { return opAttr_.num_out; }
    /** 
     *  \brief Whether the operation is commutative.
     */
    bool commutative() { return opAttr_.is_commutative; }
    /** 
     *  \brief Judge if OperatorAttr the argument's name.
     */
    bool has_arg(std::string arg_name) { return opAttr_.Args_map.count(arg_name) > 0; }

    friend class OpAttrHelper;
private:
    OperatorAttr opAttr_;
};

} /* namespace anakin */


#endif
