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

#ifndef ANAKIN_OPERATOR_H
#define ANAKIN_OPERATOR_H

#include <functional>
#include "framework/core/operator/request.h"
#include "framework/core/operator/operator_attr.h"
#include "framework/core/factory.h"
#include "framework/core/parameter.h"
#include "framework/core/singleton.h"

namespace anakin {

template<typename Ttype, DataType Dtype, Precision Ptype>
class OperatorHelper;

/** 
 *  \brief Basic operation class.
 */
class OperatorBase {
public:
    OperatorBase() {}
    virtual ~OperatorBase() {}
};

/** 
 *  \brief Operator class, it's a base class for other op defined by anakin.
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Operator : public OperatorBase {
public:
    Operator() {}
    virtual ~Operator() {
		if(_helper) {
			delete _helper;
			_helper = nullptr;
		}
	}

    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "The Operator is basic";
    }

    /** 
     *  \brief Bind helper.
     */
    Operator<Ttype, Dtype, Ptype>* operator>>(OperatorHelper<Ttype, Dtype, Ptype>* helper) {
        _helper = helper;
        return this;
    }

    ///< Receive helper and attr from outside define.
    OperatorHelper<Ttype, Dtype, Ptype>* _helper{nullptr};
};

/** 
 *  \brief Helper for operator, user defined helper should derived from it.
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class OperatorHelper {
public:
    OperatorHelper() {}
    virtual ~OperatorHelper() {}

    /** 
     *  \brief Parsing Parameter from graph, need to be overrided.
     */
    virtual Status InitParam() {
        DLOG(ERROR) << " Target ParserParam not overriden.";
        return Status::FAIL();
    }

    /** 
     *  \brief Initial all the resource needed by operator and it's also need to be overrided.
     */
    virtual Status Init(OpContext<Ttype> &ctx,
                        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs){
        DLOG(ERROR) << " Target init not overriden.";
        return Status::FAIL();
    }

    /** 
     *  \brief Infer the shape of output and input and it's also need to be overrided.
     */
    virtual Status InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                              std::vector<Tensor4dPtr<Ttype, Dtype> >& outs){
        DLOG(ERROR) << " Target infershape not overriden.";
        return Status::FAIL();
    }

    /** 
     *  \brief Bind parameter pack from graph.
     */
    void BindParam(graph::NodePtr<Ttype, Dtype, Ptype>& node_p) { 
		_node_p = std::make_shared<graph::Node<Ttype, Dtype, Ptype>>(); 
		*_node_p = *node_p;
	}

    /** 
     *  \brief Get target attr by name.
     */
    template<typename T>
    T get_attr(std::string attr_name) { return _node_p-> template get_attr<T>(attr_name); }

	/**
	 *  \brief Judge if op access target attr
	 */
	inline bool check_attr(const std::string& attr_name) {
		return _node_p->inspect_attr(attr_name);
	}

private:
    ///< Pointer to graph node.
    graph::NodePtr<Ttype, Dtype, Ptype> _node_p;
};

/**
 *  \brief Call get_attr from derived class.
 */
#define GET_PARAMETER(type, name) \
    this->template get_attr<type>(#name)

/**
 *  \brief Operator creator.
 *  Typedef std::function<Operator*()> OperatorCreator.
 */ 
template<typename Ttype, DataType Dtype, Precision Ptype>
using OperatorCreator = std::function<Operator<Ttype, Dtype, Ptype>*()>;

template<typename Ttype, DataType Dtype, Precision Ptype>
class OperatorFactory : public Factory<Operator<Ttype, Dtype, Ptype>, OperatorCreator<Ttype, Dtype, Ptype>> {
public:

    /** 
     *  \brief Get list of op name.
     */
    virtual inline std::vector<std::string>& get_list_op_name() {
        return this->get_list_name();
    }

    /** 
     *  \brief Create Operator object by op_name.
     */
    virtual Operator<Ttype, Dtype, Ptype>* operator[](const std::string op_name) {
        return Factory<Operator<Ttype, Dtype, Ptype>, OperatorCreator<Ttype, Dtype, Ptype>>::operator[](op_name);
    }

    /** 
     *  \brief Add another alias to the type_id.
     */
    virtual void add_alias(const std::string& ori_op_name, const std::string& op_name_alias) {
        this->__alias__(ori_op_name, op_name_alias);
    }
};

///< Typedef Singleton<OperatorFactory> OpFactory.
template<typename Ttype, DataType Dtype, Precision Ptype>
using OpFactory = Singleton<OperatorFactory<Ttype, Dtype, Ptype> >;

/**
 *  \brief Operator objector register type.
 */ 
class OpAttrObjectRegister : public ObjectRegister<OpAttrWarpper> {
public:

    /** 
     *  \brief Get list of op name.
     */
    virtual std::vector<std::string>& get_list_op_name();

    /** 
     *  \brief Get object pointer by op_name.
     */
    virtual OpAttrWarpper* operator[](const std::string op_name);

    /** 
     *  \brief Add another alias to the type_id.
     */
    virtual void add_alias(const std::string& ori_op_name, const std::string& op_name_alias);
};

typedef Singleton<OpAttrObjectRegister> OpAttrRegister;

///  define anakin operator register.
///  usage:
///    // you should first impl the real fullconnect operation derived from Operator.
///    // you can impl other function or member in class fullconnect.
///    class fullconnect: public Operator {
///        // override operator() for inference.
///    };
///    class fullconnectHelper: public OperatorHelper {
///        // override Init
///        // override InferShape 
///    };
///
///    // First  : define the attribute of op.
///    // Second : register the Operator Helper and Operator for operator factory
///    //
///    ANAKIN_REGISTER_OP(fullconnect, fullconnectHelper)
///    .Doc(" full connect operator .")
///    .__alias__("fc")
///    .set_in(1)
///    .set_out(1)
///    .Args<int>("axis",  " the axis in input dim index. ")
///    .Args<bool>("bias_term", " whether include bias parameter.")
///    .Args<PTuple<float>>("weight", " the weight name.")
///    .Args<PTuple<float>>("bias", " the bias name.");       
#define ANAKIN_REGISTER_OP(OpName) \
    static AK_ATTRIBUTE_UNUSED OpAttrWarpper& AK_MAKE_UNIQ_OPERATOR_NAME(OpName) =  \
                   OpAttrRegister::Global().Register(#OpName).name(#OpName)

#define ANAKIN_REGISTER_OP_HELPER(OpName, OpHelperName, TargetT, DataT, PrecisionT)                                             \
    static AK_ATTRIBUTE_UNUSED bool AK_MAKE_UNIQ_OPERATOR_NAME(OpName##_##OpHelperName##TargetT##DataT) =                       \
    OpFactory<TargetT, DataT, PrecisionT>::Global().Register(#OpName,                                                           \
                                  []() {                                                                                        \
                                        OpName<TargetT, DataT, PrecisionT>* tmpop = new OpName<TargetT, DataT, PrecisionT>();   \
                                        (*tmpop)>>(new OpHelperName<TargetT, DataT, PrecisionT>());                             \
                                        return tmpop;                                                                           \
                                  } )


} /* namespace anakin */


#endif
