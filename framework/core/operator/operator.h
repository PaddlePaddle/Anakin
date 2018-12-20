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
#include "framework/utils/parameter_fusion.h"
#include "framework/graph/graph_global_mem.h"

namespace anakin {

using namespace std::placeholders;

template<typename Ttype, Precision Ptype>
class OperatorHelper;

/** 
 *  \brief Operator class, it's a base class for other op defined by anakin.
 */
template<typename Ttype, Precision Ptype>
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
                             const std::vector<Tensor4dPtr<Ttype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
        LOG(ERROR) << "The Operator is basic";
    }

    /** 
     *  \brief Bind helper.
     */
    Operator<Ttype, Ptype>* operator>>(OperatorHelper<Ttype, Ptype>* helper) {
        _helper = helper;
        return this;
    }

    ///< Receive helper and attr from outside define.
    OperatorHelper<Ttype, Ptype>* _helper{nullptr};
};

/** 
 *  \brief Helper for operator, user defined helper should derived from it.
 */
template<typename Ttype, Precision Ptype>
class OperatorHelper {
public:
    OperatorHelper() {}
    virtual ~OperatorHelper() {}

    /** 
     *  \brief Parsing Parameter from graph, need to be overrided.
     */
    virtual Status InitParam() {
        DLOG(ERROR) << " Target ParserParam not overriden.";
        return Status::ANAKINFAIL();
    }

    /** 
     *  \brief Initial all the resource needed by operator and it's also need to be overrided.
     */
    virtual Status Init(OpContext<Ttype> &ctx,
                        const std::vector<Tensor4dPtr<Ttype> >& ins, 
                        std::vector<Tensor4dPtr<Ttype> >& outs){
        DLOG(ERROR) << " Target init not overriden.";
        return Status::ANAKINFAIL();
    }

    /** 
     *  \brief Infer the shape of output and input and it's also need to be overrided.
     */
    virtual Status InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                              std::vector<Tensor4dPtr<Ttype> >& outs){
        DLOG(ERROR) << " Target infershape not overriden.";
        return Status::ANAKINFAIL();
    }

    /** 
     *  \brief Bind parameter pack from graph.
     */
    void BindParam(graph::NodePtr& node_p) { 
        // Shareptr shallow copy
        // Note: We can also use deep copy by using node operator=, 
        //       but if change the node attrs through net class, 
        //       the base graph can't detect it.
        _node_p = node_p.get();
	}

    /** 
     *  \brief Get target attr by name.
     */
    template<typename T>
    T get_attr(std::string attr_name) { return _node_p->get_attr<T>(attr_name); }

    /**
     *  \brief Get target attr by name.
     */
    template<typename T>
    T get_attr(std::string attr_name,T default_data) { return _node_p->get_attr<T>(attr_name,default_data); }

    /**
     *  \brief find attr by name.
     */
    bool find_attr(std::string attr_name) { return _node_p->inspect_attr(attr_name); }

    /** 
     *  \brief set target attr
     */
    template<typename T> 
    void set_sttr(const std::string& attr_name, const T val) {
        _node_p->set_attr<T>(attr_name, val);
    }

	/**
	 *  \brief Judge if op access target attr
	 */
	inline bool check_attr(const std::string& attr_name) {
		return _node_p->inspect_attr(attr_name);
	}

    /**
     * \brief remove attr if it exists
     */
    inline void remove_attr(const std::string& attr_name) {
        _node_p->remove_attr(attr_name);
    }

private:
    ///< Pointer to graph node.
    graph::Node* _node_p;
};

/**
 *  \brief Call get_attr from derived class.
 */
#define GET_PARAMETER(type, name) \
    this->template get_attr<type>(#name)

/**
 *  \brief Call get_attr from derived class.
 */
#define GET_PARAMETER_WITH_DEFAULT(type, name,default_data) \
    this->template get_attr<type>(#name,default_data)

/**
 *  \brief Call get_attr from derived class.
 */
#define FIND_PARAMETER(name) this->find_attr(#name)

/**
 *  \brief Call set_sttr from derived class.
 */
#define SET_PARAMETER(name, val, type) \
    this->template set_sttr<type>(#name, val)

/**
 *  \brief Call check_attr from derived class.
 */
#define CHECK_PARAMETER(name) \
    this->check_attr(#name)

/**
 *  \brief Call remove_attr from derived class.
 */
#define REMOVE_PARAMETER(name) \
    this->remove_attr(#name)

/**
 *  \brief Operator creator.
 *  Typedef std::function<Operator*()> OperatorCreator.
 */ 
template<typename Ttype, Precision Ptype>
using OperatorCreator = std::function<Operator<Ttype, Ptype>*()>;

template<typename Ttype, Precision Ptype>
class OperatorFactory : public Factory<Operator<Ttype, Ptype>, OperatorCreator<Ttype, Ptype>> {
public:

    /** 
     *  \brief Get list of op name.
     */
    virtual inline std::vector<std::string>& get_list_op_name() {
        return this->get_list_name();
    }

    /**
     *  \brief judge if op factory has target op by it's name
     */
    virtual inline bool has_op(const std::string& op_name) {
        auto& supp_op_name_vec = get_list_op_name();
        auto ret_it = std::find(supp_op_name_vec.begin(), supp_op_name_vec.end(), op_name);
        if(ret_it != supp_op_name_vec.end()) {
            return true;
        }
        return false;
    }

    /** 
     *  \brief Create Operator object by op_name.
     *
     *   note: If Ptype is low precision( < FP32) and the low precise op doesn't exist, 
     *         this function will return nullptr.
     *         
     */
    virtual Operator<Ttype, Ptype>* operator[](const std::string op_name) {
        return Factory<Operator<Ttype, Ptype>, OperatorCreator<Ttype, Ptype>>::operator[](op_name);
    }

    /** 
     *  \brief Add another alias to the type_id.
     */
    virtual void add_alias(const std::string& ori_op_name, const std::string& op_name_alias) {
        this->__alias__(ori_op_name, op_name_alias);
    }
};

///< Typedef Singleton<OperatorFactory> OpFactory.
template<typename Ttype, Precision Ptype>
using OpFactory = Singleton<OperatorFactory<Ttype, Ptype> >;

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

#define ANAKIN_REGISTER_OP_HELPER(OpName, OpHelperName, TargetT, PrecisionT)                                             \
    static AK_ATTRIBUTE_UNUSED bool AK_MAKE_UNIQ_OPERATOR_NAME(OpName##_##OpHelperName##TargetT) =                       \
    OpFactory<TargetT, PrecisionT>::Global().Register(#OpName,                                                           \
                                  []() {                                                                                 \
                                        OpName<TargetT, PrecisionT>* tmpop = new OpName<TargetT, PrecisionT>();   		 \
                                        (*tmpop)>>(new OpHelperName<TargetT, PrecisionT>());                             \
                                        return tmpop;                                                                    \
                                  } )


} /* namespace anakin */


#endif
