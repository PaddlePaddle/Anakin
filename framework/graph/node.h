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

#ifndef ANAKIN_NODE_H
#define ANAKIN_NODE_H 

#include "framework/graph/arc.h"
#include "framework/core/any.h"
#include "framework/core/base.h"
#include "framework/core/parameter.h"

namespace anakin {

/**
 * \brief Basic operation class.
 */
class OperatorBase {
public:
    OperatorBase() {}
    virtual ~OperatorBase() {}
};

namespace graph {

/**
* \brief struct AttrInfo of node
*/
struct AttrInfo {
public:
    AttrInfo() {
        parameter_p = 
            std::make_shared<std::unordered_map<std::string, ::anakin::any> >();
    }

    inline bool inspect(const std::string& attr_name) {
		auto it_end = parameter_p->end();
		auto it_find = parameter_p->find(attr_name);
		if(it_find != it_end) {
			return true;
		}
		return false;
	}

    template<typename T>
    T get(const std::string& attr_name) {
        auto it_end = parameter_p->end();
        auto it_find = parameter_p->find(attr_name);
        if(it_find == it_end) {
            LOG(FATAL) << "Target attr name(" << attr_name << ") not found.";
            return T();
        }
        return any_cast<T>((*parameter_p)[attr_name]);
    }

    template<typename T>
    T get(const std::string& attr_name,T default_data) {
        auto it_end = parameter_p->end();
        auto it_find = parameter_p->find(attr_name);
        if(it_find == it_end) {
            return default_data;
        }
        return any_cast<T>((*parameter_p)[attr_name]);
    }

    template<typename T>
    Status set(const std::string& attr_name, const T val) {
        (*parameter_p)[attr_name] = val;
        return Status::OK();
    }

    Status remove(const std::string& attr_name) {
        auto it_end = parameter_p->end();
        auto it_find = parameter_p->find(attr_name);
        if(it_find != it_end) {
            parameter_p->erase(attr_name);
            return Status::OK();
        } else {
            return Status::OK("target attr_name not included in attrs");
        }
    }

    inline void  MergeWithPattern(AttrInfo& operand, const std::string& pattern_name) {
        auto it_begin = operand.parameter_p->begin();
        auto it_end = operand.parameter_p->end();
        for(auto it = it_begin; it != it_end; ++it ) {
            // operand name has been changed!
            std::string new_name = pattern_name + "_" + it->first; 
            (*parameter_p)[new_name] = it->second; 
        }
    }

    std::unordered_map<std::string, ::anakin::any>::iterator begin() {
        return parameter_p->begin();
    }

    std::unordered_map<std::string, ::anakin::any>::iterator end() {
        return parameter_p->end();
    }

    /// shallow copy from other AttrInfo
    AttrInfo& operator=(const AttrInfo& other_attr_info) {
        this->parameter_p = other_attr_info.parameter_p;
        return *this;
    }
private:
    /// map : parameter ---> value
    std::shared_ptr<std::unordered_map<std::string, ::anakin::any> > parameter_p;
};

/**
* \brief struct Lane, where computation occur.
*/
struct Lane {
    Lane() {}
    Lane(int value): val(value) {}
    /// Lane operator int
    operator int() {
        return val;
    }
    ///< val stand for Lane value.
    int val;
};

/**
* \brief Edge class used for Global edge type
* public inherit Arc
*/
template<typename Ttype>
class Edge : public Arc<std::string, TensorSharedPtr<Ttype> > {
public:
    Edge():Arc<std::string, TensorSharedPtr<Ttype> >() {}
    Edge(const Edge<Ttype>& edge):Arc<std::string, TensorSharedPtr<Ttype> >(edge) {
        _shared = edge._shared; 
        _share_from = edge._share_from; 
        _current_lane = edge._current_lane; 
    }

    explicit Edge(std::string first, std::string second):Arc<std::string, TensorSharedPtr<Ttype> >(first, second) {}
    explicit Edge(std::string first, std::string second, TensorSharedPtr<Ttype> tensor_ptr)
        :Arc<std::string, TensorSharedPtr<Ttype> >(first, second, tensor_ptr) {}

    /// Get first node name of the edge.
    inline std::string& first() { return this->bottom(); }

    /// Get second node name of the edge.
    inline std::string& second() { return this->top(); }

    /// get data weigts of the edge.
    inline TensorSharedPtr<Ttype> data() { return this->weight(); }

    /// If edge's data is shared from the others.
    bool& shared() { return _shared; }

    std::string& share_from() { return _share_from; }
    
    /// lane which edge reside in
    Lane& lane() { return _current_lane; }

    /// print message
    inline std::string ToString() {
        std::string msg;
        if(_shared) {
            msg = this->name() + " shared_from: " + _share_from;
        } else {
            msg = this->name() + " not_shared";
        }
        return msg;
    }

    Edge& operator=(const Edge& edge) {
        _shared = edge._shared;
        _share_from = edge._share_from;
        _current_lane = edge._current_lane;
        Arc<std::string, TensorSharedPtr<Ttype> >::operator=(edge);
    }

private:
    /// the edge'data is shared from the others.
    bool _shared{false};
    ///< _share_from :the tensor this edge share from
    std::string _share_from;
    ///< _current_lane :Current lane the edge's data resides in. 
    Lane _current_lane;
};

/**
* \brief Node class used for Graph
*/
class Node {
public:
    Node() {}
    ~Node() {
		if(_Op) {
			delete _Op;
			_Op = nullptr;
		}
	}
    /// print message
    std::string DebugString();

    /// Get io
    size_t& get_num_in() { return _in_degree; }
    size_t& get_num_out() { return _out_degree; }

    /// Node name
    std::string& name() { return _name; }
    void set_name(std::string name) { _name = name; }

    /// Node operator
    OperatorBase* Op() { return _Op; }


    /// set node operator
    void set_op(OperatorBase* other) { _Op = other; }

    /// Node need wait
    bool& need_wait() { return _need_wait; }
    /// get op name
    std::string& get_op_name() { return _op_name; }

    /// Access to attributes.
    AttrInfo& attr() { return _attr; } 

	/// inspect if node attr have target attr name
	inline bool inspect_attr(const std::string& attr_name) {
	    return this->_attr.inspect(attr_name);	
	}

    /**
    * \brief Get target attr by name
    * \param attr_name stand for target_attr name
    * \return T the value of target attribute
    */
    template<typename T>
    T get_attr(const std::string& attr_name) {
        return this->_attr.get<T>(attr_name); 
    }
    /**
    * \brief Get target attr by name
    * \param attr_name stand for target_attr name
    * \return T the value of target attribute
    */
    template<typename T>
    T get_attr(const std::string& attr_name,T default_data) {
        return this->_attr.get<T>(attr_name,default_data);
    }
    /**
    * \brief Set target attr by name and value 
    * \param attr_name stand for target_attr name
    * \param val stand for attribute value
    * \return Status
    */
    template<typename T>
    Status set_attr(const std::string& attr_name, const T val) {
        std::unique_lock<std::mutex> lock(this->_mut);
        return this->_attr.set<T>(attr_name, val);  
    }

    /**
    * \brief remove target attr by name
    * \param attr_name stand for target_attr name
    * \return Status
    */
    Status remove_attr(const std::string& attr_name) {
        std::unique_lock<std::mutex> lock(this->_mut);
        return this->_attr.remove(attr_name); 
    }

    /// get lane
    Lane& lane() { return _current_lane; }

    /**
    * \brief merge for attr
    * \param operand
    * \param pattern_name 
    * \return Node 
    */
    inline Node& Merge(Node& operand, const std::string& pattern_name) { 
        std::unique_lock<std::mutex> lock(this->_mut);
        this->_attr.MergeWithPattern(operand.attr(), pattern_name);
        return *this;
    }

    /// copy construction [ shallow copy ]
    inline Node& operator=(Node& operand) {
        _name = operand._name;
        _current_lane = operand._current_lane;
        _Op = nullptr; // Assign the op pointer with operand's should be disabled, because it causes double free after binding the nodeptr by op itself.
        _op_name = operand._op_name;
        // shallow copy of attributes
        this->_attr = operand.attr();        
        // copy others
        _need_wait = operand._need_wait;
        _in_degree = operand._in_degree;
        _out_degree = operand._out_degree;
        return *this;
    }
    
    /// print message
    inline std::string ToString() { 
        std::ostringstream msg; 
        msg << _name << " : op(" << _op_name << ") lane(" << _current_lane << ") need_wait(" << _need_wait << ")";
        return msg.str();
    }

private:
    ///< _name stand for node name.
    std::string _name;
    ///< _current_lane stand for Current lane the node resides in.
    Lane _current_lane;
    ///< _Op stand for Operator in node.default bullptr
    OperatorBase* _Op{nullptr};
    ///< _op_name stand for op name
    std::string _op_name;
    ///<  _attr stand for AttrInfo of node (e.g. shape, scale, axis... )
    AttrInfo _attr;
    ///<_need_wait stand for need wait before the execution of node operator.defalut false
    bool _need_wait{false};

    ///<  _in_degree stand for number input degree
    size_t _in_degree;
    ///<  _out_degree stand for number output degree
    size_t _out_degree;

    std::mutex _mut; 
};


/// global node pointer type
//typedef std::shared_ptr<Node> NodePtr;
using NodePtr = std::shared_ptr<Node>;

} /* namespace graph */

} /* namespace anakin */

#endif /* ANAKIN_NODE_H */
