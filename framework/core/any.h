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

#ifndef ANAKIN_ANY_H
#define ANAKIN_ANY_H 

#include "framework/core/data_types.h"

namespace anakin {

/**  
 *  \brief Anakin any class which can accept any value type.
 *   Usage:
 *   any a = 2;    // now a == 2 (int)
 *   a = 3.14;     // now a == 3.14 (float)
 *   a = tensor(); // now a == tensor type
 *   a = shape();  // now a == shape type
 *   shape shape_test = any_cast<shape>(a);
 *   now, if you use: 
 *       tensor tensor_test = any_cast<tensor>(a); 
 *   the anakin will exit and alert some useful info for you.
 *  
 *   Note:
 *       in order to use our any in other share library , we use anakin version of type_id type_info,
 *       which is different from that of boost and STL.
 */
class any {
    struct ElementBase {
        virtual ~ElementBase(){}
        virtual ElementBase* clone() { return nullptr; }
        virtual const std::string type(){ return "error"; }
    };

    template<typename ValueType>
    struct Element: ElementBase {
        Element(ValueType& value):content_(value) {}
        Element(const ValueType& value):content_(value){}

        virtual ElementBase* clone() {
            return new Element(content_);
        }

        virtual const std::string type() {
            return anakin::type_id<ValueType>().type_info();
        }

        ValueType content_;
    };
    ElementBase* element_;  /// holder of any value.

public:
    any():element_(nullptr) {}

    template<typename ValueType>
    any(ValueType& value):element_(new Element<ValueType>(value)) {}
    template<typename ValueType>
    any(const ValueType& value):element_(new Element<ValueType>(value)) {}

    any(const any& other):element_(other.element_ ? other.element_->clone() : nullptr) {}
    any(any& other):element_(other.element_ ? other.element_->clone() : nullptr) {}

    /// move construction 
    any(any&& other):element_(other.element_) { 
        delete other.element_;
        other.element_ = nullptr; 
    }

    ~any() { delete element_; }

public: 
    /// swap element_ 
    any& swap(any& rhs) {
        std::swap(element_, rhs.element_);
        return *this;
    }
    any& swap(any&& rhs) {
        std::swap(element_, rhs.element_);
        return *this;
    }

    /// this = rhs
    template<typename ValueType>
    any & operator=(const ValueType& rhs) {
        this->swap(any(rhs));
        return *this;
    }

    any & operator=(const any& rhs) {
        any(rhs).swap(*this);
        return *this;
    }

    /// move assignement
    template<typename ValueType>
    any & operator=(ValueType&& rhs) {
        any(static_cast<ValueType&&>(rhs)).swap(*this);
        return *this;
    }

    any & operator=(any&& rhs) {
        rhs.swap(*this);
        any().swap(rhs);
        return *this;
    }

    bool empty() const {
        return !element_;
    }

    void clear() {
        any().swap(*this);
    }

    const std::string type() const {
        return element_ ? element_->type() : anakin::type_id<void>().type_info();
    }

private:
    template<typename ValueType>
    friend ValueType any_cast(any*);
    template<typename ValueType>
    friend ValueType any_cast(any&);
    template<typename ValueType>
    friend ValueType any_cast(const any*);
    template<typename ValueType>
    friend ValueType any_cast(const any&);
};
/// type conversion.
template<typename ValueType>
ValueType any_cast(any* operand) {
    if (operand->type() == anakin::type_id<ValueType>().type_info()) {
        return (static_cast<any::Element<ValueType>*> (operand->element_))->content_;
    }
    /// not a FATAL error
    if(operand->type() == "") {
        LOG(WARNING)<< "The type hold by any is None" 
                    << " , but you cast to type " << anakin::type_id<ValueType>().type_info()
                    << ", and you will get a empty vector.";
    } else {
        LOG(ERROR)<< "The type hold by any is " <<operand->type() 
                    << " , but you cast to type " << anakin::type_id<ValueType>().type_info();
    }
    return ValueType();
}

template<typename ValueType>
ValueType any_cast(any& operand) {
    if (operand.type() == anakin::type_id<ValueType>().type_info()) {
        return (static_cast<any::Element<ValueType>*> (operand.element_))->content_;
    }
    // not FATAL error
    if(operand.type() == "") {
        DLOG(WARNING)<< "The type hold by any is None"
                    << " , but you cast to type " << anakin::type_id<ValueType>().type_info()
                    << ", and you will get a empty vector.";
    } else {
        DLOG(ERROR)<< "The type hold by any is " <<operand.type() 
                    << " , but you cast to type " << anakin::type_id<ValueType>().type_info();
    }

    return ValueType();
}

template<typename ValueType>
ValueType any_cast(const any* operand) {
    return any_cast<ValueType>(const_cast<any *>(operand));
}

template<typename ValueType>
ValueType any_cast(const any& operand) {
    return any_cast<ValueType>(operand);
}

} /* namespace anakin */

#endif
