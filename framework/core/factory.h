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

#ifndef ANAKIN_FACTORY_H
#define ANAKIN_FACTORY_H 

#include <mutex>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include "framework/core/thread_safe_macros.h"
#include "framework/core/singleton.h"
#include "utils/logger/logger.h"

namespace anakin {

template<typename PolicyType, 
         typename TypeIdentifier, 
         typename PolicyCreator, 
         typename PolicyTypeHash = std::hash<TypeIdentifier>>
class FactoryBase {
public:
    PolicyType* Create(const TypeIdentifier& type_id){
        if (_container.count(type_id) == 0) {
            LOG(FATAL) << type_id << " has not been registered! ";
        }
        //LOG(INFO) << "create " << type_id << " fuction " << &_container.at(type_id);
        //auto ptr = _container.at(type_id)();
        //return ptr;
        return (_container.at(type_id))();
    }
    void __ALIAS__(const TypeIdentifier& ori_type_id, const TypeIdentifier& type_id) {
        if (_container.count(ori_type_id) == 0) {
            LOG(FATAL) << type_id << " 's original "<< ori_type_id << " has not been registered! ";
        } else {
            _container[type_id] = _container[ori_type_id];
            _type_id_list.push_back(type_id);
        }
    }
    std::vector<TypeIdentifier>& GetTypeIdentifierList() {
        return _type_id_list;
    }
    bool Register(TypeIdentifier type_id, PolicyCreator creator) 
                                         EXCLUSIVE_LOCKS_REQUIRED(container_mutex_) {
        std::lock_guard<std::mutex> guard(container_mutex_);
        //LOG(ERROR) << "register " << type_id;
        if (_container.count(type_id) == 0) {
            _type_id_list.push_back(type_id);
            _container[type_id] = creator;
        }
        return true;
    }
    void UnRegister(const TypeIdentifier& type_id) 
                                         EXCLUSIVE_LOCKS_REQUIRED(container_mutex_) {
        std::lock_guard<std::mutex> guard(container_mutex_);
        _type_id_list.erase(std::remove(_type_id_list.begin(), _type_id_list.end(), type_id), _type_id_list.end());
        _container.erase(type_id) = 1;
    }
    std::mutex& GetMutex() LOCK_RETURNED(container_mutex_) { return container_mutex_; }
private:
    std::mutex container_mutex_;
    std::vector<TypeIdentifier> _type_id_list;
    typedef std::unordered_map<TypeIdentifier, PolicyCreator, PolicyTypeHash> ContainerType;
    ContainerType _container GUARDED_BY(container_mutex_);
};

template<typename PolicyType,
         typename FactoryPolicyCreator,
         ReleaseAtExit release_func = ReleaseAtExit()>
class Factory:
    public FactoryBase<PolicyType, std::string, FactoryPolicyCreator> {
public:
    /// policy type.
    typedef PolicyType type;
    /// Get list of type name.
    virtual std::vector<std::string>& get_list_name() {
        return this->GetTypeIdentifierList();
    }
    /// Get object pointer by type_id.
    virtual PolicyType* operator[](const std::string& type_id) {
        return this->Create(type_id);
    }
    /// Add another alias to the type_id.
    virtual void __alias__(const std::string& ori_name, const std::string& alias_name) {
        this->__ALIAS__(ori_name, alias_name); 
    }
};

/** 
 *  \brief Object register base class.
 */
template<typename PolicyType, 
         typename TypeIdentifier, 
         typename PolicyTypeHash = std::hash<TypeIdentifier>>
class ObjectRegisterBase {
public:
    PolicyType* Get(const TypeIdentifier& type_id) {
        if (_container.count(type_id) == 0) {
            LOG(FATAL) << type_id << " has not been registered! ";
        }
        return _container.at(type_id);
    }
    void __ALIAS__(const TypeIdentifier& ori_type_id, const TypeIdentifier& type_id) {
        if (_container.count(ori_type_id) == 0) {
            LOG(FATAL) << type_id << " 's original "<< ori_type_id << " has not been registered! ";
        } else {
            _container[type_id] = _container[ori_type_id];
        }
    }
    std::vector<TypeIdentifier>& GetTypeIdentifierList() {
        return _type_id_list;
    }
    PolicyType& Register(TypeIdentifier type_id) EXCLUSIVE_LOCKS_REQUIRED(_container_mutex) { 
        std::lock_guard<std::mutex> guard(_container_mutex); 
        //CHECK_EQ(_container.count(type_id), 0) << type_id << " has been registered! ";
        if (_container.count(type_id) == 0) {
            PolicyType* object= new PolicyType();
            _container[type_id] = object;
            _type_id_list.push_back(type_id);
            return *object;
        } else {
            PolicyType* object = _container[type_id];
            return *object;
        }

    }
    void UnRegister(const TypeIdentifier& type_id) EXCLUSIVE_LOCKS_REQUIRED(_container_mutex) {
        std::lock_guard<std::mutex> guard(_container_mutex);
        _type_id_list.erase(std::remove(_type_id_list.begin(), _type_id_list.end(), type_id), _type_id_list.end());
        _container.erase(type_id) = 1;
    }
    std::mutex& GetMutex() LOCK_RETURNED(_container_mutex) { return _container_mutex; }
private:
    std::mutex _container_mutex;
    std::vector<TypeIdentifier> _type_id_list;
    typedef std::unordered_map<TypeIdentifier, PolicyType*, PolicyTypeHash> ContainerType;
    ContainerType _container GUARDED_BY(_container_mutex);
};

/** 
 *  \brief Object register class.
 *
 */
template<typename PolicyType>
class ObjectRegister : public ObjectRegisterBase<PolicyType, std::string> {
public:
    /// get list of type name
    virtual std::vector<std::string>& get_list_name() {
        return this->GetTypeIdentifierList();
    }
    /// get object pointer by type_id
    virtual PolicyType* operator[](const std::string& type_id) {
        return this->Get(type_id);
    }
    /// Add another alias to the type_id
    virtual void __alias__(const std::string& ori_name, const std::string& alias_name) {
        this->__ALIAS__(ori_name, alias_name); 
    }
};

} /* namespace anakin */

#endif
