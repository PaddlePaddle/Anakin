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

#ifndef ANAKIN_LLVM_SCHEDULER_BASE_H
#define ANAKIN_LLVM_SCHEDULER_BASE_H

#include <queue>
#include <list>
#include <functional>
#include <unordered_map>
#include "utils/logger/logger.h"

namespace anakin {

namespace graph {

/**
*  \brief TagStatus structure used for Tag
*/
struct TagStatus{
    operator bool(){
        return status;
    }
    /// toggle button
    inline void toggle() {
        if(status) { 
            this->off(); 
        } else {
            this->on();
        }
    }
    /// set on means tag's resource is acquirable
    inline void on(){ status = true; }
    /// set on means tag's resource is un-acquirable
    inline void off(){ status = false; } 
    ///< status default false
    bool status{false};
};

/** 
 * \brief tag write and read res class
 *  tag can be used for any resource
 *  \param ResType stand for targe resource type (e.g. io )
 *  \param HASH , customed hash functor
 */
template<typename ResType, typename HASH = std::hash<ResType>>
class Tag {
public:
    Tag(){}
    ~Tag(){}

    /**
    *  \brief accept target resource
    *  \param arg stand for targe resource type 
    *  \return void
    */
    inline void accept(ResType& arg) {
        TagStatus t_status;
        if(!has(arg))  _status[arg] = t_status;
    }

     /**
    *  \brief accept target resource
    *  \param res_list stand for the list of targe resource type
    *  \return void
    */
    inline void accept(std::vector<ResType>& res_list) {
        for(auto& res : res_list) {
            accept(res);        
        }
    }

    /**
    *  \brief check whether Tag have target resource 
    *  \param arg  stand for the targe resource type
    *  \return bool the value of _status.count(arg) > 0
    */
    inline bool has(ResType& arg) { return _status.count(arg) > 0; }
    /**
    *  \brief lock the resource
    *  \param res_list stand for the list of targe resource type
    *  \return void
    */
    inline void lock(std::vector<ResType>& res_list) {
        for(auto& res : res_list) {
            if(has(res)) _status[res].off();
        }
    }
    /**
    *  \brief  free the resource
    *  \param res_list stand for the list of targe resource type
    *  \return void
    */
    inline void free(std::vector<ResType>& res_list) {
        for(auto& res : res_list) {
            if(has(res)) _status[res].on();
        }
    }
    inline void lock(ResType& arg_0) { if(has(arg_0)) _status[arg_0].off();}
    inline void free(ResType& arg_0) { if(has(arg_0)) _status[arg_0].on();}
    
    /**
    *  \brief whether the resource is accessible
    *  \param res_list stand for the list of targe resource type
    *  \return bool the value of (res_list.size() == 0 || (has(res) && _status[res]))
    */
    inline bool check_access(std::vector<ResType>& res_list) {
        if(res_list.size() == 0) {
            return true;
        }
        for(auto& res : res_list) {
            if(has(res) && _status[res]) { 
                continue; 
            } else {
                return false;
            }
        }
        return true;
    }

    inline bool check_access(ResType arg_0){
        return _status[arg_0];
    }
private:
    std::unordered_map<ResType, TagStatus, HASH> _status;
};

/**
 *  \brief Dependency schedule base class for analysing the execution of operator in VGraph
 *  \param param ResType stand for targe resource type (e.g. io )
 *  \param OpType stand for operator type which can be invoke.
 *  \param HASH , customed hash functor
*/
template<typename ResType, 
         typename OpType, 
         typename HASH = std::hash<ResType>>
class ScheduleBase {
public:
    ScheduleBase() {}
    virtual ~ScheduleBase() {}
    /**
    *  \brief register the read and write resource.
    *  \param resource stand for targe resource type
    *  \return void
    */
    inline void RegResource(ResType& resource) {
        _rw_resources.accept(resource);
    }

    /**
    *  \brief lock the resource
    *  \param res_list stand for the list of targe resource type
    *  \return void
    */
    inline void lock(std::vector<ResType>& res_list) { _rw_resources.lock(res_list); }
    /**
    *  \brief free the resource
    *  \param res_list stand for the list of targe resource type
    *  \return void
    */
    inline void free(std::vector<ResType>& res_list) { _rw_resources.free(res_list); }
    inline void lock(ResType arg_0) { _rw_resources.lock(arg_0); }
    inline void free(ResType arg_0) { _rw_resources.free(arg_0);}
    /**
    *  \brief whether the resource is accessible
    *  \param res_list stand for the list of targe resource type 
    *  \return bool the value of  check_access(res_list)
    */
    inline bool check_access(std::vector<ResType>& res_list) {
        return _rw_resources.check_access(res_list);
    }
    inline bool check_access(ResType arg_0){
        return _rw_resources.check_access(arg_0);
    }

    /**
    *  \brief decide if the target op is callable
    *  This is a virtual function and can't be succeed and implemented
    *  \return bool the value is false
    */
    virtual bool callable(OpType&) = 0;

    /**
    *  \brief operations of op queue
    *  push operation
    *  \param op stand for operation type
    *  \return void
    */
    inline void exe_push(OpType& op) {
        _exec_que.push(op);
    }
	
    /**
    *  \brief operations of op queue
    *  push_back operation
    *  \param op stand for operation type
    *  \return void
    */
    inline void wait_push(OpType& op) {
        _wait_que.push_back(op);
    }

    /**
    *  \brief launch operator and push op to execution queue
    *  This is a virtual function and can't be succeed and implemented
    *  \return void 
    */
    virtual void launch(OpType&) = 0;

	/**
     *  \brief judge if target op have been launched
	 *
     *  \param op stand for operation type
     *  \return bool
     */
	inline bool have_launched(OpType& op) {
		for(auto it = _wait_que.begin(); it != _wait_que.end();) {
			if(*it == op) {
				return false;
			}
			++it;
		}
		return true;
	}


    /**
    *  \brief get exec queue.
    *  queue operation such as push,push_back,pop
    *  \return std::vector<OpType>& the address of op
    */
    std::vector<OpType>& get_exec_que() {
        while(!_exec_que.empty()) {
            auto& op = _exec_que.front();
            _exec_ops.push_back(op);
            _exec_que.pop();
        }
        return _exec_ops;
    }

protected:
    ///< _exec_que stand for execution queue
    std::queue<OpType> _exec_que;
    ///< _wait_que stand for tmemporary wait queue
    std::list<OpType> _wait_que;
    ///< _rw_resources stand for read and write resources.
    Tag<ResType, HASH> _rw_resources;
    ///< _exec_ops stand for exec order of ops
    std::vector<OpType> _exec_ops;
};

} /* namespace graph */

} /* namespace anakin */

#endif /* ANAKIN_ALGO_H */
