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

#ifndef ANAKIN_GRAPH_GLOBAL_MEM_H
#define ANAKIN_GRAPH_GLOBAL_MEM_H 

#include <vector>
#include "framework/core/singleton.h"
#include "framework/core/parameter.h"
#include "utils/logger/logger.h"
#include <mutex>
#include "anakin_config.h"
#ifdef USE_SGX
#include <support/sgx/sgx_mutex>
#endif

namespace anakin {

using namespace saber;

/**
* \brief global resource level 
*/
enum Level {
    Level_0 = 0,
    Level_1,
    Level_2,
    Level_3,
    Level_4,
    Level_5 
};

namespace graph {

/**
* \brief global resource level stage
*/
template<Level L>
struct LevelStage {
    std::mutex _mut;
    bool accessible = true;
};

/**
* \brief global resource multi level stage and restraint
*/
template<Level ...levels>
struct GlobalResRestrain : public LevelStage<levels>... {
    GlobalResRestrain() {} 
    GlobalResRestrain<levels...>& operator=(const GlobalResRestrain<levels...>& other){ 
        return *this; 
    }

    template<Level L>
    std::mutex& get_mut() {
        return LevelStage<L>::_mut;
    }
    template<Level L>
    bool& check_access() {
        return LevelStage<L>::accessible;
    }
    template<Level L>
    void use() {
        LevelStage<L>::accessible = false;
    }
};

/**
* \brief GraphGlobalMemBase class
*/
template<typename Ttype>
class GraphGlobalMemBase {
public:
    GraphGlobalMemBase() {}
    ~GraphGlobalMemBase() {}

    /// create Block memory
    template<DataType Dtype>
    PBlock<Ttype>* new_block(saber::Shape& shape) EXCLUSIVE_LOCKS_REQUIRED(_mut) {
        std::unique_lock<std::mutex> lock(this->_mut); 
        PBlock<Ttype>* block_p = new PBlock<Ttype>(shape, Dtype);
        // register new block_p for resource guard
        _res_guard[block_p->h_tensor().data()] = LevelList();
        _push_mem_pool(block_p, DataTypeWarpper<Dtype>()); 
        return block_p;
    }

    /// apply arbitrary function to two memory block
    /// note: that args may contain target PBlock pointer
    ///       so we need to set mutex for mem management
    template<Level L, typename functor, typename ...ParamTypes>
    void apply(functor func, PBlock<Ttype> tensor_1 , PBlock<Ttype> tensor_2, ParamTypes ...args) {
        std::unique_lock<std::mutex> lock(this->_mut);
        void* key_1 = tensor_1.h_tensor().data();
        void* key_2 = tensor_1.h_tensor().data();
        if(_res_guard[key_1].template check_access<L>()) {
            std::unique_lock<std::mutex> lock(_res_guard[key_1].template get_mut<L>());
            _res_guard[key_1].template use<L>();
            _res_guard[key_2].template use<L>();
            func(tensor_1, tensor_2, std::forward<ParamTypes>(args)...);
            void* new_key_1 = tensor_1.h_tensor().data();
            void* new_key_2 = tensor_2.h_tensor().data();
            if(new_key_1 != key_1) {
                _res_guard[new_key_1] = _res_guard[key_1];
                if(_res_guard.erase(key_1) != 1) { // delete old key-vale
                    LOG(FATAL) << "target key_1(" << key_1 << ") doesn't exist.";
                }
            }
            if(new_key_2 != key_2) {
                _res_guard[new_key_2] = _res_guard[key_2];
                if(_res_guard.erase(key_2) != 1) { // delete old key-vale
                    LOG(FATAL) << "target key_2(" << key_2 << ") doesn't exist.";
                }
            }
        }
    }
    /// apply arbitrary function to one memory block
    /// note: that args may contain target PBlock pointer
    ///       so we need to set mutex for mem management
    template<Level L, typename functor, typename ...ParamTypes>
    void apply(functor func, PBlock<Ttype> tensor , ParamTypes ...args) {
        std::unique_lock<std::mutex> lock(this->_mut);
        void* key = tensor.h_tensor().data();
        if(_res_guard[key].template check_access<L>()) {
            std::unique_lock<std::mutex> lock(_res_guard[key].template get_mut<L>());
            _res_guard[key].template use<L>();
            func(tensor, std::forward<ParamTypes>(args)...);
            void* new_key = tensor.data();
            if(new_key != key) {
                _res_guard[new_key] = _res_guard[key];
                if(_res_guard.erase(key) != 1) { // delete old key-vale
                    LOG(FATAL) << "target key(" << key << ") doesn't exist.";
                }
            }
        }
    }

    /// apply arbitrary function to one memory tensor
    /// note: that args may contain target PBlock pointer
    ///       so we need to set mutex for mem management
    template<Level L, typename functor, typename ...ParamTypes>
    void apply(functor func, Tensor4d<Ttype>& tensor , ParamTypes ...args) {
        std::unique_lock<std::mutex> lock(this->_mut);
        void* key = tensor.data();
        if(_res_guard[key].template check_access<L>()) {
            std::unique_lock<std::mutex> lock(_res_guard[key].template get_mut<L>());
            _res_guard[key].template use<L>();
            func(tensor, std::forward<ParamTypes>(args)...);
            void* new_key = tensor.data(); // check if tensor data has changed 
            if(key != new_key) {
                _res_guard[new_key] = _res_guard[key];
                if(_res_guard.erase(key) != 1) { // delete old key-vale
                    LOG(FATAL) << "target key(" << key << ") doesn't exist.";
                }
            }
        }
        if(key == nullptr) {
            func(tensor, std::forward<ParamTypes>(args)...);
        }
    }
template<Level L, typename functor, typename ...ParamTypes>
void apply(functor func, Tensor4d<Ttype>& tensor1 , Tensor4d<Ttype>& tensor2, ParamTypes ...args) {
        std::unique_lock<std::mutex> lock(this->_mut);
        void* key1 = tensor1.data();
        void* key2 = tensor2.data();
        if (_res_guard[key1].template check_access<L>()) {
            std::unique_lock<std::mutex> lock(_res_guard[key1].template get_mut<L>());
            _res_guard[key1].template use<L>();
            _res_guard[key2].template use<L>();
            func(tensor1, tensor2, std::forward<ParamTypes>(args)...);
            void* new_key1 = tensor1.data(); // check if tensor data has changed
            void* new_key2 = tensor2.data(); // check if tensor data has changed
            if (key1 != new_key1) {
                _res_guard[new_key1] = _res_guard[key1];
                if (_res_guard.erase(key1) != 1) { // delete old key-vale
                    LOG(FATAL) << "target key(" << key1 << ") doesn't exist.";
                }
            }
            if (key2 != new_key2) {
                _res_guard[new_key2] = _res_guard[key2];
                if (_res_guard.erase(key2) != 1) { // delete old key-vale
                    LOG(FATAL) << "target key(" << key2 << ") doesn't exist.";
                }
            }
        }
        if (key1 == nullptr && key2 == nullptr) {
            func(tensor1, tensor2, std::forward<ParamTypes>(args)...);
        }
}

    /// get sum size in m-btyes
    size_t get_sum_mbyte() EXCLUSIVE_LOCKS_REQUIRED(_mut) {
        std::unique_lock<std::mutex> lock(this->_mut); 
        size_t sum = 0;
        for (auto block_p : _int8_mem_pool) {
            sum += block_p->count();
        }
        for (auto block_p : _fp16_mem_pool) {
            sum += block_p->count()*2;
        }
        for (auto block_p : _fp32_mem_pool) {
            sum += block_p->count()*4;
        }
        return sum / 1e6;
    }

    /// clean all
    void clean_all() EXCLUSIVE_LOCKS_REQUIRED(_mut) {
        std::unique_lock<std::mutex> lock(this->_mut);
        for(auto block_p : _int8_mem_pool) {
            delete block_p;
        }
        _int8_mem_pool.clear();
        for(auto block_p : _fp16_mem_pool) {
            delete block_p;
        }
        _fp16_mem_pool.clear();
        for(auto block_p : _fp32_mem_pool) {
            delete block_p;
        }
        _fp32_mem_pool.clear();
    }

    /// get pool size
    template<DataType Dtype>
    size_t get_pool_size() { return _get_pool_size(DataTypeWarpper<Dtype>()); }

private:
    /// push int8_mem operaiton 
    void _push_mem_pool(PBlock<Ttype>* block_p, DataTypeWarpper<AK_INT8>) {
        _int8_mem_pool.push_back(block_p);
    }
    /// push fp16_mem operaiton 
    void _push_mem_pool(PBlock<Ttype>* block_p, DataTypeWarpper<AK_HALF>) {
        _fp16_mem_pool.push_back(block_p);
    }
    /// push fp32_mem operaiton 
    void _push_mem_pool(PBlock<Ttype>* block_p, DataTypeWarpper<AK_FLOAT>) {
        _fp32_mem_pool.push_back(block_p);
    }

    /// get int8_mem pool size
    size_t _get_pool_size(DataTypeWarpper<AK_INT8>) {
        return _int8_mem_pool.size();
    }
    /// get fp16_mem pool size
    size_t _get_pool_size(DataTypeWarpper<AK_HALF>) {
        return _fp16_mem_pool.size();
    }
    /// get fp32_mem pool size
    size_t _get_pool_size(DataTypeWarpper<AK_FLOAT>) {
        return _fp32_mem_pool.size();
    }

private:
    typedef GlobalResRestrain<Level_0, Level_1, Level_2, Level_3> LevelList;
    std::unordered_map<void*, LevelList> _res_guard;
    ///< _int8_mem_pool stand for int8 type memory
    std::vector<PBlock<Ttype>* > _int8_mem_pool GUARDED_BY(_mut);
    ///< _fp16_mem_pool stand for fp16 type memory
    std::vector<PBlock<Ttype>* > _fp16_mem_pool GUARDED_BY(_mut);
    ///< _fp32_mem_pool stand for fp32 type memory
    std::vector<PBlock<Ttype>* > _fp32_mem_pool GUARDED_BY(_mut);
    ///< _mut
    std::mutex _mut;
};

/// graph memory pool for graph weights and large parameter
template<typename Ttype>
using GraphGlobalMem = Singleton<GraphGlobalMemBase<Ttype>>;

/** 
 * \brief InFO enum
 * using number to stand for memory and other info of anakin 
 */
enum INFO{
    TEMP_MEM = 0,   ///< 0 stand for TEMP_MEM
    ORI_TEMP_MEM,   ///< 1 stand for ORI_TEMP_MEM
    MODEL_MEM,      ///< 2 stand for MODEL_MEM
    SYSTEM_MEM,     ///< 3 stand for SYSTEM_MEM
    IS_OPTIMIZED    ///< 4 stand for IS_OPTIMIZED
};

template<INFO INFO_T>
struct Decide{ 
    typedef int type;
};

template<>
struct Decide<IS_OPTIMIZED> {
    typedef bool type;
};
/**
* \brief Statistics struct
* used for memory information set and get
*/
struct Statistics {
    template<INFO INFO_T>
    void set_info(typename Decide<INFO_T>::type value) {
        _set_info(value, Info_to_type<INFO_T>());
    }
    
    template<INFO INFO_T>
    typename Decide<INFO_T>::type get_info() {
        return _get_info(Info_to_type<INFO_T>());
    }
private:
    template<INFO INFO_T>
    struct Info_to_type {};

    inline void _set_info(int mem_in_mbytes, Info_to_type<TEMP_MEM>) {
        temp_mem_used = mem_in_mbytes;
    }
    inline void _set_info(int mem_in_mbytes, Info_to_type<ORI_TEMP_MEM>) {
        original_temp_mem_used = mem_in_mbytes;
    }
    inline void _set_info(int mem_in_mbytes, Info_to_type<MODEL_MEM>) {
        model_mem_used = mem_in_mbytes;
    }
    inline void _set_info(int mem_in_mbytes, Info_to_type<SYSTEM_MEM>) {
        system_mem_used = mem_in_mbytes;
    }
    inline void _set_info(bool whether_optimized, Info_to_type<IS_OPTIMIZED>) {
        is_optimized = whether_optimized;
    }

    inline typename Decide<TEMP_MEM>::type _get_info(Info_to_type<TEMP_MEM>) {
        return temp_mem_used;
    }
    inline typename Decide<ORI_TEMP_MEM>::type _get_info(Info_to_type<ORI_TEMP_MEM>) {
        return original_temp_mem_used;
    }
    inline typename Decide<MODEL_MEM>::type _get_info(Info_to_type<MODEL_MEM>) {
        return model_mem_used;
    }
    inline typename Decide<SYSTEM_MEM>::type _get_info(Info_to_type<SYSTEM_MEM>) {
        return system_mem_used;
    }
    inline typename Decide<IS_OPTIMIZED>::type _get_info(Info_to_type<IS_OPTIMIZED>) {
        return is_optimized;
    }

private:
    ///< temp_mem_used : temp memory used by anakin edge [MB].default 0
    int temp_mem_used{0};
    ///< original_temp_mem_used : temp memory used by old version [MB].default 0
    int original_temp_mem_used{0};
    ///< system_mem_used : system mem used by nvidia / amd GPU system resource [MB].default 0
    int system_mem_used{0};
    ///<  model_mem_used : mem used by model.default 0
    int model_mem_used{0};

    ///< is_optimized stand for whether optimized flag.default false
    bool is_optimized{false};
};

} /* namespace graph */

} /* namespac anakin */

#endif
