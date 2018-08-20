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
#include <mutex>
#include "framework/core/singleton.h"
#include "framework/core/parameter.h"
#include "utils/logger/logger.h"

namespace anakin {

using namespace saber;

namespace graph {

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
    PBlock<typename DataTypeWarpper<Dtype>::type, Ttype>* new_block(saber::Shape& shape) EXCLUSIVE_LOCKS_REQUIRED(_mut) {
        std::unique_lock<std::mutex> lock(this->_mut); 
        PBlock<typename DataTypeWarpper<Dtype>::type, Ttype>* block_p = new PBlock<typename DataTypeWarpper<Dtype>::type, Ttype>(shape);
        _push_mem_pool(block_p, DataTypeWarpper<Dtype>()); 
        return block_p;
    }

    /// get sum size in m-btyes
    float get_sum_mbyte() EXCLUSIVE_LOCKS_REQUIRED(_mut) {
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
    void _push_mem_pool(PBlock<int8_t, Ttype>* block_p, DataTypeWarpper<AK_INT8>) {
        _int8_mem_pool.push_back(block_p);
    }
    /// push fp16_mem operaiton 
    void _push_mem_pool(PBlock<unsigned short, Ttype>* block_p, DataTypeWarpper<AK_HALF>) {
        _fp16_mem_pool.push_back(block_p);
    }
    /// push fp32_mem operaiton 
    void _push_mem_pool(PBlock<float, Ttype>* block_p, DataTypeWarpper<AK_FLOAT>) {
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
    ///< _int8_mem_pool stand for int8 type memory
    std::vector<PBlock<typename DataTypeWarpper<AK_INT8>::type, Ttype>* > _int8_mem_pool GUARDED_BY(_mut);
    ///< _fp16_mem_pool stand for fp16 type memory
    std::vector<PBlock<typename DataTypeWarpper<AK_HALF>::type, Ttype>* > _fp16_mem_pool GUARDED_BY(_mut);
    ///< _fp32_mem_pool stand for fp32 type memory
    std::vector<PBlock<typename DataTypeWarpper<AK_FLOAT>::type, Ttype>* > _fp32_mem_pool GUARDED_BY(_mut);
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
    typedef float type;
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

    inline void _set_info(float mem_in_mbytes, Info_to_type<TEMP_MEM>) {
        temp_mem_used = mem_in_mbytes;
    }
    inline void _set_info(float mem_in_mbytes, Info_to_type<ORI_TEMP_MEM>) {
        original_temp_mem_used = mem_in_mbytes;
    }
    inline void _set_info(float mem_in_mbytes, Info_to_type<MODEL_MEM>) {
        model_mem_used = mem_in_mbytes;
    }
    inline void _set_info(float mem_in_mbytes, Info_to_type<SYSTEM_MEM>) {
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
    float temp_mem_used{0.f};
    ///< original_temp_mem_used : temp memory used by old version [MB].default 0
    float original_temp_mem_used{0.f};
    ///< system_mem_used : system mem used by nvidia / amd GPU system resource [MB].default 0
    float system_mem_used{0.f};
    ///<  model_mem_used : mem used by model.default 0
    float model_mem_used{0.f};

    ///< is_optimized stand for whether optimized flag.default false
    bool is_optimized{false};
};

} /* namespace graph */

} /* namespac anakin */

#endif
