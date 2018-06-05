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

#ifndef ANAKIN_SABER_LITE_CORE_BUFFER_LITE_H
#define ANAKIN_SABER_LITE_CORE_BUFFER_LITE_H

#include "saber/lite/core/common_lite.h"

namespace anakin{

namespace saber{

namespace lite{

//! the alignment of all the allocated buffers
const int MALLOC_ALIGN = 16;

inline void* fast_malloc(size_t size) {
    size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
    char* p;
    p = static_cast<char*>(malloc(offset + size));
    if (!p) {
        return nullptr;
    }
    void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) & (~(MALLOC_ALIGN - 1)));
    static_cast<void**>(r)[-1] = p;
    return r;
}

inline void fast_free(void* ptr) {
    if (ptr){
        free(static_cast<void**>(ptr)[-1]);
    }
}

class Buffer{
public:

    /**
     * \brief constructor
     */
    Buffer() {
        _capacity = 0;
        _data = nullptr;
        _own_data = false;
    }
    /**
     * \brief constructor, allocate data
     */
    explicit Buffer(size_t size){
        _capacity = size;
    }
    /**
     * \brief assigned function
     */
    Buffer& operator = (Buffer& buf){
        this->_capacity = buf._capacity;
        this->_own_data = false;
        this->_data = buf._data;
        return *this;
    }
	
    /**
     * \brief destructor
     */
    virtual ~Buffer(){}

	/**
	* \brief deep copy function
	*/
	virtual void copy_from(Buffer& buf) = 0;

    /**
     * \brief set _data to (c) with length of (size)
     */
    virtual void mem_set(int c, size_t size) = 0;

    /**
     * \brief re-alloc memory
     */
    virtual SaberStatus re_alloc(size_t size) = 0;

    /**
     * \brief alloc memory
     */
    virtual SaberStatus alloc(size_t size) = 0;

    /**
     * \brief free memory
     */
    virtual void clean() =0;

    /**
     * \brief return const data pointer
     */
    virtual const void* get_data() = 0;

    /**
     * \brief return mutable data pointer
     */
    virtual void* get_data_mutable() = 0;

    /**
     * \brief return total size of memory, in bytes
     */
    inline size_t get_capacity() { return _capacity;}


protected:
    void* _data;
    bool _own_data;
    size_t _capacity;

};
#ifdef USE_ARM_CL
class GpuBuffer : public Buffer {
public:
    explicit GpuBuffer();
    ~GpuBuffer();
    explicit GpuBuffer(size_t size);
    explicit GpuBuffer(void* data, size_t size);
    GpuBuffer& operator = (GpuBuffer& buf) {
        this->_capacity = buf._capacity;
        this->_own_data = false;
        this->_data = buf._data;
        return *this;
    }
    virtual void re_alloc(size_t size);
    virtual void clean();
    virtual void mem_set(int c, size_t size);
    virtual const void* get_data();
    virtual void* get_data_mutable();
	virtual void copyto(GpuBuffer& buf);

};
#endif
class CpuBuffer : public Buffer {
public:
    explicit  CpuBuffer();
    ~CpuBuffer();
    explicit CpuBuffer(size_t size);
    explicit CpuBuffer(void* data, size_t size);
    CpuBuffer& operator = (CpuBuffer& buf);
    virtual SaberStatus re_alloc(size_t size);
    virtual SaberStatus alloc(size_t size);
    virtual void clean();
    virtual void mem_set(int c, size_t size);
    virtual const void* get_data();
    virtual void* get_data_mutable();
	virtual void copy_from(Buffer& buf);
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif //ANAKIN_SABER_LITE_CORE_BUFFER_LITE_H
