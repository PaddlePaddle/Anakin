// create by lxy890123

#ifndef MERCURY_BASE_MEMORY_H
#define MERCURY_BASE_MEMORY_H

#include "common.h"

namespace mercury{

// the alignment of all the allocated buffers
const int MALLOC_ALIGN = 16;

static inline void* fast_malloc(size_t size)
{
    size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
    char* p;
#ifdef USE_CUDA
    CUDA_CHECK(cudaMallocHost(&p, offset + size));
#else
    p = static_cast<char*>(malloc(offset + size));
#endif
    if (!p) {
        return nullptr;
    }
    void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) & (~(MALLOC_ALIGN - 1)));
    static_cast<void**>(r)[-1] = p;
    return r;
}

static inline void fast_free(void* ptr)
{
    if (ptr){
#ifdef USE_CUDA
        CUDA_CHECK(cudaFreeHost(static_cast<void**>(ptr)[-1]));
#else
        free(static_cast<void**>(ptr)[-1]);
#endif
    }
}

class Memory {
public:

    /*
     * \brief constructor
     */
    Memory(){
        _capacity = 0;
        _data = nullptr;
        _own_data = false;
    }
    /*
     * \brief constructor, allocate data
     */
    explicit Memory(size_t size){
        _capacity = size;
    }
    /*
     * \brief assigned function
     */
    Memory& operator = (Memory& buf){
        this->_capacity = buf._capacity;
        this->_own_data = false;
        this->_data = buf._data;
        return *this;
    }
	
    /*
     * \brief destructor
     */
    ~Memory(){}

	/*
	* \brief deep copy function
	*/
	virtual void copyto(Memory& buf) = 0;

    /*
     * \brief set _data to (c) with length of (size)
     */
    virtual void mem_set(int c, size_t size) = 0;

    /*
     * \brief re-alloc memory
     */
    virtual void re_alloc(size_t size) = 0;

    /*
     * \brief free memory
     */
    virtual void clean() =0;

    /*
     * \brief return const data pointer
     */
    virtual const void* get_data() = 0;

    /*
     * \brief return mutable data pointer
     */
    virtual void* get_data_mutable() = 0;

    /*
     * \brief return total size of memory, in bytes
     */
    inline size_t get_capacity() { return _capacity;}


protected:
    void* _data;
    bool _own_data;
    size_t _capacity;

};

class GpuMemory : public Memory {
public:
    explicit GpuMemory();
    ~GpuMemory();
    explicit GpuMemory(size_t size);
    explicit GpuMemory(void* data, size_t size);
    GpuMemory& operator = (GpuMemory& buf) {
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
	virtual void copyto(Memory& buf);

};

class CpuMemory : public Memory {
public:
    explicit  CpuMemory();
    ~CpuMemory();
    explicit CpuMemory(size_t size);
    explicit CpuMemory(void* data, size_t size);
    CpuMemory& operator = (CpuMemory& buf) {
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
	virtual void copyto(Memory& buf);

    // register pinned memory.
    inline void reg_page_lock(size_t size);

private:
    bool page_lock;  // whether the memory is pageable
    bool num_page_aligned;
};

} //namespace mercury
#endif //MERCURY_BASE_MEMORY_H
