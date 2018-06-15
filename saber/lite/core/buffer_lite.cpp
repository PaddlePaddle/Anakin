#include "saber/lite/core/buffer_lite.h"
#include <cstring>
namespace anakin{

namespace saber{

namespace lite{

CpuBuffer::CpuBuffer() : Buffer(){}

CpuBuffer::CpuBuffer(size_t size) : Buffer(size) {
    _own_data = true;
    _data = fast_malloc(_capacity);
}
CpuBuffer::CpuBuffer(void *data, size_t size) : Buffer(size) {
    _own_data = false;
    _data = data;
}

CpuBuffer& CpuBuffer::operator = (CpuBuffer& buf) {
    this->_capacity = buf._capacity;
    this->_own_data = false;
    this->_data = buf._data;
    return *this;
}
SaberStatus CpuBuffer::re_alloc(size_t size) {
    if(_own_data && size < _capacity) {
        return SaberSuccess;
    } else {
        clean();
        _capacity = size;
        _own_data = true;
        _data = fast_malloc(_capacity);
    }
}

SaberStatus CpuBuffer::alloc(size_t size) {
    clean();
    _capacity = size;
    _own_data = true;
    _data = fast_malloc(_capacity);
}

void CpuBuffer::copy_from(Buffer &buf) {
    if (buf.get_data() == _data) {
        return;
    }
	memcpy(_data, buf.get_data(), _capacity);
}

const void* CpuBuffer::get_data() {
    return _data;
}
void* CpuBuffer::get_data_mutable() {
    return _data;
}
void CpuBuffer::mem_set(int c, size_t size) {
    //CHECK_LE(size, _capacity) << "memset size must equal to or less than buffer size! ";
    memset(_data, c, size);
}

void CpuBuffer::clean(){
    if (_own_data){
        fast_free(_data);
    }
    _own_data = false;
    _data = nullptr;
    _capacity = 0;
}
CpuBuffer::~CpuBuffer() {
    clean();
}

} //namespace lite

} //namespace saber

} //namespace anakin
