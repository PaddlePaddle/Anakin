#include "saber/lite/core/buffer_lite.h"
namespace anakin{

namespace saber{

namespace lite{

template <>
void Buffer<CPU>::clean() {
    if (_own_data){
        fast_free(_data);
    }
    _own_data = false;
    _data = nullptr;
    _capacity = 0;
}

template <>
const Buffer<CPU>::dtype* Buffer<CPU>::get_data() {
    return _data;
}

template <>
Buffer<CPU>::Buffer() {
    _capacity = 0;
    _data = nullptr;
    _own_data = false;
}

template <>
Buffer<CPU>::Buffer(size_t size) {
    _own_data = true;
    _data = fast_malloc(size);
}

template <>
Buffer<CPU>::Buffer(dtype* data, size_t size) {
    _own_data = false;
    _data = data;
    _capacity = size;
}

template <>
Buffer<CPU>::~Buffer() {
    clean();
}

template <>
Buffer<CPU>& Buffer<CPU>::operator=(Buffer<CPU>& buf) {
    this->_capacity = buf._capacity;
    this->_own_data = false;
    this->_data = buf._data;
    return *this;
}

template <>
void Buffer<CPU>::re_alloc(size_t size) {
    if(_own_data && size < _capacity) {
        return;
    } else {
        clean();
        _capacity = size;
        _own_data = true;
        _data = fast_malloc(_capacity);
    }
}

template <>
void Buffer<CPU>::alloc(size_t size) {
    clean();
    _capacity = size;
    _own_data = true;
    _data = fast_malloc(_capacity);
}

template <>
void Buffer<CPU>::copy_from(Buffer<CPU> &buf) {
    if (buf.get_data() == _data) {
        return;
    }
	memcpy(_data, buf.get_data(), _capacity);
}

template <>
Buffer<CPU>::dtype* Buffer<CPU>::get_data_mutable() {
    return _data;
}

template <>
void Buffer<CPU>::mem_set(int c, size_t size) {
    if (size > _capacity) {
        size = _capacity;
    }
    memset(_data, c, size);
}

template <ARMType ttype>
size_t Buffer<ttype>::get_capacity() {
    return _capacity;
}

} //namespace lite

} //namespace saber

} //namespace anakin
