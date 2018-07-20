#include "saber/lite/core/tensor_lite.h"
#include <cmath>
namespace anakin{

namespace saber{

namespace lite{

template<ARMType ttype, DataType dtype>
Tensor<ttype, dtype>::Tensor() {
    _buf = std::make_shared<Buffer<ttype>>();
    _is_subbuf = false;
    _is_shared = false;
}

template<ARMType ttype, DataType dtype>
Tensor<ttype, dtype>::Tensor(Shape shape) {
        _shape = shape;
        _valid_shape = shape;
        _offset = Shape::zero(shape.dims());
        _buf = std::make_shared<Buffer<ttype>>(shape.count() * _type_len);
        _is_shared = false;
        _is_subbuf = false;
    }

template<ARMType ttype, DataType dtype>
Tensor<ttype, dtype>::Tensor(Dtype* data_ptr, Shape shape) {
    _shape = shape;
    _valid_shape = shape;
    _offset = Shape::zero(shape.dims());
    _buf = std::make_shared<Buffer<ttype>>(data_ptr, shape.count() * _type_len);
    _is_shared = false;
    _is_subbuf = false;
}

template<ARMType ttype, DataType dtype>
Tensor<ttype, dtype>::Tensor(const Tensor<ttype, dtype>& tensor){
    _shape = tensor._shape;
    _valid_shape = tensor._valid_shape;
    _offset = tensor._offset;
    _buf = tensor._buf;
    _is_shared = tensor._is_shared;
    _is_subbuf = tensor._is_subbuf;
}

template<ARMType ttype, DataType dtype>
SaberStatus Tensor<ttype, dtype>::set_shape(Shape valid_shape, Shape shape, Shape offset) {

    if (shape.dims() > 0) {
        LCHECK_EQ(shape.dims(), valid_shape.dims(), "input shape dims should be the same");
        _shape = shape;
    }
    if (offset.dims() > 0 && _is_subbuf) {
        LCHECK_EQ(offset.dims(), valid_shape.dims(), "input shape dims should be the same");
        _offset = offset;
    }
    _valid_shape = valid_shape;
    if (!_is_subbuf) {
        if (_shape.count() <= _valid_shape.count()) {
            _shape = _valid_shape;
        }
        _offset = Shape::zero(valid_shape.dims());
    } else {
        auto shape_zero = Shape::zero(valid_shape.dims());
        if (_shape == shape_zero) {
            _shape = valid_shape;
        }
        LCHECK_EQ(_valid_shape + _offset <= _shape, true, "valid_shape + offet should <= shape");
    }
    return SaberSuccess;
}

template<ARMType ttype, DataType dtype>
SaberStatus Tensor<ttype, dtype>::re_alloc(Shape shape){
    LCHECK_EQ(_is_shared || _is_subbuf, false, "shared tensor could not re_alloc");
    _shape = shape;
    _valid_shape = _shape;
    _offset = Shape::zero(_shape.dims());
    _buf->alloc(_shape.count() * _type_len);
    return SaberSuccess;
}


template<ARMType ttype, DataType dtype>
SaberStatus Tensor<ttype, dtype>::reshape(Shape valid_shape, Shape shape, Shape offset) {

    if (shape.dims() > 0) {
        LCHECK_EQ(shape.dims(), valid_shape.dims(), "shape dims must be the same");
        _shape = shape;
    }
    if (offset.dims() > 0 && _is_subbuf) {
        LCHECK_EQ(offset.dims(), valid_shape.dims(), "shape dims must be the same");
        _offset = offset;
    }
    _valid_shape = valid_shape;
    if (!_is_subbuf) {
        if (_shape.count() < _valid_shape.count()) {
            _shape = _valid_shape;
        }
        _offset = Shape::zero(valid_shape.dims());
    } else {
        LCHECK_EQ(_valid_shape + _offset <= _shape, true, "valid_shape + offet should <= shape");
    }
    bool exceed_flag = _shape.count() * _type_len > _buf->get_capacity() \
            && (_is_subbuf || _is_shared);
    LCHECK_EQ(exceed_flag, false, "shared tensor shape exceed origin data buffer size");
    _buf->re_alloc(_shape.count() * _type_len);
    return SaberSuccess;
}

template<ARMType ttype, DataType dtype>
bool Tensor<ttype, dtype>::is_continue_mem() const {
    if (!_is_subbuf) {
        return true;
    }
    return _valid_shape.is_continue(_shape);
}

template<ARMType ttype, DataType dtype>
int Tensor<ttype, dtype>::count(int start, int end) const {

    LCHECK_GE(start, 0, "start index shold >= 0!");
    LCHECK_LE(end, _shape.size(), "end index shold <= shape dims!");
    LCHECK_LE(start, end, "start index should < end index!");
    int sum  = 1;
    for (int i = start; i < end; ++i) {
        sum *= _shape[i];
    }
    return sum;
}

template<ARMType ttype, DataType dtype>
int Tensor<ttype, dtype>::count_valid(int start, int end) const {

    start = std::max(start, 0);
    start  = std::min(start, _valid_shape.dims());
    end = std::max(start, end);
    end = std::min(end, _valid_shape.dims());

    int sum  = 1;
    for (int i = start; i < end; ++i) {
        sum *= _valid_shape[i];
    }
    return sum;
}

template<ARMType ttype, DataType dtype>
int Tensor<ttype, dtype>::size() const {
    return _shape.count();
}

template<ARMType ttype, DataType dtype>
int Tensor<ttype, dtype>::valid_size() const{
    return _valid_shape.count();
}

template<ARMType ttype, DataType dtype>
int Tensor<ttype, dtype>::dims() const {
    return _valid_shape.dims();
}

template<ARMType ttype, DataType dtype>
Shape Tensor<ttype, dtype>::shape() const{
    return _shape;
}

template<ARMType ttype, DataType dtype>
Shape Tensor<ttype, dtype>::valid_shape() const {
    return _valid_shape;
}

template<ARMType ttype, DataType dtype>
Shape Tensor<ttype, dtype>::get_stride() const {
    Shape data_stride = Shape::zero(dims());
    if (_is_subbuf) {
        for (int i = 0; i < dims(); ++i) {
            data_stride[i] = _shape.count(i + 1);
        }
    } else {
        for (int i = 0; i < dims(); ++i) {
            data_stride[i] = _valid_shape.count(i + 1);
        }
    }
    return data_stride;
}

template<ARMType ttype, DataType dtype>
Shape Tensor<ttype, dtype>::offset() const {
    return _offset;
}

template<ARMType ttype, DataType dtype>
int Tensor<ttype, dtype>::num() const {
    return _valid_shape.num();
}

template<ARMType ttype, DataType dtype>
void Tensor<ttype, dtype>::set_num(int num) {
    return _valid_shape.set_num(num);
};

template<ARMType ttype, DataType dtype>
int Tensor<ttype, dtype>::channel() const {
    return _valid_shape.channel();
}

template<ARMType ttype, DataType dtype>
void Tensor<ttype, dtype>::set_channel(int channel) {
    return _valid_shape.set_channel(channel);
}

template<ARMType ttype, DataType dtype>
int Tensor<ttype, dtype>::height() const {
    return _valid_shape.height();
}

template<ARMType ttype, DataType dtype>
void Tensor<ttype, dtype>::set_height(int h) {
    return _valid_shape.set_height(h);
}

template<ARMType ttype, DataType dtype>
int Tensor<ttype, dtype>::width() const {
    return _valid_shape.width();
}

template<ARMType ttype, DataType dtype>
void Tensor<ttype, dtype>::set_width(int w) {
    return _valid_shape.set_width(w);
}

template<ARMType ttype, DataType dtype>
typename Tensor<ttype, dtype>::Dtype* Tensor<ttype, dtype>::mutable_data(int index) {
    if (_buf->get_capacity() == 0){
        return nullptr;
    }
    return static_cast<Dtype*>(_buf->get_data_mutable()) + start_index() + index;
}

template<ARMType ttype, DataType dtype>
const typename Tensor<ttype, dtype>::Dtype * Tensor<ttype, dtype>::data(int index) const {
    if (_buf->get_capacity() == 0){
        return nullptr;
    }
    return static_cast<const Dtype*>(_buf->get_data()) + start_index() + index;
}

template<ARMType ttype, DataType dtype>
const std::shared_ptr<Buffer<ttype>>& Tensor<ttype, dtype>::get_buf() const {
    return _buf;
}

template<ARMType ttype, DataType dtype>
//template <typename Tensor_t>
SaberStatus Tensor<ttype, dtype>::share_from(const Tensor& tensor) {

    LCHECK_EQ(_shape.dims() > 0, true, "current tensor is not initialized (no shape info, use set_shape)");
    LCHECK_LE(size(), tensor.size(), "current tensor size should <= input tensor size");
    //typedef typename Tensor_t::Dtype_real dtype_real_t;
    //LCHECK_LE(size() * _type_len, tensor.size() * sizeof(dtype_real_t), "current tensor size should <= input tensor size");
    _buf = tensor.get_buf();
    _is_shared = true;
    _is_subbuf = false;
    return SaberSuccess;
}

template<ARMType ttype, DataType dtype>
SaberStatus Tensor<ttype, dtype>::share_sub_buffer(const Tensor<ttype, dtype>& tensor, \
        Shape valid_shape, Shape offset) {

    LCHECK_EQ(true, (offset + valid_shape) <= tensor.shape(), "offset + valid_shape <= shape");
    _valid_shape = valid_shape;
    _offset = offset;
    _shape = tensor.shape();
    _buf = tensor.get_buf();
    _is_subbuf = true;
    _is_shared = true;
    return SaberSuccess;
}

template<ARMType ttype, DataType dtype>
//template <class Tensor_t>
SaberStatus Tensor<ttype, dtype>::copy_from(const Tensor& tensor) {

    //size_t cap_dst = valid_size() * _type_len;
    //typedef typename Tensor_t::Dtype_real dtype_real_t;
    //size_t cap_src = tensor.valid_size() * sizeof(dtype_real_t);
    //LCHECK_EQ(cap_dst, cap_src, "sizes of two valid shapes must be the same");
    LCHECK_EQ(valid_size(), tensor.valid_size(), "sizes of two valid shapes must be the same");
    _buf->copy_from(*tensor.get_buf());
    return SaberSuccess;
}

template<ARMType ttype, DataType dtype>
void Tensor<ttype, dtype>::sync() {
    //!fixme
}

template<ARMType ttype, DataType dtype>
void Tensor<ttype, dtype>::record_event(stream_t* stream) {
    //! fixme
}

template<ARMType ttype, DataType dtype>
int Tensor<ttype, dtype>::start_index() const {
    if (!_is_subbuf) {
        return 0;
    }
    Shape stride = get_stride();
    int idx = 0;
    for (int i = 0; i < stride.size(); ++i) {
        idx += _offset[i] * stride[i];
    }
    return idx;
}
template class Tensor<CPU, AK_FLOAT>;

} //namespace lite

} //namespace saber

} //namespace anakin

