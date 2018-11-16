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

#ifndef ANAKIN_SABER_LITE_CORE_TENSOR_LITE_H
#define ANAKIN_SABER_LITE_CORE_TENSOR_LITE_H

#include "saber/lite/core/shape_lite.h"
#include "saber/lite/core/buffer_lite.h"

namespace anakin{

namespace saber{

namespace lite{

template <ARMType ttype>
class Tensor {
public:
//    typedef typename DataTrait<ttype, dtype>::dtype Dtype;//float, char or CLMEM
//    typedef typename DataTrait<ttype, dtype>::Dtype Dtype_real;//float, char
    typedef typename DataTraitBase<ttype>::PtrDtype BaseType;
    typedef typename TargetTrait<ttype>::event_t event_t;
    typedef typename TargetTrait<ttype>::stream_t stream_t;
    /**
     *  \brief Default constructor
     */
    Tensor(DataType type = AK_FLOAT) : _valid_shape(), _shape(), _offset() {
        _dtype = type;
        _type_len = type_length(type);
        _buf = std::make_shared<Buffer<ttype>>();
        _is_subbuf = false;
        _is_shared = false;
    }

    /**
     * \brief Constructor with shape, memory is alloced according to shape.
     */
    Tensor(Shape shape, DataType type = AK_FLOAT) {
        _shape = shape;
        _valid_shape = shape;
        _offset = Shape::zero(shape.dims());
        _dtype = type;
        _type_len = type_length(type);
        _buf = std::make_shared<Buffer<ttype>>(shape.count() * _type_len);
        _is_shared = false;
        _is_subbuf = false;
    }

    /**
     * \brief Constructor with allocated data ptr and entire memory shape.
     */
    Tensor(BaseType data_ptr, Shape shape, DataType type = AK_FLOAT) {
        _shape = shape;
        _valid_shape = shape;
        _offset = Shape::zero(shape.dims());
        _dtype = type;
        _type_len = type_length(type);
        _buf = std::make_shared<Buffer<ttype>>(data_ptr, shape.count() * _type_len);
        _is_shared = false;
        _is_subbuf = false;
    }

    /**
     * \brief Copy constructor, shallow copy.
     */
    Tensor(const Tensor<ttype>& tensor) {
        _shape = tensor._shape;
        _valid_shape = tensor._valid_shape;
        _offset = tensor._offset;
        _dtype = tensor._dtype;
        _type_len = tensor._type_len;
        _buf = tensor._buf;
        _is_subbuf = tensor._is_subbuf;
        _is_shared = tensor._is_shared;
        _seq_offset = tensor._seq_offset;
        _scale = tensor._scale;
    }

    /**
     *  \brief only change the shape and valid shape, do nothing to memory
     *  \param shape
     *  \param valid_shape
     *  \param offset
     */
    SaberStatus set_shape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape()) {
        if (shape.dims() > 0) {
            LCHECK_EQ(shape.dims(), valid_shape.dims(), "ERROR: input shape dims should be the same\n");
            _shape = shape;
        }
        if (offset.dims() > 0 && _is_subbuf) {
            LCHECK_EQ(offset.dims(), valid_shape.dims(), "ERROR: input shape dims should be the same\n");
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
            LCHECK_EQ(_valid_shape + _offset <= _shape, true, "ERROR: valid_shape + offet should <= shape\n");
        }
        return SaberSuccess;
    }

    /**
     *  \brief Free old buffer and alloc a new tensor buffer.
     */
    SaberStatus re_alloc(Shape shape, DataType type = AK_INVALID) {
        LCHECK_EQ(_is_shared || _is_subbuf, false, "ERROR: shared tensor could not re_alloc\n");
        if (type != AK_INVALID) {
            _dtype = type;
            _type_len = type_length(_dtype);
        }
        _shape = shape;
        _valid_shape = _shape;
        _offset = Shape::zero(_shape.dims());
        _buf->alloc(_shape.count() * _type_len);
        return SaberSuccess;
    }

    /**
     *  \brief Change tensor shape,
     *  if input shape's count is bigger than the capacity of buffer, alloc a new buffer.
     */
    SaberStatus reshape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape()) {
        if (shape.dims() > 0) {
            LCHECK_EQ(shape.dims(), valid_shape.dims(), "ERROR: shape dims must be the same\n");
            _shape = shape;
        }
        if (offset.dims() > 0 && _is_subbuf) {
            LCHECK_EQ(offset.dims(), valid_shape.dims(), "ERROR: shape dims must be the same\n");
            _offset = offset;
        }
        _valid_shape = valid_shape;
        if (!_is_subbuf) {
            if (_shape.count() < _valid_shape.count()) {
                _shape = _valid_shape;
            }
            _offset = Shape::zero(valid_shape.dims());
        } else {
            LCHECK_EQ(_valid_shape + _offset <= _shape, true, "ERROR: valid_shape + offet should <= shape\n");
        }
        bool exceed_flag = _shape.count() * _type_len > _buf->get_capacity() \
            && (_is_subbuf || _is_shared);
        LCHECK_EQ(exceed_flag, false, "ERROR: shared tensor shape exceed origin data buffer size\n");
        _buf->re_alloc(_shape.count() * _type_len);
        return SaberSuccess;
    }

    bool is_continue_mem() const {
        if (!_is_subbuf) {
            return true;
        }
        return _valid_shape.is_continue(_shape);
    }

    /**
     *  \brief Return shape count, from start index to end index(end index is excluded).
     *  \param start Input start index.
     *  \param end   Input end index (exclude in calculation).
     *  \return the size from start index to end index.
     */
    int count(int start, int end) const {
        LCHECK_GE(start, 0, "ERROR: start index shold >= 0!\n");
        LCHECK_LE(end, _shape.size(), "ERROR: end index shold <= shape dims!\n");
        LCHECK_LE(start, end, "ERROR: start index should < end index!\n");
        int sum  = 1;
        for (int i = start; i < end; ++i) {
            sum *= _shape[i];
        }
        return sum;
    }

    /**
     *  \brief return valid_shape count, from start index to end index(end index is excluded).
     *  \param start input start index.
     *  \param end   input end index (exclude in calculation).
     *  \return the size from start index to end index.
     */
    int count_valid(int start, int end) const {
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

    /**
     * \brief get sequence offset, lot tensor
     * @return
     */
    std::vector<std::vector<int>> get_seq_offset() const {
        return _seq_offset;
    }

    /**
     * \brief set sequence offset, lot tensor
     * @param seq_offset
     * @return
     */
    SaberStatus set_seq_offset(std::vector<std::vector<int>> seq_offset) {
        _seq_offset = seq_offset;
        return SaberSuccess;
    }

    /**
     * \brief set scale for different precision data convert
     * @param scale
     */
    void set_scale(const std::vector<float> scale) {
        _scale = scale;
    }

    /**
     * \brief get scale
     * @param scale
     */
    std::vector<float> get_scale() const {
        return _scale;
    }

    SaberStatus set_dtype(DataType type) {
        _dtype = type;
        _type_len = type_length(type);
        if (_buf->get_capacity() < _shape.count() * _type_len) {
            if (_is_shared || _is_subbuf) {
                LOGF("tensor is shared, memory can not be re-alloced");
                return SaberOutOfAuthority;
            }
            _buf->re_alloc(_shape.count() * _type_len);
        }
        return SaberSuccess;
    }

    /**
     * \brief get tensor's DataType, AK_INT8 / AK_FLOAT ...
     * @return
     */
    DataType get_dtype() const {
        return _dtype;
    }

    size_t get_dtype_size() const {
        return _type_len;
    }

    /**
     *  \brief Return tensor shape size, not the valid shape size.
     */
    int size() const {
        return _shape.count();
    }

    /**
     *  \brief Return the valid shape size.
     *  \return Return the valid shape size.
     */
    int valid_size() const {
        return _valid_shape.count();
    }

    /**
     *  \brief Return tensor shape dims.
     */
    int dims() const {
        return _valid_shape.dims();
    }

    /**
     *  \brief Return tensor shape, entire memory buffer shape.
     */
    Shape shape() const {
        return _shape;
    }

    /**
     *  \brief Return valid shape of tensor
     */
    Shape valid_shape() const {
        return _valid_shape;
    }

    /**
     *  \brief compute data stride.
     */
    Shape get_stride() const {
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
    /**
     *  \brief Return tensor offset, which holds the offset in each dim.
     */
    Shape offset() const {
        return _offset;
    }

    /**
     *  \brief Return number
     */
    int num() const {
        return _valid_shape.num();
    }

    /**
     *  \brief Return number index in shape.
     */
    void set_num(int num) {
        _valid_shape.set_num(num);
    }

    /**
     *  \brief Return channel.
     */
    int channel() const {
        return _valid_shape.channel();
    }

    /**
     *  \brief Return channel index in shape.
     *  \return
     */
    void set_channel(int channel) {
        _valid_shape.set_channel(channel);
    }

    /**
     *  \brief Return height.
     *  \return
     */
    int height() const {
        return _valid_shape.height();
    }

    /**
     *  \brief Return height index in shape.
     *  \return
     */
    void set_height(int h) {
        _valid_shape.set_height(h);
    }

    /**
     *  \brief Return width.
     *  \return
     */
    int width() const {
        return _valid_shape.width();
    }

    /**
     *  \brief Return height index in shape.
     *  \return
     */
    void set_width(int w) {
        _valid_shape.set_width(w);
    }

    /**
     *  \brief Return tensor mutable data pointer, with data type of current tensor (Dtype*).
     */
    BaseType mutable_data() {
        if (_buf->get_capacity() == 0){
            return nullptr;
        }

        return static_cast<BaseType>(_buf->get_data_mutable());
    }

    /**
     *  \brief Return tensor data pointer, with data type of current tensor (Dtype*).
     */
    const BaseType data(int index = 0) const {
        if (_buf->get_capacity() == 0){
            return nullptr;
        }
        return static_cast<BaseType>(_buf->get_data());
    }

    /**
     *  \brief Share from same layout_type and same date type tensor,
     *  if shared tensor target is the same with current tensor target, buffer is shared;
     *  otherwise, tensor buffer is deep copied.
     *  only shared buffer ptr, current tensor will have continuous memory,
     *  only if current shape and valid shape are the same, and offset is all set to 0.
     */
    SaberStatus share_from(const Tensor& tensor) {
        LCHECK_LE(size(), tensor.size(), "ERROR: current tensor size should <= input tensor size\n");
        LCHECK_GE(tensor._buf->get_capacity(), _shape.count() * _type_len, \
            "ERROR: capacity of input tensor should > current tensor\n");
        _buf = tensor._buf;
        _is_subbuf = false;
        _seq_offset = tensor._seq_offset;
        _is_shared = true;
    }

    SaberStatus share_sub_buffer(const Tensor& tensor, \
        Shape valid_shape, Shape offset) {
        LCHECK_EQ(true, (offset + valid_shape) <= tensor.shape(), "ERROR: offset + valid_shape <= shape\n");
        _valid_shape = valid_shape;
        _offset = offset;
        _shape = tensor.shape();
        _buf = tensor._buf;
        _is_subbuf = true;
        _is_shared = true;
        _seq_offset = tensor._seq_offset;
        return SaberSuccess;
    }

    /**
     *  \brief Deep copy data within region of interest from input tensor.
     */
    //template <class Tensor_t>
    SaberStatus copy_from(const Tensor& tensor) {
        LCHECK_EQ(valid_size(), tensor.valid_size(), "ERROR: sizes of two valid shapes must be the same\n");
        _buf->copy_from(*tensor._buf);
        return SaberSuccess;
    }

    /**
     *  \brief Synchronize the event tree, wait util all events are done.
     */
    void sync() {
        // fixme
    }

    /**
     *  \brief record Event to current tensor.
     *  \param stream  Input processing stream.
     */
    void record_event(stream_t* stream) {
        // fixme
    }


private:
    //! scale for quantization
    std::vector<float> _scale;

    ///< data type. AK_FLOAT AK_INT8
    DataType _dtype{AK_FLOAT};

    ///< Length of datatype.
    size_t _type_len{4};

    ///< Represent the raw mem shape.
    Shape _shape;

    ///< Represent the mem you have right to access shape.
    Shape _valid_shape;

    ///< Represent the offset idx between _shape and _real_shape.
    Shape _offset;

    ///< Buffer shared ptr, hold the data pointer, and buffer capacity.
    std::shared_ptr<Buffer<ttype>> _buf{nullptr};

    ///< share sub-buffer flag.
    bool _is_subbuf{false};
    bool _is_shared{false};

    //! lot tensor
    std::vector<std::vector<int>> _seq_offset;

    ///< event
    event_t _event;

    /// Get data real start index.
    int start_index() const {
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
};

} //namespace lite

} //namespace saber

} //namespace anakin


#endif //ANAKIN_SABER_LITE_CORE_TENSOR_LITE_H

