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

#ifndef ANAKIN_SABER_LITE_CORE_TENSOR_LITE_H
#define ANAKIN_SABER_LITE_CORE_TENSOR_LITE_H

#include "saber/lite/core/shape_lite.h"
#include "saber/lite/core/buffer_lite.h"

namespace anakin{

namespace saber{

namespace lite{
template <typename Dtype>
class Tensor {
public:
    /**
     *  \brief Default constructor
     */
    Tensor() {
        _buf = std::make_shared<CpuBuffer>();
        _target_type = eARM;
        _is_subbuf = false;
        _is_shared = false;
    }

    /**
     * \brief Constructor with shape, memory is alloced according to shape.
     */
    Tensor(Shape shape, TargetTypeEnum target = eARM) {
        _target_type = target;
        _shape = shape;
        _valid_shape = shape;
        _offset = Shape::zero(shape.dims());
        _buf = std::make_shared<CpuBuffer>(shape.count() * _type_len);
        _is_subbuf = false;
    }

    /**
     * \brief Constructor with allocated data ptr and entire memory shape.
     */
    Tensor(Dtype* data_ptr, TargetTypeEnum target, int id, Shape shape) {

        _shape = shape;
        _valid_shape = shape;
        _offset = Shape::zero(shape.dims());
        _buf = std::make_shared<CpuBuffer>(data_ptr, shape.count() * _type_len);
        _is_shared = true;
        _is_subbuf = false;
    }

    /**
     * \brief Copy constructor, shallow copy.
     */
    Tensor(const Tensor<Dtype>& tensor){
        _shape = tensor._shape;
        _valid_shape = tensor._valid_shape;
        _offset = tensor._offset;
        _buf = tensor._buf;
        _is_subbuf = tensor._is_subbuf;
        _seq_offset = tensor._seq_offset;
    }

    /**
     *  \brief only change the shape and valid shape, do nothing to memory
     *  \param shape
     *  \param valid_shape
     *  \param offset
     */
    SaberStatus set_shape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape()) {

        if (shape.dims() > 0) {
            CHECK_EQ(shape.dims(), valid_shape.dims()) << "shape dims must be the same";
            _shape = shape;
        }
        if (offset.dims() > 0 && _is_subbuf) {
            CHECK_EQ(offset.dims(), valid_shape.dims()) << "shape dims must be the same";
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
            CHECK_EQ(_valid_shape + _offset <= _shape, true) << \
                "valid_shape + offet should <= shape";
        }
        return SaberSuccess;
    }

    /**
     *  \brief Free old buffer and alloc a new tensor buffer.
     */
    SaberStatus re_alloc(Shape shape){
        CHECK_EQ(_is_shared || _is_subbuf, false) << \
            "shared tensor could not re_alloc";
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
            CHECK_EQ(shape.dims(), valid_shape.dims()) << "shape dims must be the same";
            _shape = shape;
        }
        if (offset.dims() > 0 && _is_subbuf) {
            CHECK_EQ(offset.dims(), valid_shape.dims()) << "shape dims must be the same";
            _offset = offset;
        }

        _valid_shape = valid_shape;

        if (!_is_subbuf) {
            if (_shape.count() < _valid_shape.count()) {
                _shape = _valid_shape;
            }
            _offset = Shape::zero(valid_shape.dims());
        } else {
            CHECK_EQ(_valid_shape + _offset <= _shape, true) << \
                "valid_shape + offet should <= shape";
        }
        bool exceed_flag = _shape.count() * _type_len > _buf->get_capacity() \
            && (_is_subbuf || _is_shared);
        CHECK_EQ(exceed_flag, false) << "shared tensor shape exceed origin data buffer size";
        SABER_CHECK(_buf->re_alloc(_shape.count() * _type_len));
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

        CHECK_GE(start, 0) << "start index shold >= 0!";
        CHECK_LE(end, _shape.size()) << "end index shold <= shape dims!";
        CHECK_LE(start, end) << "start index should < end index!";
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

        CHECK_GE(start, 0) << "start index shold >= 0!";
        CHECK_LE(end, _valid_shape.size()) << "end index shold <= shape dims!";
        CHECK_LE(start, end) << "start index should < end index!";
        int sum  = 1;
        for (int i = start; i < end; ++i) {
            sum *= _valid_shape[i];
        }
        return sum;
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
    int valid_size() const{
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
    Shape shape() const{
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
     *  \brief Return tensor device id.
     */
    int device_id() const {
        return 0;
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
        return _valid_shape.set_num(num);
    };

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
        return _valid_shape.set_channel(channel);
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
        return _valid_shape.set_height(h);
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
        return _valid_shape.set_width(w);
    }

    /**
     *  \brief Return tensor mutable data pointer, with data type of current tensor (Dtype*).
     */
    Dtype* mutable_data(int index = 0) {
        if (_buf->get_capacity() == 0){
            return nullptr;
        }
        return static_cast<Dtype*>(_buf->get_data_mutable()) + start_index() + index;
    }

    /**
     *  \brief Return tensor data pointer, with data type of current tensor (Dtype*).
     */
    const Dtype * data(int index = 0) const {
        if (_buf->get_capacity() == 0){
            return nullptr;
        }
        return static_cast<const Dtype*>(_buf->get_data()) + start_index() + index;
    }

    /**
     *  \brief Return reference shared_ptr of tensor.
     */
    const std::shared_ptr<Buffer>& get_buf() const {
        return _buf;
    }

    /**
     *  \brief Share from same layout_type and same date type tensor,
     *  if shared tensor target is the same with current tensor target, buffer is shared;
     *  otherwise, tensor buffer is deep copied.
     *  only shared buffer ptr, current tensor will have continuous memory,
     *  only if current shape and valid shape are the same, and offset is all set to 0.
     */
    template <typename Tensor_t>
    SaberStatus share_from(const Tensor_t& tensor) {

        CHECK_EQ(_shape.dims() > 0, true) << \
            "current tensor is not initialized (no shape info, use set_shape)";
        typedef typename Tensor_t::Dtype dtype_t;
        CHECK_LE(size() * _type_len, tensor.size() * sizeof(dtype_t)) << \
            "current tensor size should <= input tensor size";
//! fix me, when use cl memory
        _buf = std::dynamic_pointer_cast<CpuBuffer>(tensor.get_buf());
        _is_shared = true;
        _is_subbuf = false;
        _seq_offset = tensor.get_seq_offset();
        return SaberSuccess;
    }

    std::vector<int> get_seq_offset() const {return _seq_offset;}
    SaberStatus set_seq_offset(std::vector<int> seq_offset) {_seq_offset = seq_offset; return SaberSuccess;}

    SaberStatus share_sub_buffer(const Tensor<Dtype>& tensor, \
        Shape valid_shape, Shape offset) {

        CHECK_EQ(true, (offset + valid_shape) <= tensor.shape()) << \
            "offset + valid_shape <= shape";
        _valid_shape = valid_shape;
        _offset = offset;
        _shape = tensor.shape();
        _buf = std::dynamic_pointer_cast<CpuBuffer>(tensor.get_buf());//tensor.get_buf();
        _is_subbuf = true;
        _is_shared = true;
        _seq_offset = tensor.get_seq_offset();
        return SaberSuccess;
    }

    /**
     *  \brief Deep copy data within region of interest from input tensor.
     */
    SaberStatus copy_from(const Tensor<Dtype>& tensor) {

        CHECK_EQ(valid_size(), tensor.valid_size()) \
            << "sizes of two valid shapes must be the same";

        std::shared_ptr<CpuBuffer> buf_tmp = std::dynamic_pointer_cast<CpuBuffer>(tensor.get_buf());
        _buf->copy_from(*buf_tmp);

        return SaberSuccess;
    }

    /**
     *  \brief Synchronize the event tree, wait util all events are done.
     */
    void sync() {
        //!fixme
    }

    /**
     *  \brief record Event to current tensor.
     *  \param stream  Input processing stream.
     */
    void record_event(void* stream) {
        //! fixme
    }


private:
    ///< target type, cpu or gpu
    TargetTypeEnum _target_type;
    ///< _layout
    ///< Length of datatype.
    size_t _type_len{sizeof(Dtype)};
    ///< Represent the raw mem shape.
    Shape _shape;
    ///< Represent the mem you have right to access shape.
    Shape _valid_shape;
    ///< Represent the offset idx between _shape and _real_shape.
    Shape _offset;
    ///< Buffer shared ptr, hold the data pointer, and buffer capacity.
    std::shared_ptr<Buffer> _buf{nullptr};
    ///< share sub-buffer flag.
    bool _is_subbuf{false};
    bool _is_shared{false};

    ///< event
    //fixme, add event

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

    std::vector<int> _seq_offset;
};

} //namespace lite

} //namespace saber

} //namespace anakin


#endif //ANAKIN_SABER_LITE_CORE_TENSOR_LITE_H

