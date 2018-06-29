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

template <ARMType ttype, DataType dtype>
class Tensor {
public:
    typedef typename DataTrait<ttype, dtype>::dtype Dtype;//float, char or CLMEM
    typedef typename DataTrait<ttype, dtype>::Dtype Dtype_real;//float, char
    typedef typename TargetTrait<ttype>::event_t event_t;
    typedef typename TargetTrait<ttype>::stream_t stream_t;
    /**
     *  \brief Default constructor
     */
    Tensor();

    /**
     * \brief Constructor with shape, memory is alloced according to shape.
     */
    Tensor(Shape shape);

    /**
     * \brief Constructor with allocated data ptr and entire memory shape.
     */
    Tensor(Dtype* data_ptr, Shape shape);

    /**
     * \brief Copy constructor, shallow copy.
     */
    Tensor(const Tensor<ttype, dtype>& tensor);

    /**
     *  \brief only change the shape and valid shape, do nothing to memory
     *  \param shape
     *  \param valid_shape
     *  \param offset
     */
    SaberStatus set_shape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape());

    /**
     *  \brief Free old buffer and alloc a new tensor buffer.
     */
    SaberStatus re_alloc(Shape shape);


    /**
     *  \brief Change tensor shape,
     *  if input shape's count is bigger than the capacity of buffer, alloc a new buffer.
     */
    SaberStatus reshape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape());

    bool is_continue_mem() const;

    /**
     *  \brief Return shape count, from start index to end index(end index is excluded).
     *  \param start Input start index.
     *  \param end   Input end index (exclude in calculation).
     *  \return the size from start index to end index.
     */
    int count(int start, int end) const;

    /**
     *  \brief return valid_shape count, from start index to end index(end index is excluded).
     *  \param start input start index.
     *  \param end   input end index (exclude in calculation).
     *  \return the size from start index to end index.
     */
    int count_valid(int start, int end) const;

    /**
     *  \brief Return tensor shape size, not the valid shape size.
     */
    int size() const;

    /**
     *  \brief Return the valid shape size.
     *  \return Return the valid shape size.
     */
    int valid_size() const;

    /**
     *  \brief Return tensor shape dims.
     */
    int dims() const;

    /**
     *  \brief Return tensor shape, entire memory buffer shape.
     */
    Shape shape() const;

    /**
     *  \brief Return valid shape of tensor
     */
    Shape valid_shape() const;

    /**
     *  \brief compute data stride.
     */
    Shape get_stride() const;
    /**
     *  \brief Return tensor offset, which holds the offset in each dim.
     */
    Shape offset() const;

    /**
     *  \brief Return number
     */
    int num() const;

    /**
     *  \brief Return number index in shape.
     */
    void set_num(int num);

    /**
     *  \brief Return channel.
     */
    int channel() const;

    /**
     *  \brief Return channel index in shape.
     *  \return
     */
    void set_channel(int channel);

    /**
     *  \brief Return height.
     *  \return
     */
    int height() const;

    /**
     *  \brief Return height index in shape.
     *  \return
     */
    void set_height(int h);

    /**
     *  \brief Return width.
     *  \return
     */
    int width() const;

    /**
     *  \brief Return height index in shape.
     *  \return
     */
    void set_width(int w);

    /**
     *  \brief Return tensor mutable data pointer, with data type of current tensor (Dtype*).
     */
    Dtype* mutable_data(int index = 0);

    /**
     *  \brief Return tensor data pointer, with data type of current tensor (Dtype*).
     */
    const Dtype * data(int index = 0) const;

    /**
     *  \brief Return reference shared_ptr of tensor.
     */
    const std::shared_ptr<Buffer<ttype>>& get_buf() const;

    /**
     *  \brief Share from same layout_type and same date type tensor,
     *  if shared tensor target is the same with current tensor target, buffer is shared;
     *  otherwise, tensor buffer is deep copied.
     *  only shared buffer ptr, current tensor will have continuous memory,
     *  only if current shape and valid shape are the same, and offset is all set to 0.
     */

    template <class Tensor_t>
    SaberStatus share_from(const Tensor_t& tensor);

    SaberStatus share_sub_buffer(const Tensor<ttype, dtype>& tensor, \
        Shape valid_shape, Shape offset);

    /**
     *  \brief Deep copy data within region of interest from input tensor.
     */
     template <class Tensor_t>
    SaberStatus copy_from(const Tensor_t& tensor);

    /**
     *  \brief Synchronize the event tree, wait util all events are done.
     */
    void sync();

    /**
     *  \brief record Event to current tensor.
     *  \param stream  Input processing stream.
     */
    void record_event(stream_t* stream);


private:
    ///< Length of datatype.
    size_t _type_len{sizeof(Dtype_real)};

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

    ///< event
    event_t _event;

    /// Get data real start index.
    int start_index() const;
};

} //namespace lite

} //namespace saber

} //namespace anakin


#endif //ANAKIN_SABER_LITE_CORE_TENSOR_LITE_H

