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

#ifndef ANAKIN_SABER_CORE_TENSOR_H
#define ANAKIN_SABER_CORE_TENSOR_H

#include "core/shape.h"
#include "core/events.h"
#include "core/tensor_traits.h"

namespace anakin{

namespace saber{

#define INSTANTIATE_TENSOR(TargetType, datatype, LayOutType) \
  template class Tensor<TargetType, datatype, LayOutType>;

class TensorBase {
public:
    TensorBase() {}
    virtual ~TensorBase() {}
    virtual SaberStatus set_shape(Shape valid_shape, Shape shape = Shape(), \
        Shape offset = Shape()) = 0;
    virtual SaberStatus reshape(Shape valid_shape, Shape shape = Shape(), \
        Shape offset = Shape()) = 0;
    virtual SaberStatus re_alloc(Shape shape) = 0;
    virtual bool is_continue_mem() const = 0;
    virtual int size() const = 0;
    virtual int valid_size() const = 0;
    virtual int count(int start, int end) const = 0;
    virtual int count_valid(int start, int end) const = 0;
    virtual int dims() const = 0;
    virtual Shape shape() const = 0;
    virtual Shape valid_shape() const = 0;
    virtual Shape get_stride() const = 0;
    virtual Shape offset() const = 0;
    virtual int device_id() const = 0;
    virtual int num() const = 0;
    virtual int num_index() const = 0;
    virtual int channel() const = 0;
    virtual int channel_index() const = 0;
    virtual int height() const = 0;
    virtual int height_index() const = 0;
    virtual int width() const = 0;
    virtual int width_index() const = 0;
};

template<typename TargetType, DataType datatype, typename LayOutType = NCHW>
class Tensor : public TensorBase {
public:
    typedef TargetType targetType_t;
    typedef typename DataTrait<datatype>::dtype Dtype;
    typedef typename TargetTypeTraits<TargetType>::target_category target_category;
    typedef typename TargetTypeTraits<TargetType>::target_type target_type;
    typedef TargetWrapper<TargetType> API;
    typedef TensorTraits<Tensor<TargetType, datatype, LayOutType>> TensorAPI;
    typedef typename TensorAPI::layout_category layout_category;
    typedef typename TensorAPI::layout_type layout_type;

    /**
     *  \brief Default constructor
     */
    Tensor() {
        _shape = Shape::zero(TensorAPI::layout_dims::value);
        _valid_shape = Shape::zero(TensorAPI::layout_dims::value);
        _offset = Shape::zero(TensorAPI::layout_dims::value);
        _buf = std::make_shared<Buffer<TargetType>>();
        _is_subbuf = false;
    }

    /**
     * \brief Constructor with shape, memory is alloced according to shape.
     */
    Tensor(Shape shape) {

        CHECK_EQ(shape.dims(), TensorAPI::layout_dims::value) << \
            "shape dims is not matched to layout type";
        _shape = shape;
        _valid_shape = shape;
        _offset = Shape::zero(shape.dims());
        _buf = std::make_shared<Buffer<TargetType>>(shape.count() * _type_len);
        _is_subbuf = false;
    }
#if 0
    /**
     * \brief constructor with currently used shape, offset and entire memory shape,
     * memory is alloced according to the shape
     */
    Tensor(Shape shape, Shape valid_shape, Shape offset) {
        CHECK_EQ(shape.dims(), TensorAPI::layout_dims::value) << \
            "shape dims is not matched to layout type";
        CHECK_EQ(valid_shape.dims(), TensorAPI::layout_dims::value) << \
            "valid shape dims is not matched to layout type";
        CHECK_EQ(offset.dims(), TensorAPI::layout_dims::value) << \
            "offset dims is not matched to layout type";
        CHECK_EQ(true, (offset + valid_shape) <= shape) << \
            "valid shape + offset should <= shape";
        _shape = shape;
        _valid_shape = valid_shape;
        _offset = offset;
        _buf = std::make_shared<Buffer<TargetType, datatype>>(_shape.count());
        _is_subbuf = false;
    }
#endif
    /**
     * \brief Constructor with allocated data ptr and entire memory shape.
     */
    template <typename TargetType_t>
    Tensor(Dtype* data_ptr, TargetType_t target, int id, Shape shape) {

        CHECK_EQ(shape.dims(), TensorAPI::layout_dims::value) << \
            "shape dims is not matched to layout type";
        _shape = shape;
        _valid_shape = shape;
        _offset = Shape::zero(shape.dims());
        std::shared_ptr<Buffer<TargetType_t>> buf_from_date = \
            std::make_shared<Buffer<TargetType_t>>(data_ptr, shape.count() * _type_len, id);
        BufferMemShare(_buf, buf_from_date);
        _is_subbuf = false;
    }

    /**
     * \brief Copy constructor, shallow copy.
     */
    Tensor(const Tensor<TargetType, datatype, LayOutType>& tensor){
        _shape = tensor._shape;
        _valid_shape = tensor._valid_shape;
        _offset = tensor._offset;
        _buf = tensor._buf;
        _is_subbuf = tensor._is_subbuf;
        _seq_offset = tensor._seq_offset;
    }

    /**
     * \brief Copy constructor without events control.
     */
    Tensor(Tensor<TargetType, datatype, LayOutType>& tensor){
        _shape = tensor._shape;
        _valid_shape = tensor._valid_shape;
        _offset = tensor._offset;
        _buf = tensor._buf;
        tensor.add_events(&_events_tree);
        _is_subbuf = tensor._is_subbuf;
        _seq_offset = tensor._seq_offset;
    }

    /**
     *  \brief only change the shape and valid shape, do nothing to memory
     *  \param shape
     *  \param valid_shape
     *  \param offset
     */
    SaberStatus set_shape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape() \
        /*Shape shape = Shape::zero(TensorAPI::layout_dims::value), \
        Shape offset = Shape::minusone(TensorAPI::layout_dims::value)*/) {

        //if (shape.dims() != TensorAPI::layout_dims::value || \
            valid_shape.dims() != TensorAPI::layout_dims::value \
            || offset.dims() != TensorAPI::layout_dims::value || \
            !(valid_shape > Shape::zero(TensorAPI::layout_dims::value))) { \
            return SaberInvalidValue; \
        }
        CHECK_EQ(valid_shape.dims(), TensorAPI::layout_dims::value) << \
            "valid shape dims is not matched to layout type";
        if (shape.dims() > 0) {
            CHECK_EQ(shape.dims(), TensorAPI::layout_dims::value) << \
                "shape dims is not matched to layout type";
            _shape = shape;
        }
        if (offset.dims() > 0 && _is_subbuf) {
            CHECK_EQ(offset.dims(), TensorAPI::layout_dims::value) << \
                "offset dims is not matched to layout type";
            _offset = offset;
        }
        CHECK_EQ(valid_shape > Shape::zero(TensorAPI::layout_dims::value), true) << \
            "valid_shape size should > 0";
        _valid_shape = valid_shape;

        if (!_is_subbuf) {
            if (_shape.count() <= _valid_shape.count()) {
                _shape = _valid_shape;
            }
            _offset = Shape::zero(TensorAPI::layout_dims::value);
        } else {
            auto shape_zero = Shape::zero(TensorAPI::layout_dims::value);
            if (_shape == shape_zero) {
                _shape = valid_shape;
            }
            //if (!(_valid_shape + _offset <= _shape)) { \
                return SaberInvalidValue; \
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
        //if (!shape.dims() == TensorAPI::layout_dims::value) {
        //    return SaberInvalidValue;
        //}
        //if (_is_subbuf || _is_shared) {
        //    return SaberOutOfAuthority;
        //}
        CHECK_EQ(shape.dims(), TensorAPI::layout_dims::value) << \
            "shape dims is not matched to layout type";
        CHECK_EQ(_is_shared || _is_subbuf, false) << \
            "shared tensor could not re_alloc"; // by ccw 2018/4/3
        _shape = shape;
        _valid_shape = _shape;
        _offset =Shape::zero(_shape.dims());
        _buf->alloc(_shape.count() * _type_len);
        return SaberSuccess;
    }

    void try_expand_size(Shape& shape) {
        //        LOG(INFO)<<"in try expand "<<shape.count()<<","<<valid_size();
        if (shape.count() > (valid_size())) {
            re_alloc(shape);
        }

    }
    void try_expand_size(int size) {
        Shape shape(1, 1, 1, size);
        try_expand_size(shape);
    }


    /**
     *  \brief Change tensor shape,
     *  if input shape's count is bigger than the capacity of buffer, alloc a new buffer.
     */
    SaberStatus reshape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape()\
        /*Shape::zero(TensorAPI::layout_dims::value), \
        Shape offset = Shape::minusone(TensorAPI::layout_dims::value)*/) {

        //if (shape.dims() != TensorAPI::layout_dims::value || \
            valid_shape.dims() != TensorAPI::layout_dims::value \
            || offset.dims() != TensorAPI::layout_dims::value || \
            !(valid_shape > Shape::zero(TensorAPI::layout_dims::value))) { \
            return SaberInvalidValue; \
        }
        CHECK_EQ(valid_shape.dims(), TensorAPI::layout_dims::value) << \
            "valid shape dims is not matched to layout type";
        if (shape.dims() > 0) {
            CHECK_EQ(shape.dims(), TensorAPI::layout_dims::value) << \
                "shape dims is not matched to layout type";
            _shape = shape;
        }
        if (offset.dims() > 0 && _is_subbuf) {
            CHECK_EQ(offset.dims(), TensorAPI::layout_dims::value) << \
                "offset dims is not matched to layout type";
            _offset = offset;
        }
        CHECK_EQ(valid_shape > Shape::zero(TensorAPI::layout_dims::value), true) << \
            "valid_shape size should > 0";
        _valid_shape = valid_shape;

        if (!_is_subbuf) {
            if (_shape.count() < _valid_shape.count()) {
                _shape = _valid_shape;
            }
            _offset = Shape::zero(TensorAPI::layout_dims::value);
        } else {
            if (_shape == Shape::zero(TensorAPI::layout_dims::value)) {
                _shape = valid_shape;
            }
            //if (!(_valid_shape + _offset <= _shape)) { \
                return SaberInvalidValue; \
            }
            CHECK_EQ(_valid_shape + _offset <= _shape, true) << \
                "valid_shape + offet should <= shape";
        }
        bool exceed_flag = _shape.count() * _type_len > _buf->get_capacity() \
            && (_is_subbuf || _is_shared);
        //if (exceed_flag) {
        //    return SaberOutOfAuthority;
        //}
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
        //if (start < 0) { \
            start = 0; \
        }
        //if (end > dims()) { \
            end = dims(); \
        }
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
        //if (start < 0) { \
            start = 0; \
        }
        //if (end > dims()) { \
            end = dims(); \
        }
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
        return TensorAPI::layout_dims::value;
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
     *  \brief Return reference shared_ptr of tensor.
     */
     const std::shared_ptr<Buffer<TargetType>>& get_buf() const {
         return _buf;
     }

    /**
     *  \brief Return tensor device id.
     */
    int device_id() const {
        return _buf->get_id();
    }

    /**
     *  \brief Return number
     */
    int num() const {
        return TensorAPI::num(_valid_shape);
    }

    /**
     *  \brief Return number index in shape.
     */
    int num_index() const {
        return TensorAPI::num_idx::value;
    };

    /**
     *  \brief Return channel.
     */
    int channel() const {
        return TensorAPI::channel(_valid_shape);
    }

    /**
     *  \brief Return channel index in shape.
     *  \return
     */
    int channel_index() const {
        return TensorAPI::channel_idx::value;
    }

    /**
     *  \brief Return height.
     *  \return
     */
    int height() const {
        return TensorAPI::height(_valid_shape);
    }

    /**
     *  \brief Return height index in shape.
     *  \return
     */
    int height_index() const {
        return TensorAPI::height_idx::value;
    }

    /**
     *  \brief Return width.
     *  \return
     */
    int width() const {
        return TensorAPI::width(_valid_shape);
    }

    /**
     *  \brief Return height index in shape.
     *  \return
     */
    int width_index() const {
        return TensorAPI::width_idx::value;
    }

    /**
     *  \brief Return tensor mutable data pointer, with data type of current tensor (Dtype*).
     */
    Dtype* mutable_data(int index = 0) {
        // synchronize the events tree
        //sync();
        CHECK_EQ(device_id(), API::get_device_id()) << \
            "tensor is not declared in current device";
        if (_buf->get_capacity() == 0){
            return nullptr;
        }
        return static_cast<Dtype*>(_buf->get_data_mutable()) + start_index() + index;
    }

    /**
     *  \brief Return tensor data pointer, with data type of current tensor (Dtype*).
     */
    const Dtype * data(int index = 0) const {
        // synchronize the events tree
        //sync();
        CHECK_EQ(device_id(), API::get_device_id()) << \
            "tensor is not declared in current device";
        if (_buf->get_capacity() == 0){
            return nullptr;
        }
        return static_cast<const Dtype*>(_buf->get_data()) + start_index() + index;
    }

    /**
     *  \brief Share from same layout_type and same date type tensor,
     *  if shared tensor target is the same with current tensor target, buffer is shared;
     *  otherwise, tensor buffer is deep copied.
     *  only shared buffer ptr, current tensor will have continuous memory,
     *  only if current shape and valid shape are the same, and offset is all set to 0.
     */
    //template <typename Tensor1,
    //    class = typename std::enable_if<std::is_same<layout_type, typename TensorTraits<Tensor1>::layout_type>::value>::type>
        //class = typename std::enable_if<std::is_same<layout_type, typename TensorTraits<Tensor1>::layout_type>::value>::type >
    template <typename Tensor_t>
    SaberStatus share_from(const Tensor_t& tensor) {

        CHECK_EQ(_shape > Shape::zero(TensorAPI::layout_dims::value), true) << \
            "current tensor is not initialized (no shape info, use set_shape)";
        typedef typename Tensor_t::Dtype dtype_t;
        CHECK_LE(size() * _type_len, tensor.size() * sizeof(dtype_t)) << \
            "current tensor size should <= input tensor size";

        _is_shared = BufferMemShare(_buf, tensor.get_buf()) > 0;
        _is_subbuf = false;
        _seq_offset = tensor.get_seq_offset();
        //if(shared){
        //    _is_root = false;
        //    tensor.add_events((EventsTree<TargetType_t>*)(&_events_tree));
        //} else{
        //    _is_root = true;
        //}
        return SaberSuccess;
    }
    std::vector<std::vector<int>> get_seq_offset() const {return _seq_offset;}
    SaberStatus set_seq_offset(std::vector<std::vector<int>> seq_offset) {_seq_offset = seq_offset; return SaberSuccess;}

    SaberStatus share_sub_buffer(const Tensor<TargetType, datatype, LayOutType>& tensor, \
        Shape valid_shape, Shape offset) {

        //if (valid_shape.dims() != TensorAPI::layout_dims::value \
            || offset.dims() != TensorAPI::layout_dims::value || \
            !((offset + valid_shape) <= tensor.shape())) { \
            return SaberInvalidValue; \
        }
        CHECK_EQ(valid_shape.dims(), TensorAPI::layout_dims::value) << \
            "shape dims is not matched to layout type";
        CHECK_EQ(offset.dims(), TensorAPI::layout_dims::value) << \
            "shape dims is not matched to layout type";
        CHECK_EQ(true, (offset + valid_shape) <= tensor.shape()) << \
            "offset + valid_shape <= shape";
        _valid_shape = valid_shape;
        _offset = offset;
        _shape = tensor.shape();
        _buf = tensor.get_buf();
        _is_subbuf = true;
        _is_shared = true;
        _seq_offset = tensor.get_seq_offset();
        return SaberSuccess;
    }

    /**
     *  \brief Deep copy data within region of interest from input tensor.
     */
    template <typename TargetType_t, typename LayOutType_t>
    SaberStatus copy_from(const Tensor<TargetType_t, datatype, LayOutType_t>& tensor) {
        //if (valid_size() != tensor.valid_size()) { \
            return SaberInvalidValue; \
        }
        CHECK_EQ(valid_size(), tensor.valid_size()) \
            << "sizes of two valid shapes must be the same";

        /// get the proper process target wrapper
        typedef  TargetWrapper<TargetType_t> API_t;
        typedef typename TargetTypeTraits<TargetType_t>::target_type target_type_t;
        typedef typename IF<std::is_same<target_type, target_type_t>::value, __HtoH, __DtoH>::Type then_type;
        typedef typename IF<std::is_same<target_type, target_type_t>::value, __DtoD, __HtoD>::Type else_type;
        typedef typename IF<std::is_same<target_category, __host_target>::value, then_type, else_type>::Type flag_type;
        typedef typename IF<std::is_same<target_category , __host_target>::value, API_t, API>::Type process_API;

        /// return if src and dst data ptrs are the same
        if (data() == tensor.data()){
            return SaberSuccess;
        }

        /// both tensors are continuous, copy entire buffer
        if (is_continue_mem() && tensor.is_continue_mem()) {
            Dtype* ptr_dst = mutable_data();
            const Dtype* ptr_src = tensor.data();
            process_API::sync_memcpy(ptr_dst, device_id(), ptr_src, tensor.device_id(), \
                _type_len * valid_size(), flag_type());
            return SaberSuccess;
        }

        Shape sh_dst = _shape;
        Shape val_sh_dst = _valid_shape;
        Shape sh_src = tensor.shape();
        Shape val_sh_src = tensor.valid_shape();
        //Shape off_dst = _offset;
        //Shape off_src = tensor.offset();

        if (is_continue_mem()) {
            sh_dst = _valid_shape;
        }
        if (tensor.is_continue_mem()) {
            sh_src = val_sh_src;
        }

        int dim_dst = dims();
        int dim_src = tensor.dims();

        /// check the beginning axis of dis_continue memory
        int axis_discontinue_dst = -1;
        int axis_discontinue_src = -1;
        for (int i = dim_dst - 1; i >= 0; i--) {
            if (val_sh_dst[i] == sh_dst[i]) {
                continue;
            } else {
                axis_discontinue_dst = i;
                break;
            }
        }
        for (int i = dim_src - 1; i >= 0; i--) {
            if (val_sh_src[i] == sh_src[i]) {
                continue;
            } else {
                axis_discontinue_src = i;
                break;
            }
        }
        //printf("dst axis=%d, src axis=%d\n", axis_discontinue_dst, axis_discontinue_src);

        /// only copy the region of interest
        /// compute the copy length of each memcpy
        int cpy_len_dst = 1;
        int cpy_len_src = 1;
        if (axis_discontinue_dst < 0){
            cpy_len_dst = valid_size();
        } else{
            for (int i = axis_discontinue_dst; i < dim_dst; i++) {
                cpy_len_dst *= val_sh_dst[i];
            }
        }
        if (axis_discontinue_src < 0){
            cpy_len_src = tensor.valid_size();
        } else{
            for (int i = axis_discontinue_src; i < dim_src; i++) {
                cpy_len_src *= val_sh_src[i];
            }
        }
        //printf("cpy_len_dst=%d, %d, cpy_len_src=%d, %d\n", cpy_len_dst, valid_size(), cpy_len_src, tensor.valid_size());
        int cpy_len = cpy_len_dst < cpy_len_src? cpy_len_dst : cpy_len_src;

        /// compute the total copy times
        int cpy_num = valid_size() / cpy_len;
        //printf("cpy_len=%d, cpy_num=%d\n", cpy_len, cpy_num);

        /// compute the stride and start index of dst buffer and src buffer
        std::vector<int> count_dst(abs(axis_discontinue_dst) + 1);
        std::vector<int> count_src(abs(axis_discontinue_src) + 1);

        Shape stride_dst = get_stride();
        Shape stride_src = tensor.get_stride();

        count_dst[abs(axis_discontinue_dst)] = count_src[abs(axis_discontinue_src)] = 1;
        for (int i = axis_discontinue_dst - 1; i >= 0; i--) {
            if (i == axis_discontinue_dst - 1){
                count_dst[i] = 1;
            } else{
                count_dst[i] = val_sh_dst[i + 1] * count_dst[i + 1];
            }
        }
        for (int i = axis_discontinue_src - 1; i >= 0; i--) {
            if (i == axis_discontinue_src - 1){
                count_src[i] = 1;
            } else{
                count_src[i] = val_sh_src[i + 1] * count_src[i + 1];
            }
        }

        /// compute the start position of each buffer, memcpy from src to dst
        int ratio_dst = cpy_len_dst / cpy_len;
        int ratio_src = cpy_len_src / cpy_len;

        Dtype* dst = mutable_data();
        const Dtype* src = tensor.data();

        for (int i = 0; i < cpy_num; ++i) {
            int idx_dst = (i % ratio_dst) * cpy_len;//off_dst[abs(axis_discontinue_dst)] * \
                stride_dst[abs(axis_discontinue_dst)] + (i % ratio_dst) * cpy_len;
            int res_dst = i / ratio_dst;
            for (int j = 0; j < axis_discontinue_dst; ++j) {
                int div = res_dst / count_dst[j];
                idx_dst += (div /*+ off_dst[j]*/) * stride_dst[j];
                res_dst = res_dst % count_dst[j];
            }
            int idx_src = (i % ratio_src) * cpy_len;//off_src[abs(axis_discontinue_src)] * \
                stride_src[abs(axis_discontinue_src)] + (i % ratio_src) * cpy_len;
            int res_src = i / ratio_src;
            for (int j = 0; j < axis_discontinue_src; ++j) {
                int div = res_src / count_src[j];
                idx_src += (div /*+ off_src[j]*/) * stride_src[j];
                res_src = res_src % count_src[j];
            }
            //printf("i: %d, idx_src: %d, idx_dst: %d\n", i, idx_src, idx_dst);
            Dtype* ptr_dst = dst + idx_dst;//_buf->get_data_mutable() + idx_dst;
            const Dtype* ptr_src = src + idx_src;//tensor.get_buf()->get_data() + idx_src;
            process_API::sync_memcpy(ptr_dst, device_id(), ptr_src, tensor.device_id(), \
                _type_len * cpy_len, flag_type());
        }
        return SaberSuccess;
    }

    /**
     * \brief Asynchronously copy entire buffer from source tensor.
     */
    template <typename TargetType_t, typename LayOutType_t, typename stream_type \
        = typename IF<std::is_same<typename TargetTypeTraits<TargetType>::target_category, __host_target>::value, \
        typename TargetWrapper<TargetType_t>::stream_t, typename TargetWrapper<TargetType>::stream_t>::Type>
    SaberStatus async_copy_from(const Tensor<TargetType_t, datatype, LayOutType_t>& tensor, \
        stream_type stream) {
        CHECK_EQ(valid_size() == tensor.valid_size(), true) \
            << "input tensor size should equal to this tensor size";

        /// get the proper process target wrapper
        typedef  TargetWrapper<TargetType_t> API_t;
        typedef typename TargetTypeTraits<TargetType_t>::target_type target_type_t;
        typedef typename IF<std::is_same<target_type, target_type_t>::value, __HtoH, __DtoH>::Type then_type;
        typedef typename IF<std::is_same<target_type, target_type_t>::value, __DtoD, __HtoD>::Type else_type;
        typedef typename IF<std::is_same<target_category, __host_target>::value, then_type, else_type>::Type flag_type;
        typedef typename IF<std::is_same<target_category , __host_target>::value, API_t, API>::Type process_API;

        /// return if src and dst data ptrs are the same
        if (data() == tensor.data()){
            return SaberSuccess;
        }

        /// both tensors are continuous, copy entire buffer
        if (is_continue_mem() && tensor.is_continue_mem()) {
            Dtype* ptr_dst = mutable_data();
            const Dtype* ptr_src = tensor.data();
            process_API::async_memcpy(ptr_dst, device_id(), ptr_src, tensor.device_id(), \
                _type_len * valid_size(), stream, flag_type());
            return SaberSuccess;
        }

        Shape sh_dst = _shape;
        Shape val_sh_dst = _valid_shape;
        Shape sh_src = tensor.shape();
        Shape val_sh_src = tensor.valid_shape();
        Shape off_dst = _offset;
        Shape off_src = tensor.offset();

        if (is_continue_mem()) {
            sh_dst = _valid_shape;
        }
        if (tensor.is_continue_mem()) {
            sh_src = val_sh_src;
        }

        int dim_dst = dims();
        int dim_src = tensor.dims();

        /// check the beginning axis of dis_continue memory
        int axis_discontinue_dst = -1;
        int axis_discontinue_src = -1;
        for (int i = dim_dst - 1; i >= 0; i--) {
            if (val_sh_dst[i] == sh_dst[i]) {
                continue;
            } else {
                axis_discontinue_dst = i;
                break;
            }
        }
        for (int i = dim_src - 1; i >= 0; i--) {
            if (val_sh_src[i] == sh_src[i]) {
                continue;
            } else {
                axis_discontinue_src = i;
                break;
            }
        }
        //printf("dst axis=%d, src axis=%d\n", axis_discontinue_dst, axis_discontinue_src);

        /// Only copy the region of interest.
        /// Compute the copy length of each memcpy.
        int cpy_len_dst = 1;
        int cpy_len_src = 1;
        if (axis_discontinue_dst < 0){
            cpy_len_dst = valid_size();
        } else{
            for (int i = axis_discontinue_dst; i < dim_dst; i++) {
                cpy_len_dst *= val_sh_dst[i];
            }
        }
        if (axis_discontinue_src < 0){
            cpy_len_src = tensor.valid_size();
        } else{
            for (int i = axis_discontinue_src; i < dim_src; i++) {
                cpy_len_src *= val_sh_src[i];
            }
        }
        //printf("cpy_len_dst=%d, %d, cpy_len_src=%d, %d\n", cpy_len_dst, valid_size(), cpy_len_src, tensor.valid_size());
        int cpy_len = cpy_len_dst < cpy_len_src? cpy_len_dst : cpy_len_src;

        /// Compute the total copy times.
        int cpy_num = valid_size() / cpy_len;
        //printf("cpy_len=%d, cpy_num=%d\n", cpy_len, cpy_num);

        /// Compute the stride and start index of dst buffer and src buffer.
        std::vector<int> count_dst(abs(axis_discontinue_dst) + 1);
        std::vector<int> count_src(abs(axis_discontinue_src) + 1);

        Shape stride_dst = get_stride();
        Shape stride_src = tensor.get_stride();

        count_dst[abs(axis_discontinue_dst)] = count_src[abs(axis_discontinue_src)] = 1;
        for (int i = axis_discontinue_dst - 1; i >= 0; i--) {
            if (i == axis_discontinue_dst - 1){
                count_dst[i] = 1;
            } else{
                count_dst[i] = val_sh_dst[i + 1] * count_dst[i + 1];
            }
        }
        for (int i = axis_discontinue_src - 1; i >= 0; i--) {
            if (i == axis_discontinue_src - 1){
                count_src[i] = 1;
            } else{
                count_src[i] = val_sh_src[i + 1] * count_src[i + 1];
            }
        }

        /// Compute the start position of each buffer, memcpy from src to dst.
        int ratio_dst = cpy_len_dst / cpy_len;
        int ratio_src = cpy_len_src / cpy_len;

        Dtype* dst = mutable_data();
        const Dtype* src = tensor.data();

        for (int i = 0; i < cpy_num; ++i) {
            int idx_dst = (i % ratio_dst) * cpy_len;//off_dst[abs(axis_discontinue_dst)] * \
                stride_dst[abs(axis_discontinue_dst)] + (i % ratio_dst) * cpy_len;
            int res_dst = i / ratio_dst;
            for (int j = 0; j < axis_discontinue_dst; ++j) {
                int div = res_dst / count_dst[j];
                idx_dst += (div /*+ off_dst[j]*/) * stride_dst[j];
                res_dst = res_dst % count_dst[j];
            }
            int idx_src = (i % ratio_src) * cpy_len;//off_src[abs(axis_discontinue_src)] * \
                stride_src[abs(axis_discontinue_src)] + (i % ratio_src) * cpy_len;
            int res_src = i / ratio_src;
            for (int j = 0; j < axis_discontinue_src; ++j) {
                int div = res_src / count_src[j];
                idx_src += (div /*+ off_src[j]*/) * stride_src[j];
                res_src = res_src % count_src[j];
            }
            //printf("i: %d, idx_src: %d, idx_dst: %d\n", i, idx_src, idx_dst);
            Dtype* ptr_dst = dst + idx_dst;//_buf->get_data_mutable() + idx_dst;
            const Dtype* ptr_src = src + idx_src;//tensor.get_buf()->get_data() + idx_src;
            process_API::async_memcpy(ptr_dst, device_id(), ptr_src, tensor.device_id(), \
                _type_len * cpy_len, stream, flag_type());
        }
        return SaberSuccess;
    }

    /**
     *  \brief Add events when tensor is shared to others.
     */
    void add_events(EventsTree<TargetType>* events) {
        _events_tree.insert_children(events);
    }

    /**
     *  \brief Synchronize the event tree, wait util all events are done.
     */
    void sync() {
        _events_tree.sync_tree();
    }

    /**
     *  \brief record Event to current tensor.
     *  \param stream  Input processing stream.
     */
    void record_event(typename API::stream_t stream) {
        _events_tree._events.record(stream);
    }


private:
    ///< Length of datatype.
    size_t _type_len{sizeof(Dtype)};
    ///< Represent the raw mem shape.
    Shape _shape;
    ///< Represent the mem you have right to access shape.
    Shape _valid_shape;
    ///< Represent the offset idx between _shape and _real_shape.
    Shape _offset;
    ///< Buffer shared ptr, hold the data pointer, and buffer capacity.
    std::shared_ptr<Buffer<TargetType>> _buf{nullptr};
    ///< Events tree, to synchronize the tensor.
    EventsTree<TargetType> _events_tree;
    ///< share sub-buffer flag.
    bool _is_subbuf{false};
    bool _is_shared{false};

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

    std::vector<std::vector<int>> _seq_offset;
};

} //namespace saber

} //namespace anakin


#endif //ANAKIN_SABER_CORE_TENSOR_H

