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

#ifndef ANAKIN_SABER_CORE_TENSOR_H
#define ANAKIN_SABER_CORE_TENSOR_H

#include "saber/core/shape.h"
#include "saber/core/events.h"
#include "saber/core/buffer.h"

namespace anakin{

namespace saber{

template<typename TargetType>
class Tensor {
public:

    typedef typename DataTrait<TargetType, AK_FLOAT>::Dtype FDtype;
    typedef typename DataTrait<TargetType, AK_FLOAT>::PtrDtype PtrFDtype;
    typedef typename DataTraitLp<TargetType>::PtrDtype PtrLpDtype;
    typedef typename DataTraitBase<TargetType>::PtrDtype BaseDtype;

    typedef typename TargetTypeTraits<TargetType>::target_category target_category;
    typedef typename TargetTypeTraits<TargetType>::target_type target_type;
    typedef TargetWrapper<TargetType> API;

    /**`
     *  \brief Default constructor
     */
    Tensor(size_t lptype_len = 1) : _valid_shape(), _shape(), _offset(), \
        _ftype_len(sizeof(FDtype)), _lptype_len(lptype_len) {
        _fbuf = std::make_shared<Buffer<TargetType>>();
        _lpbuf = std::make_shared<Buffer<TargetType>>();
        _state = DSYNC;
        _is_subbuf = false;
    }

    /**
     * \brief Constructor with shape, memory is alloced according to shape.
     */
    Tensor(Shape shape, size_t type_len = 1, bool flag_create_lp = false) : _ftype_len(sizeof(FDtype)), \
        _lptype_len(type_len), _has_lp_buf(flag_create_lp) {
        _shape = shape;
        _valid_shape = shape;
        _offset = Shape::zero(shape);
        _fbuf = std::make_shared<Buffer<TargetType>>(shape.count() * _ftype_len);
        if (_has_lp_buf) {
            _lpbuf = std::make_shared<Buffer<TargetType>>(shape.count() * _lptype_len);
            _state = DSYNC;
        } else {
            _lpbuf = std::make_shared<Buffer<TargetType>>();
            _state = DFP32;
        }
        _is_shared = false;
        _is_subbuf = false;
    }

    /**
     * \brief Constructor with allocated data ptr and entire memory shape.
     */
    //! now only support fp32 data pointer
    template <typename TargetType_t>
    Tensor(void* data_ptr, TargetType_t target, int id, Shape shape, size_t type_len = 1, bool flag_create_lp = false) : \
        _ftype_len(sizeof(FDtype)), _has_lp_buf(flag_create_lp) {

        _shape = shape;
        _valid_shape = shape;
        _offset = Shape::zero(shape);
        std::shared_ptr<Buffer<TargetType_t>> buf_from_date = \
            std::make_shared<Buffer<TargetType_t>>(data_ptr, shape.count() * _ftype_len, id);
        BufferMemShare(_fbuf, buf_from_date);
        if (_has_lp_buf) {
            _lpbuf = std::make_shared<Buffer<TargetType>>(shape.count() * _lptype_len);
        } else {
            _lpbuf = std::make_shared<Buffer<TargetType>>();
        }
        _state = DFP32;
        _is_shared = true;
        _is_subbuf = false;
    }

    /**
     * \brief Copy constructor, shallow copy.
     */
    Tensor(const Tensor<TargetType>& tensor) : _ftype_len(sizeof(FDtype)) {
        _shape = tensor._shape;
        _valid_shape = tensor._valid_shape;
        _offset = tensor._offset;
        _fbuf = tensor._fbuf;
        _is_subbuf = tensor._is_subbuf;
        _is_shared = tensor._is_shared;
        _seq_offset = tensor._seq_offset;

        _state = tensor._state;
        _scale = tensor._scale;
        _lpbuf = tensor._lpbuf;
        _lptype_len = tensor._lptype_len;
        _has_lp_buf = tensor._has_lp_buf;
    }

    /**
     * \brief Copy constructor without events control.
     */
    Tensor(Tensor<TargetType>& tensor) : _ftype_len(sizeof(FDtype)) {
        _shape = tensor._shape;
        _valid_shape = tensor._valid_shape;
        _offset = tensor._offset;
        _fbuf = tensor._fbuf;
        tensor.add_events(&_events_tree);
        _is_subbuf = tensor._is_subbuf;
        _is_shared = tensor._is_shared;
        _seq_offset = tensor._seq_offset;

        _state = tensor._state;
        _scale = tensor._scale;
        _lptype_len = tensor._lptype_len;
        _lpbuf = tensor._lpbuf;
        _has_lp_buf = tensor._has_lp_buf;
    }

    /**
     * \brief create tensor with buffer
     * @param shape
     * @param type_len
     * @param flag_create_lp
     */
    void create(Shape shape, size_t type_len = 1, bool flag_create_lp = false) {
        _lptype_len = type_len;
        _has_lp_buf = flag_create_lp;
        _shape = shape;
        _valid_shape = shape;
        _offset = Shape::zero(shape);
        _fbuf = std::make_shared<Buffer<TargetType>>(shape.count() * _ftype_len);
        if (_has_lp_buf) {
            _lpbuf = std::make_shared<Buffer<TargetType>>(shape.count() * _lptype_len);
            _state = DSYNC;
        } else {
            _lpbuf = std::make_shared<Buffer<TargetType>>();
            _state = DFP32;
        }
        _is_shared = false;
        _is_subbuf = false;
    }

    /**
     * \brief set scale for different precision data convert
     * @param scale
     */
    void set_scale(std::vector<FDtype> scale) {
        _scale = scale;
    }

    /**
     * \brief get scale
     * @param scale
     */
    std::vector<FDtype> get_scale() const {
        return _scale;
    }

    /**
     *  \brief only change the shape and valid shape, do nothing to memory
     *  \param shape
     *  \param valid_shape
     *  \param offset
     */
    SaberStatus set_shape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape()) {

        if (shape.dims() > 0) {
            _shape = shape;
        }
        if (offset.dims() > 0 && _is_subbuf) {
            _offset = offset;
        }
        CHECK_EQ(valid_shape > Shape::zero(valid_shape), true) << "valid_shape size should > 0";
        _valid_shape = valid_shape;

        if (!_is_subbuf) {
            if (_shape.count() <= _valid_shape.count()) {
                _shape = _valid_shape;
            }
            _offset = Shape::zero(valid_shape);
        } else {
            if (_shape == Shape::zero(_valid_shape)) {
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
        CHECK_EQ(_is_shared || _is_subbuf, false) << "shared tensor could not re_alloc";
        _shape = shape;
        _valid_shape = _shape;
        _offset =Shape::zero(_shape);
        _fbuf->alloc(_shape.count() * _ftype_len);
        if (_has_lp_buf) {
            _lpbuf->alloc(_shape.count() * _lptype_len);
        }
        return SaberSuccess;
    }

    /**
     *  \brief Change tensor shape,
     *  if input shape's count is bigger than the capacity of buffer, alloc a new buffer.
     */
    SaberStatus reshape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape()) {

        if (shape.dims() > 0) {
            _shape = shape;
        }
        if (offset.dims() > 0 && _is_subbuf) {
            _offset = offset;
        }
        CHECK_EQ(valid_shape > Shape::zero(valid_shape), true) << \
            "valid_shape size should > 0";
        _valid_shape = valid_shape;

        if (!_is_subbuf) {
            if (_shape.count() < _valid_shape.count()) {
                _shape = _valid_shape;
            }
            _offset = Shape::zero(_valid_shape);
        } else {
            if (_shape == Shape::zero(valid_shape)) {
                _shape = valid_shape;
            }
            //if (!(_valid_shape + _offset <= _shape)) { \
                return SaberInvalidValue; \
            }
            CHECK_EQ(_valid_shape + _offset <= _shape, true) << \
                "valid_shape + offet should <= shape";
        }
        bool exceed_flag = _shape.count() * _ftype_len > _fbuf->get_capacity() \
            && (_is_subbuf || _is_shared);
        //if (exceed_flag) {
        //    return SaberOutOfAuthority;
        //}
        CHECK_EQ(exceed_flag, false) << "shared tensor shape exceed origin data buffer size";
        SABER_CHECK(_fbuf->re_alloc(_shape.count() * _ftype_len));
        if (_has_lp_buf) {
            SABER_CHECK(_lpbuf->re_alloc(_shape.count() * _lptype_len));
        }
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
        return _shape.count(start, end);
    }

    /**
     *  \brief return valid_shape count, from start index to end index(end index is excluded).
     *  \param start input start index.
     *  \param end   input end index (exclude in calculation).
     *  \return the size from start index to end index.
     */
    int count_valid(int start, int end) const {
        return _valid_shape.count(start, end);
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
        if (_is_subbuf) {
            return  _shape.get_stride();
        }
        return  _valid_shape.get_stride();
    }

    /**
     *  \brief Return tensor offset, which holds the offset in each dim.
     */
    Shape offset() const {
        return _offset;
    }

    /**
     *  \brief Return valid shape of tensor
     */
    int data_offset() const {
        return start_index();
    }


    /**
     * \brief get sequence offset, lot tensor
     * @return
     */
    std::vector<int> get_seq_offset() const {
        return _seq_offset;
    }

    /**
     * \brief set sequence offset, lot tensor
     * @param seq_offset
     * @return
     */
    SaberStatus set_seq_offset(std::vector<int> seq_offset) {
        _seq_offset = seq_offset;
        return SaberSuccess;
    }

//    /**
//     *  \brief Return reference shared_ptr of tensor.
//     */
//     const std::shared_ptr<Buffer<TargetType>>& get_buf() const {
//         return _fbuf;
//     }
//
//    /**
//     *  \brief Return reference shared_ptr of tensor.
//     */
//    const std::shared_ptr<Buffer<TargetType>>& get_lpbuf() const {
//        return _lpbuf;
//    }

    /**
     *  \brief Return tensor device id.
     */
    int device_id() const {
        return _lpbuf->get_id();
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
    int num_index() const {
        return _valid_shape.num_index();
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
    int channel_index() const {
        return _valid_shape.channel_index();
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
    int height_index() const {
        return _valid_shape.height_index();
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
    int width_index() const {
        return _valid_shape.width_index();
    }

    size_t lptype_len() const {
        return _lptype_len;
    }

    bool has_lpbuf() const {
        return _has_lp_buf;
    }

    /**
     *  \brief Return tensor mutable data pointer void*.
     */
    PtrFDtype mutable_data() {
        // synchronize the events tree
        //sync();
        CHECK_EQ(device_id(), API::get_device_id()) << \
            "tensor is not declared in current device";
        if (_fbuf->get_capacity() == 0){
            return nullptr;
        }

        return static_cast<PtrFDtype>(_fbuf->get_data_mutable());
    }

    /**
     *  \brief Return tensor data pointer, with data type of current tensor (Dtype*).
     */
    const PtrFDtype data() const {
        // synchronize the events tree
        //sync();
        CHECK_EQ(device_id(), API::get_device_id()) << \
            "tensor is not declared in current device";
        if (_fbuf->get_capacity() == 0){
            return nullptr;
        }
        return static_cast<PtrFDtype>(_fbuf->get_data_mutable());
    }

    /**
     *  \brief Return tensor low precision mutable data pointer, with data type of current tensor (Dtype*).
     */
    PtrLpDtype lp_mutable_data() {
        // synchronize the events tree
        //sync();
        CHECK_EQ(device_id(), API::get_device_id()) << \
            "tensor is not declared in current device";
        if (_fbuf->get_capacity() == 0){
            return nullptr;
        }
        return _fbuf->get_data_mutable();
    }

    /**
     *  \brief Return tensor data pointer, with data type of current tensor (Dtype*).
     */
    const PtrLpDtype lp_data() const {
        // synchronize the events tree
        //sync();
                CHECK_EQ(device_id(), API::get_device_id()) << \
            "tensor is not declared in current device";
        if (_fbuf->get_capacity() == 0){
            return nullptr;
        }
        return  _fbuf->get_data_mutable();
    }

    /**
     *  \brief Share from same layout_type and same date type tensor,
     *  if shared tensor target is the same with current tensor target, buffer is shared;
     *  otherwise, tensor buffer is deep copied.
     *  only shared buffer ptr, current tensor will have continuous memory,
     *  only if current shape and valid shape are the same, and offset is all set to 0.
     */
    SaberStatus share_from(const Tensor& tensor) {

        CHECK_LE(size(), tensor.size()) << "current tensor size should <= input tensor size";

        //_is_shared = BufferMemShare(_buf, tensor.get_buf()) > 0;
        _fbuf = tensor._fbuf;
        _is_subbuf = false;
        _seq_offset = tensor._seq_offset;
        _is_shared = true;

        _lpbuf = tensor._lpbuf;

        //if(shared){
        //    _is_root = false;
        //    tensor.add_events((EventsTree<TargetType_t>*)(&_events_tree));
        //} else{
        //    _is_root = true;
        //}
        return SaberSuccess;
    }


    SaberStatus share_sub_buffer(const Tensor& tensor, Shape valid_shape, Shape offset) {

        //if (valid_shape.dims() != TensorAPI::layout_dims::value \
            || offset.dims() != TensorAPI::layout_dims::value || \
            !((offset + valid_shape) <= tensor.shape())) { \
            return SaberInvalidValue; \
        }
        CHECK_EQ(true, (offset + valid_shape) <= tensor.shape()) << \
            "offset + valid_shape <= shape";
        _valid_shape = valid_shape;
        _offset = offset;
        _shape = tensor.shape();
        _fbuf = tensor._fbuf;
        _lpbuf = tensor._lpbuf;
        _is_subbuf = true;
        _is_shared = true;
        _seq_offset = tensor._seq_offset;
        return SaberSuccess;
    }

    /**
     *  \brief Deep copy data within region of interest from input tensor.
     */
    template <typename TargetType_t>
    SaberStatus copy_from(const Tensor<TargetType_t>& tensor) {

        //if (valid_size() != tensor.valid_size()) { \
            return SaberInvalidValue; \
        }
        CHECK_EQ(valid_size(), tensor.valid_size()) \
            << "sizes of two valid shapes must be the same";

        if (_fbuf->get_capacity() == 0) {
            reshape(_valid_shape);
        }

        /// get the proper process target wrapper
        typedef  TargetWrapper<TargetType_t> API_t;
        typedef typename TargetTypeTraits<TargetType_t>::target_type target_type_t;
        typedef typename IF<std::is_same<target_type, target_type_t>::value, __HtoH, __DtoH>::Type then_type;
        typedef typename IF<std::is_same<target_type, target_type_t>::value, __DtoD, __HtoD>::Type else_type;
        typedef typename IF<std::is_same<target_category, __host_target>::value, then_type, else_type>::Type flag_type;
        typedef typename IF<std::is_same<target_category , __host_target>::value, API_t, API>::Type process_API;

        typedef typename DataTrait<TargetType_t, AK_FLOAT>::Dtype FDtype_src;
        typedef typename DataTrait<TargetType_t, AK_FLOAT>::PtrDtype PtrFDtype_src;
        typedef typename DataTraitLp<TargetType_t>::PtrDtype PtrLpDtype_src;
        typedef typename DataTraitBase<TargetType>::PtrDtype BaseDtype_src;

        bool flag_copy_lp = _has_lp_buf && tensor.has_lpbuf();

        /// return if src and dst data ptrs are the same
        if (std::is_same<TargetType, TargetType_t>::value){
            if ((const void*)data() == (const void*)(tensor.data())) {
                return SaberSuccess;
            }
        }

        /// both tensors are continuous, copy entire buffer
        if (is_continue_mem() && tensor.is_continue_mem()) {
            int dst_data_offset = data_offset();
            int src_data_offset = tensor.data_offset();

            BaseDtype ptr_dst = _fbuf->get_data_mutable();
            const PtrFDtype_src ptr_src = tensor.data();

            process_API::sync_memcpy(ptr_dst, _ftype_len * dst_data_offset, device_id(), \
                ptr_src, _ftype_len * src_data_offset, tensor.device_id(), \
                _ftype_len * valid_size(), flag_type());

            if (flag_copy_lp) {
                if (_lpbuf->get_capacity() == 0) {
                }
                BaseDtype ptr_lp_dst = _lpbuf->get_data_mutable();
                const PtrLpDtype_src ptr_lp_src = tensor.lp_data();
                process_API::sync_memcpy(ptr_dst, _lptype_len * dst_data_offset, device_id(), \
                ptr_src, _lptype_len * src_data_offset, tensor.device_id(), \
                _lptype_len * valid_size(), flag_type());
            }

            _state = DSYNC;

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


        int dst_data_offset = data_offset();
        int src_data_offset = tensor.data_offset();

        BaseDtype ptr_dst = _fbuf->get_data_mutable();
        const PtrFDtype_src ptr_src = tensor.data();

        BaseDtype ptr_dst_lp = nullptr;
        PtrLpDtype_src ptr_src_lp = nullptr;

        if (flag_copy_lp) {
            ptr_dst_lp = _lpbuf->get_data_mutable();
            ptr_src_lp = tensor.lp_data();
        }

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

            int cpy_dst_offset = dst_data_offset + idx_dst;
            int cpy_src_offset = src_data_offset + idx_src;

            process_API::sync_memcpy(ptr_dst, _ftype_len * cpy_dst_offset, device_id(), \
                ptr_src, _ftype_len * cpy_src_offset, tensor.device_id(), \
                    _ftype_len * cpy_len, flag_type());

            if (flag_copy_lp) {
                process_API::sync_memcpy(ptr_dst_lp, _lptype_len * cpy_dst_offset, device_id(), \
                    ptr_src_lp, _lptype_len * cpy_src_offset, tensor.device_id(), \
                    _ftype_len * cpy_len, flag_type());
            }
        }
        return SaberSuccess;
    }

    /**
     * \brief Asynchronously copy entire buffer from source tensor.
     */
    template <typename TargetType_t, typename stream_type \
        = typename IF<std::is_same<typename TargetTypeTraits<TargetType>::target_category, __host_target>::value, \
        typename TargetWrapper<TargetType_t>::stream_t, typename TargetWrapper<TargetType>::stream_t>::Type>
    SaberStatus async_copy_from(const Tensor<TargetType_t>& tensor, stream_type stream) {

        CHECK_EQ(valid_size(), tensor.valid_size()) \
            << "sizes of two valid shapes must be the same";

        if (_fbuf->get_capacity() == 0) {
            reshape(_valid_shape);
        }

        /// get the proper process target wrapper
        typedef  TargetWrapper<TargetType_t> API_t;
        typedef typename TargetTypeTraits<TargetType_t>::target_type target_type_t;
        typedef typename IF<std::is_same<target_type, target_type_t>::value, __HtoH, __DtoH>::Type then_type;
        typedef typename IF<std::is_same<target_type, target_type_t>::value, __DtoD, __HtoD>::Type else_type;
        typedef typename IF<std::is_same<target_category, __host_target>::value, then_type, else_type>::Type flag_type;
        typedef typename IF<std::is_same<target_category , __host_target>::value, API_t, API>::Type process_API;

        typedef typename DataTrait<TargetType_t, AK_FLOAT>::Dtype FDtype_src;
        typedef typename DataTrait<TargetType_t, AK_FLOAT>::PtrDtype PtrFDtype_src;
        typedef typename DataTraitLp<TargetType_t>::PtrDtype PtrLpDtype_src;
        typedef typename DataTraitBase<TargetType>::PtrDtype BaseDtype_src;

        bool flag_copy_lp = _has_lp_buf && tensor.has_lpbuf();

        /// return if src and dst data ptrs are the same
        if (std::is_same<TargetType, TargetType_t>::value){
            if ((const void*)data() == (const void*)(tensor.data())) {
                return SaberSuccess;
            }
        }

        /// both tensors are continuous, copy entire buffer
        if (is_continue_mem() && tensor.is_continue_mem()) {
            int dst_data_offset = data_offset();
            int src_data_offset = tensor.data_offset();

            BaseDtype ptr_dst = _fbuf->get_data_mutable();
            const PtrFDtype_src ptr_src = tensor.data();

            process_API::async_memcpy(ptr_dst, _ftype_len * dst_data_offset, device_id(), \
                ptr_src, _ftype_len * src_data_offset, tensor.device_id(), \
                _ftype_len * valid_size(), stream, flag_type());

            if (flag_copy_lp) {
                BaseDtype ptr_lp_dst = _lpbuf->get_data_mutable();
                const PtrLpDtype_src ptr_lp_src = tensor.lp_data();
                process_API::async_memcpy(ptr_dst, _lptype_len * dst_data_offset, device_id(), \
                ptr_src, _lptype_len * src_data_offset, tensor.device_id(), \
                _lptype_len * valid_size(), stream, flag_type());
            }

            _state = DSYNC;

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


        int dst_data_offset = data_offset();
        int src_data_offset = tensor.data_offset();

        BaseDtype ptr_dst = _fbuf->get_data_mutable();
        const PtrFDtype_src ptr_src = tensor.data();

        BaseDtype ptr_dst_lp = nullptr;
        PtrLpDtype_src ptr_src_lp = nullptr;

        if (flag_copy_lp) {
            ptr_dst_lp = _lpbuf->get_data_mutable();
            ptr_src_lp = tensor.lp_data();
        }

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

            int cpy_dst_offset = dst_data_offset + idx_dst;
            int cpy_src_offset = src_data_offset + idx_src;

            process_API::async_memcpy(ptr_dst, _ftype_len * cpy_dst_offset, device_id(), \
                ptr_src, _ftype_len * cpy_src_offset, tensor.device_id(), \
                    _ftype_len * cpy_len, stream, flag_type());

            if (flag_copy_lp) {
                process_API::async_memcpy(ptr_dst_lp, _lptype_len * cpy_dst_offset, device_id(), \
                    ptr_src_lp, _lptype_len * cpy_src_offset, tensor.device_id(), \
                    _ftype_len * cpy_len, stream, flag_type());
            }
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
    //! tensor data type state
    enum DState{
        DSYNC = 0,
        DFP32 = 1,
        DINt8 = 2
    };

    DState _state{DSYNC};

    //! scale for quantization
    std::vector<FDtype> _scale;

    ///< Length of datatype.
    const size_t _ftype_len;
    size_t _lptype_len{1};
    bool _has_lp_buf{false};

    ///< Represent the raw mem shape.
    Shape _shape;
    ///< Represent the mem you have right to access shape.
    Shape _valid_shape;
    ///< Represent the offset idx between _shape and _real_shape.
    Shape _offset;
    ///< Buffer shared ptr, hold the data pointer, and buffer capacity.
    std::shared_ptr<Buffer<TargetType>> _fbuf{nullptr};
    std::shared_ptr<Buffer<TargetType>> _lpbuf{nullptr};
    ///< Events tree, to synchronize the tensor.
    EventsTree<TargetType> _events_tree;
    ///< share sub-buffer flag.
    bool _is_subbuf{false};
    bool _is_shared{false};

    //! lot tensor
    std::vector<int> _seq_offset;

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

} //namespace saber

} //namespace anakin


#endif //ANAKIN_SABER_CORE_TENSOR_H

