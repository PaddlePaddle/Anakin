/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_CORE_TENSOR_TRAITS_H
#define ANAKIN_SABER_CORE_TENSOR_TRAITS_H
#include "core/buffer.h"
#include "core/data_traits.h"

namespace anakin{

namespace saber{

template<typename TargetType, DataType datatype, typename LayeOutType>
class Tensor;

template <typename TensorT>
struct TensorTraits {
    typedef typename TensorT::target_category target_category;
    typedef typename TensorT::target_type target_type;
    typedef typename TensorT::layout_category layout_category;
    typedef typename TensorT::layout_type layout_type;
    using layout_dims = std::integral_constant<int, 0>;
};

// NCHW_C16, the last dim is always 16
template<typename TargetType, DataType datatype>
struct TensorTraits<Tensor<TargetType, datatype, NCHW_C16>>
{
    typedef typename Tensor<TargetType, datatype, NCHW_C16>::target_category  target_category;
    typedef typename Tensor<TargetType, datatype, NCHW_C16>::target_type target_type;
    typedef typename DataTrait<datatype>::dtype Dtype;
    typedef _5D layout_category;
    typedef NCHW_C16 layout_type;
    using layout_dims = std::integral_constant<int, 5>;
    using num_idx = std::integral_constant<int, 0>;
    using channel_idx = std::integral_constant<int, 1>;
    using height_idx = std::integral_constant<int, 2>;
    using width_idx = std::integral_constant<int, 3>;
    using k_idx = std::integral_constant<int, 4>;
    static int num(const Shape& shape) {
        return shape[0];
    }
    static int channel(const Shape& shape) {
        return shape[1] * 16;
    }
    static int height(const Shape& shape) {
        return shape[2];
    }
    static int width(const Shape& shape) {
        return shape[3];
    }
    static int depth(const Shape& shape) {
        return shape[4];
    }
};

// NCHW_C8, the last dim is always 8
template<typename TargetType, DataType datatype>
struct TensorTraits<Tensor<TargetType, datatype, NCHW_C8>>
{
    typedef typename Tensor<TargetType, datatype, NCHW_C8>::target_category  target_category;
    typedef typename Tensor<TargetType, datatype, NCHW_C8>::target_type target_type;
    typedef typename DataTrait<datatype>::dtype Dtype;
    typedef _5D layout_category;
    typedef NCHW_C8 layout_type;
    using layout_dims = std::integral_constant<int, 5>;
    using num_idx = std::integral_constant<int, 0>;
    using channel_idx = std::integral_constant<int, 1>;
    using height_idx = std::integral_constant<int, 2>;
    using width_idx = std::integral_constant<int, 3>;
    using k_idx = std::integral_constant<int, 8>;
    static int num(const Shape& shape) {
        return shape[0];
    }
    static int channel(const Shape& shape) {
        return shape[1] * 8;
    }
    static int height(const Shape& shape) {
        return shape[2];
    }
    static int width(const Shape& shape) {
        return shape[3];
    }
    static int depth(const Shape& shape) {
        return shape[4];
    }
};

// NCHW_C4, the last dim is always 4
template<typename TargetType, DataType datatype>
struct TensorTraits<Tensor<TargetType, datatype, NCHW_C4>>
{
    typedef typename Tensor<TargetType, datatype, NCHW_C4>::target_category  target_category;
    typedef typename Tensor<TargetType, datatype, NCHW_C4>::target_type target_type;
    typedef typename DataTrait<datatype>::dtype Dtype;
    typedef _5D layout_category;
    typedef NCHW_C4 layout_type;
    using layout_dims = std::integral_constant<int, 5>;
    using num_idx = std::integral_constant<int, 0>;
    using channel_idx = std::integral_constant<int, 1>;
    using height_idx = std::integral_constant<int, 2>;
    using width_idx = std::integral_constant<int, 3>;
    using k_idx = std::integral_constant<int, 4>;
    static int num(const Shape& shape) {
        return shape[0];
    }
    static int channel(const Shape& shape) {
        return shape[1] * 4;
    }
    static int height(const Shape& shape) {
        return shape[2];
    }
    static int width(const Shape& shape) {
        return shape[3];
    }
    static int depth(const Shape& shape) {
        return shape[4];
    }
};

template<typename TargetType, DataType datatype>
struct TensorTraits<Tensor<TargetType, datatype, NCHW>>
{
    typedef typename Tensor<TargetType, datatype, NCHW>::target_category  target_category;
    typedef typename Tensor<TargetType, datatype, NCHW>::target_type target_type;
    typedef typename DataTrait<datatype>::dtype Dtype;
    typedef _4D layout_category;
    typedef NCHW layout_type;
    using layout_dims = std::integral_constant<int, 4>;
    using num_idx = std::integral_constant<int, 0>;
    using channel_idx = std::integral_constant<int, 1>;
    using height_idx = std::integral_constant<int, 2>;
    using width_idx = std::integral_constant<int, 3>;
    static int num(const Shape& shape){
        return shape[0];
    }
    static int channel(const Shape& shape){
        return shape[1];
    }
    static int height(const Shape& shape){
        return shape[2];
    }
    static int width(const Shape& shape){
        return shape[3];
    }
};

template<typename TargetType, DataType datatype>
struct TensorTraits<Tensor<TargetType, datatype, NHWC>>
{
    typedef typename Tensor<TargetType, datatype, NHWC>::target_category  target_category;
    typedef typename Tensor<TargetType, datatype, NHWC>::target_type target_type;
    typedef _4D layout_category;
    typedef NHWC layout_type;
    using layout_dims = std::integral_constant<int, 4>;
    using num_idx = std::integral_constant<int, 0>;
    using channel_idx = std::integral_constant<int, 3>;
    using height_idx = std::integral_constant<int, 1>;
    using width_idx = std::integral_constant<int, 2>;
    static int num(const Shape& shape){
        return shape[0];
    }
    static int channel(const Shape& shape){
        return shape[3];
    }
    static int height(const Shape& shape){
        return shape[1];
    }
    static int width(const Shape& shape){
        return shape[2];
    }
};

template<typename TargetType, DataType datatype>
struct TensorTraits<Tensor<TargetType, datatype, NHW>>
{
    typedef typename Tensor<TargetType, datatype, NHW>::target_category  target_category;
    typedef typename Tensor<TargetType, datatype, NHW>::target_type target_type;
    typedef _3D layout_category;
    typedef NHW layout_type;
    using layout_dims = std::integral_constant<int, 3>;
    using num_idx = std::integral_constant<int, 0>;
    using channel_idx = std::integral_constant<int, -1>;
    using height_idx = std::integral_constant<int, 1>;
    using width_idx = std::integral_constant<int, 2>;
    static int num(const Shape& shape){
        return shape[0];
    }
    static int channel(const Shape& shape){
        return 1;
    }
    static int height(const Shape& shape){
        return shape[1];
    }
    static int width(const Shape& shape){
        return shape[2];
    }
};

template<typename TargetType, DataType datatype>
struct TensorTraits<Tensor<TargetType, datatype, NW>>
{
    typedef typename Tensor<TargetType, datatype, NW>::target_category  target_category;
    typedef typename Tensor<TargetType, datatype, NW>::target_type target_type;
    typedef _2D layout_category;
    typedef NW layout_type;
    using layout_dims = std::integral_constant<int, 2>;
    using num_idx = std::integral_constant<int, 0>;
    using channel_idx = std::integral_constant<int, -1>;
    using height_idx = std::integral_constant<int, -1>;
    using width_idx = std::integral_constant<int, 1>;
    static int num(const Shape& shape){
        return shape[0];
    }
    static int channel(const Shape& shape){
        return 1;
    }
    static int height(const Shape& shape){
        return 1;
    }
    static int width(const Shape& shape){
        return shape[2];
    }
};

template<typename TargetType, DataType datatype>
struct TensorTraits<Tensor<TargetType, datatype, HW>>
{
    typedef typename Tensor<TargetType, datatype, HW>::target_category  target_category;
    typedef typename Tensor<TargetType, datatype, HW>::target_type target_type;
    typedef _2D layout_category;
    typedef HW layout_type;
    using layout_dims = std::integral_constant<int, 2>;
    using num_idx = std::integral_constant<int, -1>;
    using channel_idx = std::integral_constant<int, -1>;
    using height_idx = std::integral_constant<int, 0>;
    using width_idx = std::integral_constant<int, 1>;
    static int num(const Shape& shape){
        return 1;
    }
    static int channel(const Shape& shape){
        return 1;
    }
    static int height(const Shape& shape){
        return shape[0];
    }
    static int width(const Shape& shape){
        return shape[1];
    }
};

template<typename TargetType, DataType datatype>
struct TensorTraits<Tensor<TargetType, datatype, W>>
{
    typedef typename Tensor<TargetType, datatype, W>::target_category  target_category;
    typedef typename Tensor<TargetType, datatype, W>::target_type target_type;
    typedef _1D layout_category;
    typedef HW layout_type;
    using layout_dims = std::integral_constant<int, 1>;
    using num_idx = std::integral_constant<int, -1>;
    using channel_idx = std::integral_constant<int, -1>;
    using height_idx = std::integral_constant<int, -1>;
    using width_idx = std::integral_constant<int, 1>;
    static int num(const Shape& shape){
        return 1;
    }
    static int channel(const Shape& shape){
        return 1;
    }
    static int height(const Shape& shape){
        return 1;
    }
    static int width(const Shape& shape){
        return shape[0];
    }
};

template <typename TargetType_dst, typename TargetType_src>
static inline int MemShare(std::shared_ptr<Buffer<TargetType_dst>>& dst, \
    const std::shared_ptr<Buffer<TargetType_src>>& src, __DtoD) {
    //LOG(INFO) << "shared D2D";
    if(dst->get_id() == src->get_id()){
        dst = src;
        return 1;
    }
    //LOG(INFO) << "copied D2D";
    SABER_CHECK(dst->re_alloc(src->get_count()));
    SABER_CHECK(dst->sync_copy_from(*src));
    return 0;
}

template <typename TargetType_dst, typename TargetType_src>
static inline int MemShare(std::shared_ptr<Buffer<TargetType_dst>>& dst, \
    const std::shared_ptr<Buffer<TargetType_src>>& src, __HtoD) {
    //LOG(INFO) << "copied H2D";
    SABER_CHECK(dst->re_alloc(src->get_count()));
    SABER_CHECK(dst->sync_copy_from(*src));
    return 0;
}

template <typename TargetType_dst, typename TargetType_src>
static inline int MemShare(std::shared_ptr<Buffer<TargetType_dst>>& dst, \
    const std::shared_ptr<Buffer<TargetType_src>>& src, __HtoH) {
    //LOG(INFO) << "shared H2H";
    dst = src;
    return 1;
}

template <typename TargetType_dst, typename TargetType_src>
static inline int MemShare(std::shared_ptr<Buffer<TargetType_dst>>& dst, \
    const std::shared_ptr<Buffer<TargetType_src>>& src, __DtoH) {
    //LOG(INFO) << "copied D2H";
    SABER_CHECK(dst->re_alloc(src->get_count()));
    SABER_CHECK(dst->sync_copy_from(*src));
    return 0;
}

template <typename TargetType_dst, typename TargetType_src>
static inline int BufferMemShare(std::shared_ptr<Buffer<TargetType_dst>>& dst, \
    const std::shared_ptr<Buffer<TargetType_src>>& src){

    typedef typename TargetTypeTraits<TargetType_dst>::target_type target_type_dst;
    typedef typename TargetTypeTraits<TargetType_src>::target_type target_type_src;
    typedef typename TargetTypeTraits<TargetType_dst>::target_category target_category_dst;

    typedef typename IF<std::is_same<target_type_dst, target_type_src>::value, __HtoH, __DtoH>::Type then_type;
    typedef typename IF<std::is_same<target_type_dst, target_type_src>::value, __DtoD, __HtoD>::Type else_type;
    typedef typename IF<std::is_same<target_category_dst, __host_target>::value, then_type, else_type>::Type flag_type;
    CHECK_EQ(src == nullptr, false) << "input buffer is null!";
    if (!dst){
        dst = std::make_shared<Buffer<TargetType_dst>>(src->get_count());
    }
    return MemShare(dst, src, flag_type());
}

} //names

} //namespace anakin

#endif //ANAKIN_SABER_CORE_TENSOR_TRAITS_H
