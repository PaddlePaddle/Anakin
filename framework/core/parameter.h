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

#ifndef ANAKIN_PARAMETER_H
#define ANAKIN_PARAMETER_H 

#include <vector>
#include <type_traits>
#include "framework/core/types.h"
#include "framework/core/type_traits_extend.h"
#include <utility>
#include "utils/logger/logger.h"
#include "saber/saber_types.h"
#include "saber/core/tensor.h"
#include "saber/core/context.h"

namespace anakin {

using namespace saber;

/// Basic type define comes from lib saber by hac-sys-baidu.
#ifndef USE_SABER
#define USE_SABER
template<typename Ttype, DataType Dtype>
using Tensor4d = Tensor<Ttype, Dtype>;/// saber::Tensor<4, Ttype, Dtype, NCHW>;
/// Global type to tensor pointer.
template<typename Ttype, DataType Dtype>
using Tensor4dPtr = Tensor4d<Ttype, Dtype>*;/// std::shared_ptr<Tensor4d<Ttype, Dtype> >;

template<typename Ttype, DataType Dtype>
using TensorSharedPtr = std::shared_ptr<Tensor4d<Ttype, Dtype> >;

using Shape4d = Shape;/// saber::Shape<4>;

template<typename Ttype>
using OpContext = Context<Ttype>;///saber::Context<Ttype>;

template<typename Ttype>
using OpContextPtr = std::shared_ptr<OpContext<Ttype>>;

#else
#pragma error(" Anakin 2.0 can't Run without saber..")
typedef void Tensor4d;
typedef void Shape4d;
#endif

/** 
 *  \brief Tuple type parameter used for list parameter.
 */
template<typename T>
class PTuple {
public:
    PTuple() {}
    template<typename ...ArgTs>
    PTuple(ArgTs ...elems) {
        _init(elems...);
    }
    PTuple(const PTuple& p_tuple) {
        _size = p_tuple._size;
        _elems = p_tuple._elems;
    }
    PTuple(PTuple& p_tuple) {
        _size = p_tuple._size;
        _elems = p_tuple._elems;
    }

    /// Can't used for bool type.
    PTuple(std::vector<T>& elems) {
        _size = elems.size();
        _elems = elems;
    }
    PTuple(std::vector<T>&& elems) {
        _size = elems.size();
        _elems = std::move(elems);
    }

    inline int size() { return _size; }
    inline typename std_vector_type_warpper<T>::type& operator[](int index) {
        CHECK_GE(index, 0) << "index("<<index<<") must >= the size("<< _size <<") of PTuple";
        CHECK_LT(index, _size) << "index("<<index<<") must < the size("<< _size <<") of PTuple";
        return _elems[index];
    }

    inline std::vector<typename std_vector_type_warpper<T>::ret_type>&  vector() { 
        return _get_vector(Bool2Type<is_bool_type<T>::value>());
    }

    /// return the first data pointer.
    inline typename std_vector_type_warpper<T>::type* data() { return _elems.data(); }

    inline void push_back(T& value) { 
        _push_back(value, Bool2Type<is_bool_type<T>::value>());
    }

    inline void push_back(const T& value) { 
        _push_back(value, Bool2Type<is_bool_type<T>::value>());
    }

private:
    std::vector<typename std_vector_type_warpper<T>::ret_type>& _get_vector(Bool2Type<true>) {
        for(auto& item : _elems) {
            if(item == "true") {
                _ret_elems.push_back(true);
            } else {
                _ret_elems.push_back(false);
            }
        }
        return _ret_elems;
    }
    std::vector<typename std_vector_type_warpper<T>::ret_type>& _get_vector(Bool2Type<false>) {
        return _elems;
    }

    void _push_back(T& value, Bool2Type<true>) {
        _elems.push_back(value ? "true":"false");
        _size += 1;
    }
    void _push_back(const T& value, Bool2Type<true>) {
        _elems.push_back(value ? "true":"false");
        _size += 1;
    }
    void _push_back(T& value, Bool2Type<false>) { 
        _elems.push_back(value);
        _size += 1;
    }
    void _push_back(const T& value, Bool2Type<false>) { 
        _elems.push_back(value);
        _size += 1;
    }

    void _init(T elem) {
        CHECK_GE(elem, 0) << "Elem parameter( "<<elem<< ") must > 0.";
        _size++;
        _elems.push_back(elem);
    }
    template<typename ArgT, typename ...ArgTs>
    void _init(ArgT elem0, ArgTs ...elems) {
        static_assert(std::is_same<typename std_vector_type_warpper<ArgT>::type, 
                                   typename std_vector_type_warpper<T>::type>::value, 
                                   "ArgT must be same type as T");
        CHECK_GE(elem0, 0) << "Elem parameter( "<<elem0<< ") must > 0."; 
        _size++; 
        _elems.push_back(elem0);
        _init(elems...);
    }
private:
    int _size{0};
    std::vector<typename std_vector_type_warpper<T>::type> _elems;
    /// in case of T is bool, when get vector form.
    std::vector<typename std_vector_type_warpper<T>::ret_type> _ret_elems; 
};

template<typename T>
struct DataTypeRecover; /// declare for PBlock
//////////////////////////
//template<typename T >
//class PBlock_X86_2 {
//public:
//    typedef Tensor4d<NV, DataTypeRecover<T>::type> d_type;
//    typedef Tensor4d<X86, DataTypeRecover<T>::type> h_type;
//
//
//    PBlock_X86_2() {
//        _d_inner_tensor = std::make_shared<d_type>();
//        _h_inner_tensor = std::make_shared<h_type>();
//    }
//    PBlock_X86_2(Shape4d& shape) {
//        _d_inner_tensor = std::make_shared<d_type>(shape);
//        _h_inner_tensor = std::make_shared<h_type>(shape);
//    }
//
//    /// shallow copy construction
//    PBlock_X86_2(PBlock_X86_2<T>& p_block) { *this = p_block; }
//
//    PBlock_X86_2(const PBlock_X86_2<T>& p_block) { *this = p_block; }
//
//    /// assign
//    PBlock_X86_2<T>& operator=(const PBlock_X86_2<T>& p_block) {
//        _d_inner_tensor = p_block._d_inner_tensor;
//        _h_inner_tensor = p_block._h_inner_tensor;
//        /*_d_inner_tensor = std::make_shared<d_type>();
//        _h_inner_tensor = std::make_shared<h_type>();
//        _d_inner_tensor->set_shape(p_block._d_inner_tensor->shape());
//        _d_inner_tensor->share_from(*(p_block._d_inner_tensor));
//        _h_inner_tensor->set_shape(p_block._h_inner_tensor->shape());
//        _h_inner_tensor->share_from(*(p_block._h_inner_tensor));*/
//        /*_d_inner_tensor->share_sub_buffer(*(p_block._d_inner_tensor),
//                                          p_block._d_inner_tensor->shape(),
//                                          p_block._d_inner_tensor->offset());
//        _h_inner_tensor->share_sub_buffer(*(p_block._h_inner_tensor),
//                                          p_block._h_inner_tensor->shape(),
//                                          p_block._h_inner_tensor->offset());*/
//    }
//
//    PBlock_X86_2<T>& operator=(PBlock_X86_2<T>& p_block) {
//        _d_inner_tensor = p_block._d_inner_tensor;
//        _h_inner_tensor = p_block._h_inner_tensor;
//        /*_d_inner_tensor = std::make_shared<d_type>();
//        _h_inner_tensor = std::make_shared<h_type>();
//        _d_inner_tensor->set_shape(p_block._d_inner_tensor->shape());
//        _d_inner_tensor->share_from(*(p_block._d_inner_tensor));
//        _h_inner_tensor->set_shape(p_block._h_inner_tensor->shape());
//        _h_inner_tensor->share_from(*(p_block._h_inner_tensor));*/
//        /*_d_inner_tensor->share_sub_buffer(*(p_block._d_inner_tensor),
//                                          p_block._d_inner_tensor->shape(),
//                                          p_block._d_inner_tensor->offset());
//        _h_inner_tensor->share_sub_buffer(*(p_block._h_inner_tensor),
//                                          p_block._h_inner_tensor->shape(),
//                                          p_block._h_inner_tensor->offset());*/
//    }
//
//    /// Get tensor.
//    d_type& d_tensor() { return *(_d_inner_tensor); }
//    h_type& h_tensor() { return *(_h_inner_tensor); }
//
//    /// Get host data to vector.
//    std::vector<T> vector() {
//        std::vector<T> ret;
//        auto* data = _h_inner_tensor->mutable_data();
//        for (int i = 0; i <_h_inner_tensor->valid_size(); i++) {
//            ret.push_back(data[i]);
//        }
//        return ret;
//    }
//
//    /// Get shape.
//    Shape4d shape() {
//        CHECK(_d_inner_tensor->valid_shape() == _h_inner_tensor->valid_shape())
//        << " [Fatal Err]  device shape is not equal to that of host in PBlock";
//        return _d_inner_tensor->valid_shape();
//    }
//
//    /// Get size.
//    size_t count() {
//        return this->shape().count();
//    }
//
//    ~PBlock_X86_2() {}
//private:
//    std::shared_ptr<d_type> _d_inner_tensor;
//    std::shared_ptr<h_type> _h_inner_tensor;
//};
//////////////////////////////////
//template<typename T >
//class PBlock_X86 {
//public:
//
//    typedef Tensor4d<X86, DataTypeRecover<T>::type> h_type;
//
//
//    PBlock_X86() {
//        _h_inner_tensor = std::make_shared<h_type>();
//    }
//    PBlock_X86(Shape4d& shape) {
//        _h_inner_tensor = std::make_shared<h_type>(shape);
//    }
//
//    /// shallow copy construction
//    PBlock_X86(PBlock_X86<T>& p_block) { *this = p_block; }
//
//    PBlock_X86(const PBlock_X86<T>& p_block) { *this = p_block; }
//
//    /// assign
//    PBlock_X86<T>& operator=(const PBlock_X86<T>& p_block) {
//        _h_inner_tensor = p_block._h_inner_tensor;
//    }
//
//    PBlock_X86<T>& operator=(PBlock_X86<T>& p_block) {
//        _h_inner_tensor = p_block._h_inner_tensor;
//    }
//
//    /// Get tensor.
//    h_type& h_tensor() { return *(_h_inner_tensor); }
//
//    /// Get host data to vector.
//    std::vector<T> vector() {
//        std::vector<T> ret;
//        auto* data = _h_inner_tensor->mutable_data();
//        for (int i = 0; i <_h_inner_tensor->valid_size(); i++) {
//            ret.push_back(data[i]);
//        }
//        return ret;
//    }
//
//    /// Get shape.
//    Shape4d shape() {
//        CHECK(_h_inner_tensor->valid_shape() == _h_inner_tensor->valid_shape())
//        << " [Fatal Err]  device shape is not equal to that of host in PBlock";
//        return _h_inner_tensor->valid_shape();
//    }
//
//    /// Get size.
//    size_t count() {
//        return this->shape().count();
//    }
//
//    ~PBlock_X86() {}
//private:
//    std::shared_ptr<h_type> _h_inner_tensor;
//};
/** 
 *  \brief a simple wrapper of tensor use in weights parameter.
 *   default layout [ NCHW ]
 */
template<typename T >
class PBlock {
public:
    typedef Tensor4d<NV, DataTypeRecover<T>::type> d_type;
    typedef Tensor4d<NVHX86, DataTypeRecover<T>::type> h_type;


    PBlock() {
        _d_inner_tensor = std::make_shared<d_type>();
        _h_inner_tensor = std::make_shared<h_type>();
    }
    PBlock(Shape4d& shape) {
        _d_inner_tensor = std::make_shared<d_type>(shape);
        _h_inner_tensor = std::make_shared<h_type>(shape);
    }

    /// shallow copy construction
    PBlock(PBlock<T>& p_block) { *this = p_block; }

    PBlock(const PBlock<T>& p_block) { *this = p_block; }

    /// assign
    PBlock<T>& operator=(const PBlock<T>& p_block) {
        _d_inner_tensor = p_block._d_inner_tensor;
        _h_inner_tensor = p_block._h_inner_tensor;
        /*_d_inner_tensor = std::make_shared<d_type>();
        _h_inner_tensor = std::make_shared<h_type>();
        _d_inner_tensor->set_shape(p_block._d_inner_tensor->shape());
        _d_inner_tensor->share_from(*(p_block._d_inner_tensor));
        _h_inner_tensor->set_shape(p_block._h_inner_tensor->shape()); 
        _h_inner_tensor->share_from(*(p_block._h_inner_tensor));*/
        /*_d_inner_tensor->share_sub_buffer(*(p_block._d_inner_tensor), 
                                          p_block._d_inner_tensor->shape(),
                                          p_block._d_inner_tensor->offset());
        _h_inner_tensor->share_sub_buffer(*(p_block._h_inner_tensor),
                                          p_block._h_inner_tensor->shape(),
                                          p_block._h_inner_tensor->offset());*/
    }

    PBlock<T>& operator=(PBlock<T>& p_block) {
        _d_inner_tensor = p_block._d_inner_tensor;
        _h_inner_tensor = p_block._h_inner_tensor;
        /*_d_inner_tensor = std::make_shared<d_type>();
        _h_inner_tensor = std::make_shared<h_type>();
        _d_inner_tensor->set_shape(p_block._d_inner_tensor->shape());
        _d_inner_tensor->share_from(*(p_block._d_inner_tensor));
        _h_inner_tensor->set_shape(p_block._h_inner_tensor->shape()); 
        _h_inner_tensor->share_from(*(p_block._h_inner_tensor));*/
        /*_d_inner_tensor->share_sub_buffer(*(p_block._d_inner_tensor), 
                                          p_block._d_inner_tensor->shape(),
                                          p_block._d_inner_tensor->offset());
        _h_inner_tensor->share_sub_buffer(*(p_block._h_inner_tensor),
                                          p_block._h_inner_tensor->shape(),
                                          p_block._h_inner_tensor->offset());*/
    }

    /// Get tensor.
    d_type& d_tensor() { return *(_d_inner_tensor); }
    h_type& h_tensor() { return *(_h_inner_tensor); }

    /// Get host data to vector.
    std::vector<T> vector() {
        std::vector<T> ret;
        auto* data = _h_inner_tensor->mutable_data();
        for (int i = 0; i <_h_inner_tensor->valid_size(); i++) {
            ret.push_back(data[i]);
        }
        return ret;
    }

    /// Get shape.
    Shape4d shape() { 
        CHECK(_d_inner_tensor->valid_shape() == _h_inner_tensor->valid_shape()) 
            << " [Fatal Err]  device shape is not equal to that of host in PBlock";
        return _d_inner_tensor->valid_shape(); 
    }

    /// Get size.
    size_t count() { 
        return this->shape().count();
    }

    ~PBlock() {}
private:
    std::shared_ptr<d_type> _d_inner_tensor;
    std::shared_ptr<h_type> _h_inner_tensor;
};

/** 
 *  \brief Enum type.
 */
struct Enum {
    Enum(int val):value(val) {}

    template<typename REAL_ENUM_T> 
    REAL_ENUM_T cast() {
        return static_cast<REAL_ENUM_T>(value);
    }
    int value;
};

} /* namespace anakin */

#endif
