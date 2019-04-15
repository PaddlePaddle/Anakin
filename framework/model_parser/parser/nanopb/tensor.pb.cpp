#include <pb_decode.h>

#include <algorithm>
#include <memory>

#include "tensor.pb.hpp"

#include <pb_cpp_decode.h>

namespace nanopb_cpp {

void TensorShape::Dim::fill(Nanopb *pb) {

    // value: repeated int32
    pb->value.funcs.decode = decode_repeated<int32_t, decode_varint<int32_t>>;
    pb->value.arg = &_value;
    
    // size: optional int64
    
}

void TensorShape::Dim::retrieve(const Nanopb *pb) {

    // value: repeated int32
    
    // size: optional int64
    _size = static_cast<decltype(_size)>(pb->size);
    
}

IMPLEMENT_PARSING_WRAPPERS(TensorShape::Dim);

void TensorShape::fill(Nanopb *pb) {

    // dim: optional TensorShape.Dim
    _dim.fill(&pb->dim);
    
}

void TensorShape::retrieve(const Nanopb *pb) {

    // dim: optional TensorShape.Dim
    _dim.retrieve(&pb->dim);
    
}

IMPLEMENT_PARSING_WRAPPERS(TensorShape);

void CacheDate::fill(Nanopb *pb) {

    // s: repeated bytes
    pb->s.funcs.decode = decode_repeated<std::string, decode_string>;
    pb->s.arg = &_s;
    
    // i: repeated int32
    pb->i.funcs.decode = decode_repeated<int32_t, decode_varint<int32_t>>;
    pb->i.arg = &_i;
    
    // f: repeated float
    pb->f.funcs.decode = decode_repeated<float, decode_fixed32<float>>;
    pb->f.arg = &_f;
    
    // b: repeated bool
    pb->b.funcs.decode = decode_repeated<bool, decode_varint<bool>>;
    pb->b.arg = &_b;
    
    // l: repeated CacheDate
    pb->l.funcs.decode = decode_repeated<nanopb_cpp::CacheDate, decode_message<nanopb_cpp::CacheDate>>;
    pb->l.arg = &_l;
    
    // c: optional bytes
    pb->c.funcs.decode = decode_string;
    pb->c.arg = &_c;
    
    // type: optional DateTypeProto
    
    // size: optional int64
    
}

void CacheDate::retrieve(const Nanopb *pb) {

    // s: repeated bytes
    
    // i: repeated int32
    
    // f: repeated float
    
    // b: repeated bool
    
    // l: repeated CacheDate
    
    // c: optional bytes
    
    // type: optional DateTypeProto
    _type = static_cast<decltype(_type)>(pb->type);
    
    // size: optional int64
    _size = static_cast<decltype(_size)>(pb->size);
    
}

IMPLEMENT_PARSING_WRAPPERS(CacheDate);

void TensorProto::fill(Nanopb *pb) {

    // name: optional bytes
    pb->name.funcs.decode = decode_string;
    pb->name.arg = &_name;
    
    // shared: optional bool
    
    // share_from: optional bytes
    pb->share_from.funcs.decode = decode_string;
    pb->share_from.arg = &_share_from;
    
    // shape: optional TensorShape
    _shape.fill(&pb->shape);
    
    // valid_shape: optional TensorShape
    _valid_shape.fill(&pb->valid_shape);
    
    // data: optional CacheDate
    _data.fill(&pb->data);
    
    // scale: optional CacheDate
    _scale.fill(&pb->scale);
    
}

void TensorProto::retrieve(const Nanopb *pb) {

    // name: optional bytes
    
    // shared: optional bool
    _shared = static_cast<decltype(_shared)>(pb->shared);
    
    // share_from: optional bytes
    
    // shape: optional TensorShape
    _shape.retrieve(&pb->shape);
    
    // valid_shape: optional TensorShape
    _valid_shape.retrieve(&pb->valid_shape);
    
    // data: optional CacheDate
    _data.retrieve(&pb->data);
    
    // scale: optional CacheDate
    _scale.retrieve(&pb->scale);
    
}

IMPLEMENT_PARSING_WRAPPERS(TensorProto);

} // namespace nanopb_cpp
