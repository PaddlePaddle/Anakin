#include "tensor.pb.hpp"
#include <pb_codec.hpp>

namespace nanopb_cpp {

void TensorShape::Dim::CopyFrom(const TensorShape::Dim &other) {

    // value: repeated int32
    field_copy(_value, other._value);

    // size: optional int64
    field_copy(_size, other._size);

}

void TensorShape::Dim::pre_decode(Nanopb *pb) {

    // value: repeated int32
    pb->value.funcs.decode = codec_repeat<int32_t, codec_varint>::decode;
    pb->value.arg = &_value;
    
    // size: optional int64
    
}

void TensorShape::Dim::post_decode(const Nanopb *pb) {

    // value: repeated int32
    
    // size: optional int64
    _size = static_cast<decltype(_size)>(pb->size);
    
}

void TensorShape::Dim::pre_encode(Nanopb *pb) const {

    // value: repeated int32
    pb->value.funcs.encode = codec_repeat<int32_t, codec_varint>::encode;
    pb->value.arg = const_cast<void *>(static_cast<const void *>(&_value));
    
    // size: optional int64
    pb->size = static_cast<decltype(pb->size)>(_size);
    
}

IMPLEMENT_CODEC_MEMBERS(TensorShape::Dim);
void TensorShape::CopyFrom(const TensorShape &other) {

    // dim: optional TensorShape.Dim
    field_copy(_dim, other._dim);

}

void TensorShape::pre_decode(Nanopb *pb) {

    // dim: optional TensorShape.Dim
    _dim->pre_decode(&pb->dim);
    
}

void TensorShape::post_decode(const Nanopb *pb) {

    // dim: optional TensorShape.Dim
    _dim->post_decode(&pb->dim);
    
}

void TensorShape::pre_encode(Nanopb *pb) const {

    // dim: optional TensorShape.Dim
    _dim->pre_encode(&pb->dim);
    
}

IMPLEMENT_CODEC_MEMBERS(TensorShape);
void CacheDate::CopyFrom(const CacheDate &other) {

    // s: repeated bytes
    field_copy(_s, other._s);

    // i: repeated int32
    field_copy(_i, other._i);

    // f: repeated float
    field_copy(_f, other._f);

    // b: repeated bool
    field_copy(_b, other._b);

    // l: repeated CacheDate
    field_copy(_l, other._l);

    // c: optional bytes
    field_copy(_c, other._c);

    // type: optional DateTypeProto
    field_copy(_type, other._type);

    // size: optional int64
    field_copy(_size, other._size);

}

void CacheDate::pre_decode(Nanopb *pb) {

    // s: repeated bytes
    pb->s.funcs.decode = codec_repeat<std::string, codec_obj>::decode;
    pb->s.arg = &_s;
    
    // i: repeated int32
    pb->i.funcs.decode = codec_repeat<int32_t, codec_varint>::decode;
    pb->i.arg = &_i;
    
    // f: repeated float
    pb->f.funcs.decode = codec_repeat<float, codec_fixed32>::decode;
    pb->f.arg = &_f;
    
    // b: repeated bool
    pb->b.funcs.decode = codec_repeat<bool, codec_varint>::decode;
    pb->b.arg = &_b;
    
    // l: repeated CacheDate
    pb->l.funcs.decode = codec_repeat<nanopb_cpp::CacheDate, codec_obj>::decode;
    pb->l.arg = &_l;
    
    // c: optional bytes
    pb->c.funcs.decode = codec_obj<std::string>::decode;
    pb->c.arg = &_c;
    
    // type: optional DateTypeProto
    
    // size: optional int64
    
}

void CacheDate::post_decode(const Nanopb *pb) {

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

void CacheDate::pre_encode(Nanopb *pb) const {

    // s: repeated bytes
    pb->s.funcs.encode = codec_repeat<std::string, codec_obj>::encode;
    pb->s.arg = const_cast<void *>(static_cast<const void *>(&_s));
    
    // i: repeated int32
    pb->i.funcs.encode = codec_repeat<int32_t, codec_varint>::encode;
    pb->i.arg = const_cast<void *>(static_cast<const void *>(&_i));
    
    // f: repeated float
    pb->f.funcs.encode = codec_repeat<float, codec_fixed32>::encode;
    pb->f.arg = const_cast<void *>(static_cast<const void *>(&_f));
    
    // b: repeated bool
    pb->b.funcs.encode = codec_repeat<bool, codec_varint>::encode;
    pb->b.arg = const_cast<void *>(static_cast<const void *>(&_b));
    
    // l: repeated CacheDate
    pb->l.funcs.encode = codec_repeat<nanopb_cpp::CacheDate, codec_obj>::encode;
    pb->l.arg = const_cast<void *>(static_cast<const void *>(&_l));
    
    // c: optional bytes
    pb->c.funcs.encode = codec_obj<std::string>::encode;
    pb->c.arg = const_cast<void *>(static_cast<const void *>(&_c));
    
    // type: optional DateTypeProto
    pb->type = static_cast<decltype(pb->type)>(_type);
    
    // size: optional int64
    pb->size = static_cast<decltype(pb->size)>(_size);
    
}

IMPLEMENT_CODEC_MEMBERS(CacheDate);
void TensorProto::CopyFrom(const TensorProto &other) {

    // name: optional bytes
    field_copy(_name, other._name);

    // shared: optional bool
    field_copy(_shared, other._shared);

    // share_from: optional bytes
    field_copy(_share_from, other._share_from);

    // shape: optional TensorShape
    field_copy(_shape, other._shape);

    // valid_shape: optional TensorShape
    field_copy(_valid_shape, other._valid_shape);

    // data: optional CacheDate
    field_copy(_data, other._data);

    // scale: optional CacheDate
    field_copy(_scale, other._scale);

}

void TensorProto::pre_decode(Nanopb *pb) {

    // name: optional bytes
    pb->name.funcs.decode = codec_obj<std::string>::decode;
    pb->name.arg = &_name;
    
    // shared: optional bool
    
    // share_from: optional bytes
    pb->share_from.funcs.decode = codec_obj<std::string>::decode;
    pb->share_from.arg = &_share_from;
    
    // shape: optional TensorShape
    _shape->pre_decode(&pb->shape);
    
    // valid_shape: optional TensorShape
    _valid_shape->pre_decode(&pb->valid_shape);
    
    // data: optional CacheDate
    _data->pre_decode(&pb->data);
    
    // scale: optional CacheDate
    _scale->pre_decode(&pb->scale);
    
}

void TensorProto::post_decode(const Nanopb *pb) {

    // name: optional bytes
    
    // shared: optional bool
    _shared = static_cast<decltype(_shared)>(pb->shared);
    
    // share_from: optional bytes
    
    // shape: optional TensorShape
    _shape->post_decode(&pb->shape);
    
    // valid_shape: optional TensorShape
    _valid_shape->post_decode(&pb->valid_shape);
    
    // data: optional CacheDate
    _data->post_decode(&pb->data);
    
    // scale: optional CacheDate
    _scale->post_decode(&pb->scale);
    
}

void TensorProto::pre_encode(Nanopb *pb) const {

    // name: optional bytes
    pb->name.funcs.encode = codec_obj<std::string>::encode;
    pb->name.arg = const_cast<void *>(static_cast<const void *>(&_name));
    
    // shared: optional bool
    pb->shared = static_cast<decltype(pb->shared)>(_shared);
    
    // share_from: optional bytes
    pb->share_from.funcs.encode = codec_obj<std::string>::encode;
    pb->share_from.arg = const_cast<void *>(static_cast<const void *>(&_share_from));
    
    // shape: optional TensorShape
    _shape->pre_encode(&pb->shape);
    
    // valid_shape: optional TensorShape
    _valid_shape->pre_encode(&pb->valid_shape);
    
    // data: optional CacheDate
    _data->pre_encode(&pb->data);
    
    // scale: optional CacheDate
    _scale->pre_encode(&pb->scale);
    
}

IMPLEMENT_CODEC_MEMBERS(TensorProto);
} // namespace nanopb_cpp
