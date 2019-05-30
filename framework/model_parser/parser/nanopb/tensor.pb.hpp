#ifndef NANOPB_CPP_TENSOR_PROTO_HPP
#define NANOPB_CPP_TENSOR_PROTO_HPP

#include <pb_common.hpp>


#define TensorShape Nanopb_TensorShape
#define CacheDate Nanopb_CacheDate
#define TensorProto Nanopb_TensorProto
#define TensorShape_Dim Nanopb_TensorShape_Dim
#define DateTypeProto Nanopb_DateTypeProto
#include "tensor.pb.h"
#undef TensorShape
#undef CacheDate
#undef TensorProto
#undef TensorShape_Dim
#undef DateTypeProto

namespace nanopb_cpp {

enum DateTypeProto {
    STR = 0,
    INT8 = 2,
    INT32 = 4,
    FLOAT16 = 8,
    FLOAT = 13,
    DOUBLE = 14,
    BOOLEN = 20,
    CACHE_LIST = 30,
    TENSOR = 31,
};

class TensorShape {

    class Dim {

        PROTO_REPEATED_NUMERIC_FIELD(int32_t, value);

        PROTO_SINGULAR_NUMERIC_FIELD(int64_t, size);

        PROTO_MESSAGE_MEMBERS(Dim, TensorShape_Dim);
    }; // end class Dim;

    PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::TensorShape::Dim, dim);

    PROTO_MESSAGE_MEMBERS(TensorShape, TensorShape);
}; // end class TensorShape;

class CacheDate {

    PROTO_REPEATED_STRING_FIELD(s);

    PROTO_REPEATED_NUMERIC_FIELD(int32_t, i);

    PROTO_REPEATED_NUMERIC_FIELD(float, f);

    PROTO_REPEATED_NUMERIC_FIELD(bool, b);

    PROTO_REPEATED_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::CacheDate, l);

    PROTO_SINGULAR_STRING_FIELD(c);

    PROTO_SINGULAR_ENUM_FIELD(nanopb_cpp::DateTypeProto, type);

    PROTO_SINGULAR_NUMERIC_FIELD(int64_t, size);

    PROTO_MESSAGE_MEMBERS(CacheDate, CacheDate);
}; // end class CacheDate;

class TensorProto {

    PROTO_SINGULAR_STRING_FIELD(name);

    PROTO_SINGULAR_NUMERIC_FIELD(bool, shared);

    PROTO_SINGULAR_STRING_FIELD(share_from);

    PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::TensorShape, shape);

    PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::TensorShape, valid_shape);

    PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::CacheDate, data);

    PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::CacheDate, scale);

    PROTO_MESSAGE_MEMBERS(TensorProto, TensorProto);
}; // end class TensorProto;

} // namespace nanopb_cpp

using nanopb_cpp::TensorShape;
using nanopb_cpp::CacheDate;
using nanopb_cpp::TensorProto;

using nanopb_cpp::DateTypeProto;
using nanopb_cpp::STR;
using nanopb_cpp::INT8;
using nanopb_cpp::INT32;
using nanopb_cpp::FLOAT16;
using nanopb_cpp::FLOAT;
using nanopb_cpp::DOUBLE;
using nanopb_cpp::BOOLEN;
using nanopb_cpp::CACHE_LIST;
using nanopb_cpp::TENSOR;

#endif
