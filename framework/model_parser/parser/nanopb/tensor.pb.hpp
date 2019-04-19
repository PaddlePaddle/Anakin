#ifndef NANOPB_CPP_TENSOR_PROTO_HPP
#define NANOPB_CPP_TENSOR_PROTO_HPP

#include <pb_cpp_common.h>


#define TensorShape Nanopb_TensorShape
#define CacheDate Nanopb_CacheDate
#define TensorProto Nanopb_TensorProto
#define TensorShape_Dim Nanopb_TensorShape_Dim
#include "tensor.pb.h"
#undef TensorShape
#undef CacheDate
#undef TensorProto
#undef TensorShape_Dim

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
        REPEATED_PROTO_FIELD(int32_t, value);
        PROTO_FIELD(int64_t, size);

        PARSING_MEMBERS(TensorShape_Dim);
    }; // end class Dim;

    PROTO_FIELD(nanopb_cpp::TensorShape::Dim, dim);

    PARSING_MEMBERS(TensorShape);
}; // end class TensorShape;

class CacheDate {
    REPEATED_PROTO_FIELD(std::string, s);
    REPEATED_PROTO_FIELD(int32_t, i);
    REPEATED_PROTO_FIELD(float, f);
    REPEATED_PROTO_FIELD(bool, b);
    REPEATED_PROTO_FIELD(nanopb_cpp::CacheDate, l);
    PROTO_FIELD(std::string, c);
    PROTO_FIELD(nanopb_cpp::DateTypeProto, type);
    PROTO_FIELD(int64_t, size);

    PARSING_MEMBERS(CacheDate);
}; // end class CacheDate;

class TensorProto {
    PROTO_FIELD(std::string, name);
    PROTO_FIELD(bool, shared);
    PROTO_FIELD(std::string, share_from);
    PROTO_FIELD(nanopb_cpp::TensorShape, shape);
    PROTO_FIELD(nanopb_cpp::TensorShape, valid_shape);
    PROTO_FIELD(nanopb_cpp::CacheDate, data);
    PROTO_FIELD(nanopb_cpp::CacheDate, scale);

    PARSING_MEMBERS(TensorProto);
}; // end class TensorProto;

} // namespace nanopb_cpp

using nanopb_cpp::TensorShape;
using nanopb_cpp::CacheDate;
using nanopb_cpp::TensorProto;

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
