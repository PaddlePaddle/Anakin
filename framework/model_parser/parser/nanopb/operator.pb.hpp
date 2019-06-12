#ifndef NANOPB_CPP_OPERATOR_PROTO_HPP
#define NANOPB_CPP_OPERATOR_PROTO_HPP

#include <pb_common.hpp>


#define OpProto Nanopb_OpProto
#include "operator.pb.h"
#undef OpProto

namespace nanopb_cpp {

class OpProto {

    PROTO_SINGULAR_STRING_FIELD(name);

    PROTO_SINGULAR_NUMERIC_FIELD(bool, is_commutative);

    PROTO_SINGULAR_NUMERIC_FIELD(int32_t, in_num);

    PROTO_SINGULAR_NUMERIC_FIELD(int32_t, out_num);

    PROTO_SINGULAR_STRING_FIELD(description);

    PROTO_MESSAGE_MEMBERS(OpProto, OpProto);
}; // end class OpProto;

} // namespace nanopb_cpp

using nanopb_cpp::OpProto;


#endif
