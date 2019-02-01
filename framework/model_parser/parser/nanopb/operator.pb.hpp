#ifndef NANOPB_CPP_OPERATOR_PROTO_HPP
#define NANOPB_CPP_OPERATOR_PROTO_HPP

#include <pb_cpp_common.h>


#define OpProto Nanopb_OpProto
#include "operator.pb.h"
#undef OpProto

namespace nanopb_cpp {

class OpProto {
    PROTO_FIELD(std::string, name);
    PROTO_FIELD(bool, is_commutative);
    PROTO_FIELD(int32_t, in_num);
    PROTO_FIELD(int32_t, out_num);
    PROTO_FIELD(std::string, description);

    PARSING_MEMBERS(OpProto);
}; // end class OpProto;

} // namespace nanopb_cpp

using nanopb_cpp::OpProto;


#endif
