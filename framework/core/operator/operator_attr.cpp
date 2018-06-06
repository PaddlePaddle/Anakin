#include "framework/core/operator/operator_attr.h"
#include "framework/core/operator/operator.h"

namespace anakin {

OpAttrWarpper& OpAttrWarpper::name(std::string op_name) {
    //! set the origin op name.
    opAttr_.name = op_name;
}

template<typename Ttype, DataType Dtype, Precision Ptype>
OpAttrWarpper& OpAttrWarpper::__alias__(std::string op_name) {
    OpAttrRegister::Global().add_alias(this->opAttr_.name, op_name);
    OpFactory<Ttype, Dtype, Ptype>::Global().add_alias(this->opAttr_.name, op_name);
    return *(this);
}

template
OpAttrWarpper& OpAttrWarpper::__alias__<NV, AK_FLOAT, Precision::FP32>(std::string op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<NV, AK_FLOAT, Precision::FP16>(std::string op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<NV, AK_FLOAT, Precision::INT8>(std::string op_name);

template
OpAttrWarpper& OpAttrWarpper::__alias__<ARM, AK_FLOAT, Precision::FP32>(std::string op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<ARM, AK_FLOAT, Precision::FP16>(std::string op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<ARM, AK_FLOAT, Precision::INT8>(std::string op_name);

OpAttrWarpper& OpAttrWarpper::Doc(std::string doc) {
    opAttr_.doc = doc;
    return *(this);
}

OpAttrWarpper& OpAttrWarpper::num_in(size_t num) {
    opAttr_.num_in = num;
    return *(this);
}

OpAttrWarpper& OpAttrWarpper::num_out(size_t num) {
    opAttr_.num_out = num;
    return *(this);
}

OpAttrWarpper& OpAttrWarpper::commutative(bool is_commutative) {
    opAttr_.is_commutative = is_commutative;
    return *(this);
}

template<typename T>
T& OpAttrWarpper::GetArg(std::string arg_name, graph::AttrInfo& info) {
    CHECK(this->has_arg(arg_name)) << " the operator doesn't have target argument: " << arg_name;
    CHECK(info.parameter.count(arg_name) > 0) << " Attr info doesn't have target argument: " <<
            arg_name;
    any& target_arg = info.parameter[arg_name];
    return any_cast<T>(target_arg);
}

} /* namespace anakin */
