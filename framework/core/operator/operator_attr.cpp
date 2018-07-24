#include "framework/core/operator/operator_attr.h"
#include "framework/core/operator/operator.h"

namespace anakin {

OpAttrWarpper& OpAttrWarpper::name(const std::string& op_name) {
    //! set the origin op name.
    opAttr_.name = op_name;
    return *this;
}

template<typename Ttype, Precision Ptype>
OpAttrWarpper& OpAttrWarpper::__alias__(const std::string& op_name) {
    OpAttrRegister::Global().add_alias(this->opAttr_.name, op_name);
    OpFactory<Ttype, Ptype>::Global().add_alias(this->opAttr_.name, op_name);
    return *this;
}
//#ifdef USE_CUDA
template
OpAttrWarpper& OpAttrWarpper::__alias__<NV, AK_FLOAT, Precision::FP32>(const std::string& op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<NV, AK_FLOAT, Precision::FP16>(const std::string& op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<NV, AK_FLOAT, Precision::INT8>(const std::string& op_name);
//#endif

//#ifdef USE_X86_PLACE
template
OpAttrWarpper& OpAttrWarpper::__alias__<X86, AK_FLOAT, Precision::FP32>(const std::string& op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<X86, AK_FLOAT, Precision::FP16>(const std::string& op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<X86, AK_FLOAT, Precision::INT8>(const std::string& op_name);
//#endif

//#ifdef USE_ARM_PLACE
//#ifdef ANAKIN_TYPE_FP32
template
OpAttrWarpper& OpAttrWarpper::__alias__<ARM, AK_FLOAT, Precision::FP32>(const std::string& op_name);
//#endif

//#ifdef ANAKIN_TYPE_FP16
template
OpAttrWarpper& OpAttrWarpper::__alias__<ARM, AK_FLOAT, Precision::FP16>(const std::string& op_name);
//#endif

//#ifdef ANAKIN_TYPE_INT8
template
OpAttrWarpper& OpAttrWarpper::__alias__<ARM, AK_FLOAT, Precision::INT8>(const std::string& op_name);
//#endif

//#endif

template
OpAttrWarpper& OpAttrWarpper::__alias__<AMD, AK_FLOAT, Precision::FP32>(const std::string& op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<AMD, AK_FLOAT, Precision::FP16>(const std::string& op_name);
template
OpAttrWarpper& OpAttrWarpper::__alias__<AMD, AK_FLOAT, Precision::INT8>(const std::string& op_name);

OpAttrWarpper& OpAttrWarpper::Doc(const std::string& doc) {
    opAttr_.doc = doc;
    return *this;
}

OpAttrWarpper& OpAttrWarpper::num_in(size_t num) {
    opAttr_.num_in = num;
    return *this;
}

OpAttrWarpper& OpAttrWarpper::num_out(size_t num) {
    opAttr_.num_out = num;
    return *this;
}

OpAttrWarpper& OpAttrWarpper::commutative(bool is_commutative) {
    opAttr_.is_commutative = is_commutative;
    return *this;
}

} /* namespace anakin */
