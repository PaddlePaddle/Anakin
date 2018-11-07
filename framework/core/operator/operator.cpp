#include "framework/core/operator/operator.h"

namespace anakin {

std::vector<std::string>& OpAttrObjectRegister::get_list_op_name() {
    return this->get_list_name();
}

// Operator attributes warpper object register
OpAttrWarpper* OpAttrObjectRegister::operator[](const std::string op_name) {
    return ObjectRegister<OpAttrWarpper>::operator[](op_name);
}

void OpAttrObjectRegister::add_alias(const std::string& ori_op_name,
                                     const std::string& op_name_alias) {
    this->__alias__(ori_op_name, op_name_alias);
}

} /* namespace anakin */
