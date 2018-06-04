#include "framework/lite/code_gen_cpp.h"

namespace anakin {

namespace lite {

void GenCPP::gen_header_start() {
	auto code_name = this->get_code_name();
	_code.Clean();
	_code.feed("#ifndef ANAKIN_%s_H \n", code_name.c_str());
	_code.feed("#define ANAKIN_%s_H \n\n", code_name.c_str());
	_code<<"namepsace anakin \{ \n\n";
}	

void GenCPP::gen_header_end() {
	_code<<"\} /* namespace anakin */\n";
	_code<<"#endif\n";
}

void GenCPP::gen_source_start() {
	auto code_name = this->get_code_name();
	_code.Clean();
	_code.feed("#include \"%s.h\" \n\n", code_name.c_str());
	_code<<"namepsace anakin \{ \n\n";
	// add running impl for model api
}	

void GenCPP::gen_source_end() {
	_code<<"\} /* namespace anakin */\n";
}


} /* namespace lite */

} /* namespace anakin */

