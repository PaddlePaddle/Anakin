/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/

#ifndef ANAKIN_FRAMEWORK_LITE_CODE_GENERATE_CPP_H
#define ANAKIN_FRAMEWORK_LITE_CODE_GENERATE_CPP_H

namespace anakin {

namespace lite {

/**  
 *  \brief class to generate cpp files.
 *
 */
class GenCPP : public CodeGenBase {
public:
	explicit GenCPP(std::string& file_name):CodeGenBase(file_name) {}
	~GenCPP() {}

	/// generate all cpp files
	virtual void gen_files() {
		this->gen_header();
		this->gen_source();
	}

private:
	/// 
	void gen_header() {
		auto code_name = this->get_code_name();
		this->Clean();
		this->feed("#ifndef ANAKIN_%s_H \n", code_name.c_str());
		this->feed("#define ANAKIN_%s_H \n\n", code_name.c_str());
		(*this)<<"namepsace anakin \{ \n\n";
		// add running api for anakin-lite model
		
		(*this)<<"\} /* namespace anakin */\n";
		(*this)<<"#endif\n";
	}

	void gen_source() {
		auto code_name = this->get_code_name();
		this->Clean();
		this->feed("#include \"%s.h\" \n\n", code_name.c_str());
		(*this)<<"namepsace anakin \{ \n\n";
		// add running impl for model api
		
		(*this)<<"\} /* namespace anakin */\n";
	}

	std::string gen_include_guard() {
	}

private:
};

} /* namespace lite */

} /* namespace anakin */

#endif
