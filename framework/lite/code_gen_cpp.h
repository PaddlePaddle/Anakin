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
	GenCPP() {}
	explicit GenCPP(std::string& file_name):_file_name(file_name) {
		_code.open(file_name);
	}
	~GenCPP() {}

	/// generate all cpp files
	virtual void gen_files() {
		this->gen_header();
		this->gen_source();
	}

	inline void write_to(std::string& file_path) {
	    _code.open(file_path, "w");
		_code.save();
	}


private:
	void gen_header_start();
	void gen_header_end();
	void gen_source_start();
	void gen_source_end();

	std::string gen_include_guard();

private:
	std::string _file_name;
	CodeWritter _code;
	CodeWritter _weights;
};

} /* namespace lite */

} /* namespace anakin */

#endif
