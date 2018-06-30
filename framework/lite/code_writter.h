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

#ifndef ANAKIN_FRAMEWORK_LITE_CODE_WRITTER_H
#define ANAKIN_FRAMEWORK_LITE_CODE_WRITTER_H

#include <sstream>
#include "framework/lite/file_stream.h"

namespace anakin {

namespace lite {

/**  
 *  \brief class to help generating code string.
 *
 */
class CodeWritter {
public:
	CodeWritter() {}
	explicit CodeWritter(std::string path) {
		this->open(path);
	}

	// CodeWritter open file for code generating.
	void open(std::string& path, const char* file_mode = "w" ) {
		_file_io.open(path, file_mode);
	}

	// get CodeWritter's target name
	std::string get_code_name() {
		auto path = _file_io.get_file_path();
		char* file_path = strdup(path.c_str()); 
		char* pos_end = file_path + path.size()-1;
		char* split_idx = nullptr;
		while(*pos_end != '/') {
			if(*pos_end == '.') {
				*pos_end = '\0';
				split_idx = pos_end;
			}
			pos_end--;
		}
		std::string name = std::string(pos_end+1);
		*split_idx='/';
		free(file_path);
		return name;
	}

	/// feed format string for code writter.
	void feed(const char* format, ...) {
		va_list vlist;
		va_start(vlist, format);
		auto code_str_p = pick_format(format, vlist);
		// get msg
		_code<<code_str_p;
		free(code_str_p);
		code_str_p = nullptr;
		va_end(vlist);
	}

	/// access to multi data type
	template<typename T>
	CodeWritter& operator<<(const T& var) {
		_code<<var;
		return *this;
	}

	/// access for std::endl and other std io
	CodeWritter& operator<<(std::ostream&(*func)(std::ostream&)) {
		func(_code);
		return *this;
	}

	/// clean the current code writter's code.
	void Clean() {
		_code.str("");
		_code.clear();
	}

	/// save code to target file path
	void save() {
		_file_io.write(_code.str());
	}

	inline std::string get_code_string() {
		return _code.str();
	}

private:
	inline char* pick_format(const char* format, va_list vlist) {
		char* msg = nullptr;
		int result = vasprintf(&msg, format, vlist);
		if(result == -1){ 
			LOG(ERROR) <<"Bad string format: "<< format ;
			return nullptr;
		} 
		return msg;
	}

private:
	std::ostringstream _code;
	LiteFileIO _file_io;
};

} /* namespace lite */

} /* namespace anakin */

#endif
