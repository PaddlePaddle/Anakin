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

#ifndef ANAKIN_FRAMEWORK_LITE_WRITTER_H
#define ANAKIN_FRAMEWORK_LITE_WRITTER_H

namespace anakin {

namespace lite {

/**  
 *  \brief file io class for generating code [ lack of output stream ]
 *
 */
class LiteFileIO {
public:
	LiteFileIO() {}

	explicit LiteFileIO(std::string path, const char* file_mode = "w") {
		this->open(path, file_mode);
	}

	~LiteFileIO() {
		if(_file_p) {
			fflush(this->_file_p);
			fclose(this->_file_p);
		}
	}

	// write msg to file
	inline size_t write(std::string& msg) {
		fprintf(this->_file_p, "%s\n", msg.c_str());
		fflush(this->_file_p);
	}

	inline bool is_file_open() {
		return _file_p != nullptr ? true:false;
	}
	
	inline std::string get_file_path() {
		return _file_path;
	}

	/// open the target file path
	void open(std::string& path, const char* file_mode) {
		if (!this->is_file_open()) {
		    _file_path = path;
		    char* file_path = strdup(path.c_str()); 
		    for (char* p = strchr(file_path + 1, '/'); p!=NULL; p = strchr(p + 1, '/')){ 
		    	*p = '\0'; 
		    	struct stat st; 
		    	if ((stat(file_path, &st) == 0) && (((st.st_mode) & S_IFMT) == S_IFDIR)){ 
		    		// file_path exists and is a directory. do nothing 
		    		*p = '/'; 
		    		continue; 
		    	} else { 
		    		if(mkdir(file_path,0755)==-1){ 
		    			LOG(FATAL) << "Failed to ceate the path "<< file_path;
		    		} 
		    	} 
		    	*p = '/'; 
		    } 
		    free(file_path); 
		    this->_file_p = fopen(path.c_str(), file_mode);
		    if (!this->_file_p){ 
		    	LOG(FATAL)<< "Failed to open " << path.c_str();
		    }
		}
	}

private:
	std::string _file_path{""};
	FILE* _file_p{nullptr};
};


/**  
 *  \brief class to help generating code string.
 *
 */
class CodeWritter {
public:
	Writter() {}
	explicit Writter(std::string path, const char* file_mode = "w") {
		_file_io.open(path, file_mode);
	}

	// Writter open file for code generating.
	void open(std::string& path, const char* file_mode) {
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
		split_idx='/';
		free(file_path);
		return name;
	}

	/// feed code string for writter directly.
	void feed(std::string& code_str) const {
		_code<<code_str;
	}

	/// feed format string for code writter.
	void feed(const char* format, ...) {
		va_list vlist;
		va_start(vlist, format);
		auto code_str = pick_format(format, vlist);
		// get msg
		free(code_str);
		msg = nullptr;
		va_end(vlist);
	}

	/// access to multi data type
	template<typename T>
	Writter& operator<<(const T& var) {
		_code<<var;
		return *this;
	}

	/// access for std::endl and other std io
	Writter& operator<<(std::ostream&(*func)(std::ostream&)) {
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

private:
	inline std::string get_code_string() {
		return _code.str();
	}

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
