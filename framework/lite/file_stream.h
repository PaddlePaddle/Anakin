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

#ifndef ANAKIN_FRAMEWORK_LITE_FILE_STREAM_H
#define ANAKIN_FRAMEWORK_LITE_FILE_STREAM_H

#include "utils/logger/logger.h"

namespace anakin {

namespace lite {

/**  
 *  \brief file io class for generating code [ lack of output stream ]
 *
 */
class LiteFileIO {
public:
	LiteFileIO() {}

	explicit LiteFileIO(const std::string path, const char* file_mode = "wb") {
		this->open(path, file_mode);
	}

	~LiteFileIO() {
		if(_file_p) {
			fflush(this->_file_p);
			fclose(this->_file_p);
			this->_file_p = nullptr;
		}
	}

	// write msg to file
	inline bool write(const std::string& msg) {
		fprintf(this->_file_p, "%s\n", msg.c_str());
		fflush(this->_file_p);
		return true;
	}

	// write data list to file
	inline bool write(const void* ptr, size_t size, size_t count) {
		size_t ret = fwrite(ptr, size, count, this->_file_p);
		fflush(this->_file_p);
		if(ret != count) {
			LOG(ERROR) << "Writing error " << stderr;
			return false;
		}
		return true;
	}

	// read data list from file
	inline bool read(void* ptr, size_t size, size_t count) {
		size_t ret = fread(ptr, size, count, this->_file_p);
		if(ret != count) {
			LOG(ERROR) << "Reading error " << stderr;
			return false;
		}
		return true;
	}

	inline bool is_file_open() {
		return _file_p != nullptr ? true:false;
	}
	
	inline std::string get_file_path() {
		return _file_path;
	}

	/// open the target file path
	void open(const std::string& path, const char* file_mode) {
		// close old 
		if(is_file_open()) {
			fflush(this->_file_p); 
			fclose(this->_file_p);
			this->_file_p = nullptr;
		}
		// open new
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

} /* namespace lite */

} /* namespace anakin */

#endif
