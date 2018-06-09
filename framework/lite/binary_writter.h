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

#ifndef ANAKIN_FRAMEWORK_LITE_BINARY_WRITTER_H
#define ANAKIN_FRAMEWORK_LITE_BINARY_WRITTER_H

#include "framework/lite/file_stream.h"

namespace anakin {

namespace lite {

/**  
 *  \brief class to help generating code string.
 *
 */
class BinaryWritter {
public:
	BinaryWritter() {}

	explicit BinaryWritter(std::string path, const char* file_mode = "wb") {
		_file_io.open(path, file_mode);
	}

	// BinaryWritteropen file for code generating.
	void open(std::string& path, const char* file_mode) {
		_file_io.open(path, file_mode);
	}

	// write data list to file
	inline bool write(const void* ptr, size_t size, size_t count) {
		return write(ptr, size, count);
	}

	// read data list from file
	inline bool read(void* ptr, size_t size, size_t count) {
		return _file_io.read(ptr, size, count);
	}
	
private:
	LiteFileIO _file_io;
};

} /* namespace lite */

} /* namespace anakin */

#endif
