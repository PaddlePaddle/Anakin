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
#include "framework/graph/graph.h"

namespace anakin {

namespace lite {

/**  
 *  \brief class to help generating binary file.
 *
 */
class BinaryWritter {
public:
    BinaryWritter() {}

    explicit BinaryWritter(std::string path) {
        this->open(path);
    }

    // BinaryWritteropen file for code generating.
    void open(std::string& path, const char* file_mode = "wb") {
        _file_io.open(path, file_mode);
    }

	// write data list to file
    inline bool write(void* ptr, size_t size, size_t count) {
        return _file_io.write(ptr, size, count);
    }

	// read data list from file
    inline bool read(void* ptr, size_t size, size_t count) {
        return _file_io.read(ptr, size, count);
    }
	
private:
    LiteFileIO _file_io;
};

/**
 * \brief class Weghts
 */
struct WeghtOffset {
    struct Offset{
		size_t offset{0}; // offset from start
		size_t length{0}; // weight length
	};
	std::vector<Offset> weights;
};

/**  
 *  \brief class to help generating model weigth file.
 *
 */
class WeightsWritter : public BinaryWritter {
public:
	WeightsWritter() {}
	~WeightsWritter() {}

	// set weight
	template<typename Ttype>
	void register_weights(const std::string& node_name, PBlock<Ttype>& weight) {
		size_t type_size = weight.h_tensor().get_dtype_size();
		WeghtOffset::Offset offset_tmp;
		offset_tmp.offset = _offset;
		offset_tmp.length = weight.count();
		_offset += offset_tmp.length * type_size;
		_node_weights_map[node_name].weights.push_back(offset_tmp);
		write(weight.h_tensor().mutable_data(), type_size, offset_tmp.length);
	}

	bool has_node(std::string node_name) {
		return _node_weights_map.count(node_name) > 0 ? true : false;
	}

	WeghtOffset get_weights_by_name(std::string node_name) {
		if (!has_node(node_name)) {
			LOG(FATAL) << "WeightsWritter doesn't have target node name: " << node_name;
			return WeghtOffset();
		}
		return _node_weights_map[node_name];
	}

private:
	size_t _offset{0};
	std::unordered_map<std::string, WeghtOffset> _node_weights_map;
};



} /* namespace lite */

} /* namespace anakin */

#endif
