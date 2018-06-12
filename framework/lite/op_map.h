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

#ifndef ANAKIN_FRAMEWORK_LITE_OPERATION_MAP_H
#define ANAKIN_FRAMEWORK_LITE_OPERATION_MAP_H

#include <string>
#include <unordered_map>

namespace anakin {

namespace lite {

template<typename T> 
inline T get_attr(std::string& attr_name, graph::AttrInfo& attrs) { 
	const auto& it_end = attrs.parameter.end(); 
	auto it_find = attrs.parameter.find(attr_name); 
	if(it_find == it_end) { 
		LOG(FATAL) << "Target attr name(" << attr_name << ") not found."; 
		return T(); 
	} 
	return any_cast<T>(attrs.parameter[attr_name]); 
}

/**
 * \brief class Weghts
 */
struct WeghtOffset {
	struct Offset{
		size_t offset; // offset from start
		size_t length; // weight length
	}
	std::vector<Offset> weights;
};

class WeightsWritter {
public:

private:
	int _offset{0};
	std::vector<WeghtOffset> _weights;
	BinaryWritter _writter;
};

typedef Singleton<WeightsWritter> GraphWeghts;

/// function type for parser
typedef std::fuction<std::string(graph::AttrInfo& attr, 
								 std::string& op_class_name, 
								 std::string& node_name)> ParseParamFunctor;
/**
 * \brief class OpParser
 */
struct OpParser {
	std::string OpClassName;
	ParseParamFunctor parse;
};

/// operations map
extern std::unordered_map<std::string, OpParser> OPERATION_MAP;

} /* namespace lite */

} /* namespace anakin */

#endif
