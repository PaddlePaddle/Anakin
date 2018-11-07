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

#ifndef ANAKIN_FRAMEWORK_UTILS_LAYOUT_COMMON_H
#define ANAKIN_FRAMEWORK_UTILS_LAYOUT_COMMON_H
#include <string>
#include <map>
namespace anakin {

template <typename K, typename V, typename C>
bool map_get(const K& k, const C& c, V* value) {
  auto it = c.find(k);
  if (it != c.end()) {
    *value = it->second;
    return true;
  } else {
    return false;
  }
}

inline
LayoutType layout_from_string(const std::string layout_str) {
    const std::map<std::string, LayoutType> MapStringToLayoutType = {
        {"W", Layout_W},
        {"HW", Layout_HW},
        {"WH", Layout_WH},
        {"NW", Layout_NW},
        {"NHW", Layout_NHW},
        {"NCHW", Layout_NCHW},
        {"NHWC", Layout_NHWC},
        {"NCHW_C4", Layout_NCHW_C4},
        {"NCHW_C8", Layout_NCHW_C8},
        {"NCHW_C16", Layout_NCHW_C16}};
    LayoutType layout = Layout_invalid;
    map_get(layout_str, MapStringToLayoutType, &layout);
    return layout;
}

inline
int dims_from_layout(const LayoutType layouttype) {
    Layout* layout = nullptr;
    switch (layouttype) {
        case Layout_invalid: layout = nullptr; break;
        case Layout_W: layout = new W(); break;
        case Layout_HW: layout = new HW(); break;
        case Layout_WH: layout = new WH(); break;
        case Layout_NW: layout = new NW(); break;
        case Layout_NHW: layout = new NHW(); break;
        case Layout_NCHW: layout = new NCHW(); break;
        case Layout_NHWC: layout = new NHWC(); break;
        case Layout_NCHW_C4: layout = new NCHW_C4(); break;
        case Layout_NCHW_C8: layout = new NCHW_C8(); break;
        case Layout_NCHW_C16: layout = new NCHW_C16(); break;
    }
    if (layout != nullptr) {
        int dims = layout->dims();
        delete layout;
        return dims;
    } else {
        return -1;
    }
}

#endif

} /* namespace anakin */
