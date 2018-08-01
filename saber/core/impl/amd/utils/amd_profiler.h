/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_CORE_IMPL_AMD_AMDPROFILER_H
#define ANAKIN_SABER_CORE_IMPL_AMD_AMDPROFILER_H

#include "core/device.h"

namespace anakin {

namespace saber {

#ifdef AMD_GPU
typedef std::list<cl_event> cl_event_list;

class AMDProfiler {
public:
    static void add_event(const char* tag, cl_event_list event);
    static void add_event(cl_event_list event) {
        add_event(mTag.c_str(), event);
    }

    static void pop();
    static void set_tag(const char* tag) {
        mTag = std::string(tag);
    }

    static const std::string& get_tag() {
        return mTag;
    }

    static bool is_recording() {
        return record;
    }
    static void start_record() {
        record = true;
    }
    static void stop_record() {
        record = false;
    }

private:
    AMDProfiler() {}

    static std::map<std::string, std::list<cl_event_list>> eMap;
    static std::list<std::string> tList;
    static bool record;
    static std::string mTag;
};
#endif

} // namespace saber

} // namespace anakin

#endif // ANAKIN_SABER_CORE_IMPL_AMD_AMDPROFILER_H
