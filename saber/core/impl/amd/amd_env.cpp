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

#include "core/env.h"
namespace anakin {

namespace saber {

#ifdef AMD_GPU

typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;

cl_platform_id AMD_ENV::platform_id = NULL;

void AMD_ENV::env_init(int max_stream) {
    Devs& devs = cur_env();

    if (devs.size() > 0) {
        return;
    }

    platform_id = AMD_API::get_platform_id();

    int count = 0;
    AMD_API::get_device_count(count);

    if (count == 0) {
        LOG(WARNING) << "no device found!";
    } else {
        LOG(INFO) << "found " << count << " device(s)";
    }

    int cur_id = AMD_API::get_device_id();

    for (int i = 0; i < count; i++) {
        AMD_API::set_device(i);
        devs.push_back(Device<AMD>(max_stream));
    }

    AMD_API::set_device(cur_id);
}

bool AMD_ENV::is_init() {
    CHECK(platform_id != NULL);
    return true;
}

cl_platform_id AMD_ENV::get_platform_id() {
    if (!is_init()) {
        return NULL;
    }

    return platform_id;
}

bool AMD_ENV::record = false;
std::string AMD_ENV::mTag;
std::list<std::string> AMD_ENV::tList;
std::map<std::string, std::list<cl_event_list>> AMD_ENV::eMap;

void AMD_ENV::add_event(const char* tag, cl_event_list event) {
    if (!record) {
        return;
    }

    std::map<std::string, std::list<cl_event_list>>::iterator it;
    it = eMap.find(std::string(tag));

    if (it != eMap.end()) {
        it->second.push_back(event);
    } else {
        LOG(INFO) << "record [" << tList.size() << "]=" << tag;
        tList.push_back(std::string(tag));
        std::list<cl_event_list> list;
        list.push_back(event);
        eMap[std::string(tag)] = list;
    }
}
void AMD_ENV::pop() {
    std::map<std::string, std::list<cl_event_list>>::iterator it;
    size_t t_size;
    size_t e_size;
    size_t s_size;
    size_t size;
    float executionTime = 0;
    float waitTime = 0;
    float g_execute = 0;
    float g_wait = 0;
    cl_ulong submit;
    cl_ulong start;
    cl_ulong end;
    cl_ulong wait;
    cl_ulong execute;
    CHECK(tList.size() == eMap.size());
    t_size = tList.size();
    std::string log;
    log.append("\n");
    std::string tmp;

    for (int i = 0 ; i < t_size; i++) {
        waitTime = executionTime = 0;
        std::string tag = tList.front();
        it = eMap.find(tag);
        std::list<cl_event_list> list = it->second;
        e_size = list.size();

        cl_ulong* s_waits = NULL, *s_executes = NULL;

        for (int j = 0 ; j < e_size; j++) {
            cl_event_list eList = list.front();

            s_size = eList.size();

            if (s_size > 1) {

                if (s_waits == NULL) {
                    s_waits = new cl_ulong[s_size];
                    s_executes = new cl_ulong[s_size];
                    memset(s_waits, 0, s_size * sizeof(cl_ulong));
                    memset(s_executes, 0, s_size * sizeof(cl_ulong));
                }

                int tmps = 0;

                for (cl_event_list::iterator ite = eList.begin(); ite != eList.end(); ite++) {

                    cl_event event = *ite;
                    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit, NULL);
                    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

                    s_waits[tmps] += (start - submit);
                    s_executes[tmps] += (end - start);
                    tmps++;
                }

                cl_event eventS = eList.front();
                cl_event eventE = eList.back();
                clGetEventProfilingInfo(eventS, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                clGetEventProfilingInfo(eventE, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                execute = end - start;

                clGetEventProfilingInfo(eventS, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit, NULL);
                wait = start - submit;

                clGetEventProfilingInfo(eventS, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &end, NULL);
                clGetEventProfilingInfo(eventE, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
                wait += start - end;
            } else {

                cl_event event = eList.front();
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit, NULL);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

                wait = start - submit;
                execute = end - start;
            }

            eList.clear();

            executionTime += (execute) * 1e-6;
            waitTime += (wait) * 1e-6;
            list.pop_front();
            //LOG(INFO) << tag << ":" << "["<< wait <<", " << execute <<"]";
        }

        //LOG(INFO) << "[" << i << "]" << tag << " avg - wait :"<< waitTime/e_size << " ms, execute " << executionTime/e_size <<" ms";
        tmp = std::string("[") + std::to_string(i) + std::string("]\t") + \
              tag + std::string("\t") + std::to_string(executionTime / e_size) + std::string(" ms\n");

        if (s_size > 1) {
            for (int s = 0; s < s_size; s++)  {
                tmp.append(std::string("--[") + std::to_string(i) + std::string("-") + std::to_string(
                               s) + std::string("]\t\t"));
                tmp.append(std::to_string((float)s_executes[s] * 1e-6 / e_size) + std::string(" ms\n"));
            }

            delete []s_waits;
            delete []s_executes;
            s_waits = NULL;
            s_executes = NULL;
        }

        log.append(tmp);

        g_wait += (waitTime / e_size);
        g_execute += (executionTime / e_size);
        //LOG(INFO) << "[" << i << "]"  << tag << '\t' << " avg - execute " << '\t'<< executionTime/e_size <<" ms";

        tList.pop_front();
    }

    tmp = std::string("[Total]\t\t") + \
          std::to_string(g_execute) + std::string(" ms\n");
    log.append(tmp);
    LOG(INFO) << log;
}



//template void AMD_ENV::evn_init();

#endif // AMD_GPU

} //namespace saber
} //namespace anakin

