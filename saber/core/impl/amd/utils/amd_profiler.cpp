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

#include "amd_profiler.h"
namespace anakin {

namespace saber {

#ifdef AMD_GPU

bool AMDProfiler::record = false;
std::string AMDProfiler::mTag;
std::list<std::string> AMDProfiler::tList;
std::map<std::string, std::list<cl_event_list>> AMDProfiler::eMap;

void AMDProfiler::add_event(const char* tag, cl_event_list event) {
    if (!record)
        return;

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
void AMDProfiler::pop() {
    std::map<std::string, std::list<cl_event_list>>::iterator it;
    size_t t_size, e_size, s_size, size;
    float executionTime = 0, waitTime = 0, g_execute = 0, g_wait = 0;
    cl_ulong submit, start, end, wait, execute;
    CHECK(tList.size() == eMap.size());
    t_size = tList.size();
    std::string log;
    log.append("\n");
    std::string tmp;

    for (int i = 0; i < t_size; i++) {
        waitTime = executionTime      = 0;
        std::string tag               = tList.front();
        it                            = eMap.find(tag);
        std::list<cl_event_list> list = it->second;
        e_size                        = list.size();

        cl_ulong *s_waits = NULL, *s_executes = NULL;
        for (int j = 0; j < e_size; j++) {
            cl_event_list eList = list.front();

            s_size = eList.size();

            if (s_size > 1) {

                if (s_waits == NULL) {
                    s_waits    = new cl_ulong[s_size];
                    s_executes = new cl_ulong[s_size];
                    memset(s_waits, 0, s_size * sizeof(cl_ulong));
                    memset(s_executes, 0, s_size * sizeof(cl_ulong));
                }

                int tmps = 0;
                for (cl_event_list::iterator ite = eList.begin(); ite != eList.end(); ite++) {

                    cl_event event = *ite;
                    clGetEventProfilingInfo(
                            event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit, NULL);
                    clGetEventProfilingInfo(
                            event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                    clGetEventProfilingInfo(
                            event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

                    s_waits[tmps] += (start - submit);
                    s_executes[tmps] += (end - start);
                    tmps++;
                }

                cl_event eventS = eList.front();
                cl_event eventE = eList.back();
                clGetEventProfilingInfo(
                        eventS, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                clGetEventProfilingInfo(
                        eventE, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                execute = end - start;

                clGetEventProfilingInfo(
                        eventS, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit, NULL);
                wait = start - submit;

                clGetEventProfilingInfo(
                        eventS, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &end, NULL);
                clGetEventProfilingInfo(
                        eventE, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
                wait += start - end;
            } else {

                cl_event event = eList.front();
                clGetEventProfilingInfo(
                        event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit, NULL);
                clGetEventProfilingInfo(
                        event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                clGetEventProfilingInfo(
                        event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

                wait    = start - submit;
                execute = end - start;
            }
            eList.clear();

            executionTime += (execute)*1e-6;
            waitTime += (wait)*1e-6;
            list.pop_front();
            // LOG(INFO) << tag << ":" << "["<< wait <<", " << execute <<"]";
        }
        // LOG(INFO) << "[" << i << "]" << tag << " avg - wait :"<< waitTime/e_size
        // << " ms, execute " << executionTime/e_size <<" ms";
        tmp = std::string("[") + std::to_string(i) + std::string("]\t") + tag + std::string("\t")
              + std::to_string(executionTime / e_size) + std::string(" ms\n");

        if (s_size > 1) {
            for (int s = 0; s < s_size; s++) {
                tmp.append(
                        std::string("--[") + std::to_string(i) + std::string("-")
                        + std::to_string(s) + std::string("]\t\t"));
                tmp.append(
                        std::to_string((float)s_executes[s] * 1e-6 / e_size)
                        + std::string(" ms\n"));
            }

            delete s_waits;
            delete s_executes;
            s_waits = s_executes = NULL;
        }

        log.append(tmp);

        g_wait += (waitTime / e_size);
        g_execute += (executionTime / e_size);
        // LOG(INFO) << "[" << i << "]"  << tag << '\t' << " avg - execute " <<
        // '\t'<< executionTime/e_size <<" ms";

        tList.pop_front();
    }

    tmp = std::string("[Total]\t\t") + std::to_string(g_execute) + std::string(" ms\n");
    log.append(tmp);
    LOG(INFO) << log;
}
#endif // AMD_GPU

} // namespace saber
} // namespace anakin
