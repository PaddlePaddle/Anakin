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

#ifndef ANAKIN_SABER_FUNCS_TIMER_H
#define ANAKIN_SABER_FUNCS_TIMER_H

#include "anakin_config.h"
//#include <sys/time.h>
#include <chrono>
#include <list>
#include <limits>
#include "saber/core/common.h"
#include "saber/core/context.h"

namespace anakin{
namespace saber{

template <typename TargetType>
class SaberTimer final {

public:
    SaberTimer() {}

    ~SaberTimer() {}

    void clear() {
        ms_time.clear();
    }

    void start(Context<TargetType> &ctx) {
        tstart = std::chrono::system_clock::now();
    }

    void end(Context<TargetType> &ctx) {
        tend = std::chrono::system_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart);
        float elapse_ms = 1000.f * float(ts.count()) * std::chrono::microseconds::period::num / \
            std::chrono::microseconds::period::den;
        ms_time.push_back(elapse_ms);
    }

    float get_average_ms() {
        if (ms_time.size() == 0) {
            return 0.f;
        }
        float sum = 0.f;
        for (auto i : ms_time){
            sum += i;
        }
        return sum / ms_time.size();
    }

    // return tile (0-99) time.
    float get_tile_time(float tile) {

        if (tile <0 || tile > 100) {
            return -1.f;
        }
        int total_items = (int)ms_time.size();
        if (total_items <= 0) {
            return -2.f;
        }
        ms_time.sort();
        int pos = (int)(tile * total_items / 100);
        auto it = ms_time.begin();
        for (int i = 0; i < pos; ++i) {
            ++it;
        }
        return *it;
    }

    const std::list<float> get_time_stat() {
        return ms_time;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> tstart;
    std::chrono::time_point<std::chrono::system_clock> tend;
    std::list<float> ms_time;
};

#ifdef USE_CUDA
template <>
class SaberTimer<NV> final {

public:
    SaberTimer() {
        CUDA_CHECK(cudaEventCreate(&_e_start));
        CUDA_CHECK(cudaEventCreate(&_e_end));
    }

    ~SaberTimer() {
        CUDA_CHECK(cudaEventDestroy(_e_start));
        CUDA_CHECK(cudaEventDestroy(_e_end));
    }

    void clear() {
        ms_time.clear();
    }

    void start(Context<NV> &ctx) {
        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();
        CUDA_CHECK(cudaEventRecord(_e_start, cuda_stream));
    }

    void end(Context<NV> &ctx) {
        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();
        CUDA_CHECK(cudaEventRecord(_e_end, cuda_stream));
        CUDA_CHECK(cudaEventSynchronize(_e_end));
        float elapse_ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&elapse_ms, _e_start, _e_end));
        ms_time.push_back(elapse_ms);
    }

    float get_average_ms() {
        if (ms_time.size() == 0) {
            return 0.f;
        }
        float sum = 0.f;
        for (auto i : ms_time){
            sum += i;
        }
        return sum / ms_time.size();
    }

    // return tile (0-99) time.
    float get_tile_time(float tile) {

        if (tile <0 || tile > 100) {
            return -1.f;
        }
        int total_items = (int)ms_time.size();
        if (total_items <= 0) {
            return -2.f;
        }
        ms_time.sort();
        int pos = (int)(tile * total_items / 100);
        auto it = ms_time.begin();
        for (int i = 0; i < pos; ++i) {
            ++it;
        }
        return *it;
    }

    const std::list<float> get_time_stat() {
        return ms_time;
    }

private:
    cudaEvent_t _e_start, _e_end;
    std::list<float> ms_time;
};
#endif


#ifdef AMD_GPU 

typedef TargetWrapper<AMD> AMD_API;

template <>
class SaberTimer<AMD> final {

public:
    SaberTimer() {
        Env<AMD>::env_init();
        AMD_API::create_event(&_e_start);
        AMD_API::create_event(&_e_end);
    }

    ~SaberTimer() {
        AMD_API::destroy_event(_e_start);
        AMD_API::destroy_event(_e_end);
    }

    void clear() {
        ms_time.clear();
    }

    void start(Context<AMD> &ctx) {
        AMD_API::destroy_event(_e_start);
        AMD_API::record_event(_e_start, ctx.get_compute_stream());
    }

    void end(Context<AMD> &ctx) {
        if(_e_start == nullptr) {
            LOG(ERROR) << "please call start() befoer call end()";
            return;
        }

        AMD_API::destroy_event(_e_end);
        AMD_API::record_event(_e_end, ctx.get_compute_stream());
        AMD_API::sync_event(_e_end);

        cl_ulong start;
        clGetEventProfilingInfo(_e_start, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start,NULL);

        cl_ulong end;
        clGetEventProfilingInfo(_e_end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

        float executionTime = 1e-6 * (end - start);
        ms_time.push_back(executionTime);
    }

    float get_average_ms() {
        if (ms_time.size() == 0) {
            return 0.f;
        }
        float sum = 0.f;
        for (auto i : ms_time){
            sum += i;
        }
        return sum / ms_time.size();
    }

    float get_best_ms(){
        if (ms_time.size() == 0) {
            return 0.f;
        }
#if 0
        for(auto time : ms_time)
           LOG(INFO) << time; 
#endif
        ms_time.sort();
        LOG(INFO) << ms_time.front() <<" - " << ms_time.back();

        return ms_time.front();
    }

   // return tile (0-99) time.
   float get_tile_time(float tile) {

        if (tile <0 || tile > 100) {
            return -1.f;
        }
        int total_items = (int)ms_time.size();
        if (total_items <= 0) {
            return -2.f;
        }
        ms_time.sort();
        int pos = (int)(tile * total_items / 100);
        auto it = ms_time.begin();
        for (int i = 0; i < pos; ++i) {
            ++it;
        }
        return *it;
    }

    const std::list<float> get_time_stat() {
        return ms_time;
    }

private:
    cl_event _e_start, _e_end;
    std::list<float> ms_time;
};
#endif


}
}

#endif //SABER_TIMER_H
