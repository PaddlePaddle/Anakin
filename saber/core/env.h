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

#ifndef ANAKIN_SABER_CORE_ENV_H
#define ANAKIN_SABER_CORE_ENV_H

#include "core/device.h"

namespace anakin{

namespace saber{

template <typename TargetType>
class Env {
public:
    typedef TargetWrapper<TargetType> API;
    typedef std::vector<Device<TargetType>> Devs;
    static Devs& cur_env() {
        static Devs* _g_env = new Devs();
        return *_g_env;
    }
    static void env_init(int max_stream = 4){
        Devs& devs = cur_env();
        if (devs.size() > 0){
            return;
        }
        int count = 0;
        API::get_device_count(count);
        if (count == 0) {
            CHECK(false) << "no device found!";
        } else {
            LOG(INFO) << "found " << count << " device(s)";
        }
        int cur_id = API::get_device_id();
        for (int i = 0; i < count; i++) {
            API::set_device(i);
            devs.push_back(Device<TargetType>(max_stream));
        }
        API::set_device(cur_id);
        LOG(INFO)<<"dev size = "<<devs.size();
    }
private:
    Env(){}
};

#ifdef USE_BM_PLACE
static std::shared_ptr<Env<BM>> env_holder;

template <>
class Env<BM> {
public:
    typedef TargetWrapper<BM> API;
    typedef std::vector<Device<BM>> Devs;
    static Devs& cur_env() {
        static Devs* _g_env = new Devs();
        return *_g_env;
    }
    static void env_init(int max_stream = 4){
        Devs& devs = cur_env();
        if (devs.size() > 0){
            return;
        }
        int count = 0;
        API::get_device_count(count);
        if (count == 0) {
            CHECK(false) << "no device found!";
        } else {
            LOG(INFO) << "found " << count << " device(s)";
        }
        int cur_id = API::get_device_id();
        for (int i = 0; i < count; i++) {
            API::set_device(i);
            devs.push_back(Device<BM>(max_stream));
        }
        API::set_device(cur_id);
        LOG(INFO)<<"dev size = "<<devs.size();

        env_holder.reset(new Env());
        API::init_handle();
    }

    ~Env() {
        API::deinit_handle();
    }
private:
    Env(){}
};
#endif

#ifdef AMD_GPU 
typedef std::list<cl_event> cl_event_list;

template <>
class Env<AMD> {
public:
    typedef TargetWrapper<AMD> API;
    typedef std::vector<Device<AMD>> Devs;
    static Devs& cur_env() {
        static Devs* _g_env = new Devs();
        return *_g_env;
    }

    static void env_init(int max_stream = 4);
    static bool is_init(); 
    static cl_platform_id get_platform_id();

    static void add_event(const char *tag, cl_event_list event);
    static void add_event(cl_event_list event) {
        add_event(mTag.c_str(), event);
    }

    static void pop();
    static void set_tag(const char *tag){
        mTag = std::string(tag);
    }

    static const std::string& get_tag(){
        return mTag;
    }

    static void start_record(){
        record = true;
    }
    static void stop_record(){
        record = false;
    }
private:
    Env(){}

    static cl_platform_id platform_id;
    static std::map<std::string, std::list<cl_event_list>> eMap;
    static std::list<std::string> tList;
    static bool record;
    static std::string  mTag;

};
#endif

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_CORE_ENV_H

