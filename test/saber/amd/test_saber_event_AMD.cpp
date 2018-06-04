/* Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 
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
#include "test_saber_func_AMD.h"
#include "saber/core/context.h"
#include "saber/core/env.h"
#include "saber/funcs/timer.h"

using namespace anakin::saber;

typedef TargetWrapper<AMD> AMD_API;

void set_complete(cl_event e, int sec){
    sleep(sec);
    clSetUserEventStatus(e, CL_COMPLETE);
}

TEST(TestSaberFuncAMD, test_record_event) {

   cl_event event;
   AMD_API::create_event(event);
   Context<AMD> ctx(AMD_API::get_device_id(),0,0);

   LOG(INFO) << "event " <<event;
   AMD_API::record_event(event, ctx.get_compute_stream());
   LOG(INFO) << "event " <<event;

   AMD_API::destroy_event(event);
   LOG(INFO) << "event " <<event;
}

TEST(TestSaberFuncAMD, test_saber_timer){

   Context<AMD> ctx(AMD_API::get_device_id(), 0, 0);
   SaberTimer<AMD> t ;
   t.start(ctx);

   cl_event ue = clCreateUserEvent(Env<AMD>::cur_env()[AMD_API::get_device_id()].get_context(), NULL);
   AMD_API::query_event(ue);
   std::thread wt(set_complete, ue, 2);
   wt.join();

   AMD_API::sync_stream(ue,Env<AMD>::cur_env()[AMD_API::get_device_id()]._compute_stream[0]);
   t.end(ctx);
   float time = t.get_average_ms();
   LOG(INFO) << "Execution time " << time << " ms";
}

int main(int argc, const char** argv){
    Env<AMD>::env_init();
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
