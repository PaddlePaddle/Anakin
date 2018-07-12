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

#include "test_saber_core_AMD.h"

#ifdef USE_AMD

using namespace anakin::saber;

TEST(TestSaberCoreAMD, test_AMD_context) {
    Env<AMD>::env_init();
    typedef TargetWrapper<AMD> API;
    typename API::event_t event;
    API::create_event(event);
    LOG(INFO) << "test context constructor";
    Context<AMD> ctx0;
    Context<AMD> ctx1(0, 1, 1);
    LOG(INFO) << "test record event to context data stream and compute stream";
    API::record_event(event, ctx0.get_data_stream());
    API::record_event(event, ctx0.get_compute_stream());
    API::record_event(event, ctx1.get_data_stream());
    API::record_event(event, ctx1.get_compute_stream());
}

#endif

int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

