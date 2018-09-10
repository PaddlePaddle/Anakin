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

#ifndef ANAKIN_SERVICE_TEST_H
#define ANAKIN_SERVICE_TEST_H

#include <iostream>
#include "utils/unit_test/aktest.h"
#include "utils/logger/logger.h"
#include "graph_base.h"
#include "graph.h"
#include "scheduler.h"
#include "net.h"
#include "worker.h"
#include "service_daemon.h"

using namespace anakin;
using ::anakin::test::Test;

using namespace anakin::rpc;

/**
 * \brief anakin service test is base Test class for anakin rpc service  
 */
class ServiceTest: public Test {
public:
    ServiceTest(){}

    void SetUp(){}

    void TearDown(){}

protected:
};

#endif


