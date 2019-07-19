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

#ifndef ANAKIN_LLVM_SCHEDULER_PARALLEL_H
#define ANAKIN_LLVM_SCHEDULER_PARALLEL_H

#include "utils/logger/logger.h"
#include "framework/graph/llvm/schedule_base.h"
#include "framework/graph/llvm/virtual_graph.h"
#include "framework/graph/llvm/scheduler.h"

namespace anakin {

namespace graph {

/**
* \brief SyncFlagController class
* the controller the lanes and sync flags for vgraph
*/
class SyncFlagController {
public:    
    /**
    * \brief [INITIAL] set lanes for vgraph
    * note: 
    *    the total number of lane is equal to that of graph inputs
    */
    void init(VGraph*);
    /// set sync flags at io stage of node
    void node_sync_flags(node&, VGraph*);  
    void io_sync_flags(node&, VGraph*);

private:
    void map_io_to_vgraph(io&, VGraph*);

    std::unordered_map<io, int, HashIO> _map_io_to_lane;
};

/**
* \brief ParallScheduler class
* Dependency scheduler for analysing the execution of ops in graph
*/
class ParallScheduler : public Scheduler {
public:
    ParallScheduler() {}
    virtual ~ParallScheduler() {}

    /// run scheduler
    virtual void Run();

    /// ...TODO
    //
private:
    ///< _sync_flag_ctl 
    SyncFlagController _sync_flag_ctl;
};


} /* namespace graph */

} /* namespace anakin */

#endif /* ANAKIN_ALGO_H */
