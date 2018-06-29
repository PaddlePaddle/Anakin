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

#ifndef ANAKIN_LLVM_SCHEDULER_H
#define ANAKIN_LLVM_SCHEDULER_H

#include "utils/logger/logger.h"
#include "framework/graph/llvm/schedule_base.h"
#include "framework/graph/llvm/virtual_graph.h"

namespace anakin {

namespace graph {

/// \brief customed hash for io structure
struct HashIO {
    /**
    *  \brief hash_io operation
    *  \return size_t the value of hash_io
    */
    size_t operator()(io const& _io) const noexcept {
        size_t hash_io = std::hash<std::string>{}(_io.name);
        return hash_io;
    }
};

/**
 * \brief Dependency scheduler for analysing the execution of ops in graph
 *
 *  note:
 *      lock all the out io resource in graph.
 *      1. if the in io res of the op are accessible, then invoke the op and free the out io res.
 *      2. if not accessible, block the op and invoke wait
 *      if in GPU mode:
 *          Check the TagStatus and lane of all the in io res of the target op
 *          - if all io lane are same value and TagStatus of io are accessible, then invoke the op
 *          - if not same, wait until the they are accessible.
 */
class Scheduler : public ScheduleBase<io, node, HashIO> {
public:
    Scheduler() {}
    virtual ~Scheduler() {}

    /// register the graph's read and write io resource.
    virtual void RegIOResource(VGraph*);
    
    /// decide if the target node's op is callable
    virtual bool callable(node&);

    /// launch operator and push op to execution queue
    virtual void launch(node&);

    /// run scheduler
    virtual void Run();

    /// get node name list in exec order
    std::vector<std::string> get_exec_node_in_order();

    /// check if io is fixed
    bool is_fixed(io&);

    /// ...TODO
    //
public:
    VGraph* _vgraph;  ///< _vgraph pointer hold from outside , so don't free it
    
    std::vector<io>  _fix_io_res; ///< _fix_io_res stand for io
};


} /* namespace graph */

} /* namespace anakin */

#endif /* ANAKIN_ALGO_H */
