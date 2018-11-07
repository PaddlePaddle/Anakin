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

#ifndef ANAKIN_LLVM_SCHEDULER_MEMORY_H
#define ANAKIN_LLVM_SCHEDULER_MEMORY_H

#include "utils/logger/logger.h"
#include "framework/graph/llvm/schedule_base.h"
#include "framework/graph/llvm/virtual_graph.h"
#include "framework/graph/llvm/scheduler.h"

namespace anakin {

namespace graph {

/**
 * \brief check_self_shared struct
 *  used to check arcs in graph whether is shared
 */
struct check_self_shared {
    /// ops : Split and Reshape  
    std::vector<std::string> ops{
        "Split",
        "Reshape",
		"Gather",
		"Flatten"
    };
    /**
     * \brief whether node_arg's op is in ops
     * \param node_arg stand for certain node
     * \return bool the value of ops == node_arg.opName
     */
    inline bool operator()(node& node_arg) {
        for (auto& op_type : ops) {
            if (op_type == node_arg.opName) {
                return true;
            }
        }
        return false;
    }

    /**
     * \brief whether bottom_node's op is in ops
     * \param graph stand for current graph
     * \param node_tmp stand for certain node
     * \param self_shared_ios stand for shared ios queue
     * \return bool the value of ret
     */
    inline bool last_op_is_self_shared(VGraph* graph, node& node_tmp, std::vector<io>& self_shared_ios) {
	bool ret = false;
        auto node_arc_in_its = graph->get_in_arc_its(node_tmp.name);
        for (auto arc_in_it : node_arc_in_its) {
            auto& node_ref = (*graph)[arc_in_it->bottom()];
            for (auto& op_type : ops) {
                if (op_type == node_ref.opName) {
		    		self_shared_ios.push_back(arc_in_it->weight());
                    ret = true;
                }
            }
        }
        return ret;
    }
};

/**
 * \brief io block resource class used for scheduler of VGraph memory usage
 */
class IOBlockResource {
public:
    IOBlockResource() {}
    ~IOBlockResource() {}

    void free(std::vector<io>&, VGraph*);
    inline bool has_free(io& target) { 
        for (auto it = _free.begin(); it != _free.end();) { 
            auto& io_tmp = *it; 
            if (target.lane == io_tmp.lane) { 
                return true; 
            } 
            ++it; 
        } 
        return false; 
    } 
    inline io get_free(io& target) { 
        for (auto it = _free.begin(); it != _free.end();) { 
            auto& io_tmp = *it; 
            if (target.lane == io_tmp.lane) { 
                it = _free.erase(it); 
                return io_tmp; 
            } else { 
                ++it; 
            } 
        } 
        return io(); 
    }
    bool is_same_target(io&, io&, VGraph*);
    void push_free(io&, VGraph*);
    void lock(std::vector<io>&);
	bool is_locked(io&);
    inline void push_self_lock(io& io_tmp) { _self_lock.push_back(io_tmp);}
    void reg_self_lock_tree(io&, std::vector<io>&);
    void rm_self_lock_tree(io&);
	bool is_in_self_tree(io&);
    void free_self(std::vector<io>&, VGraph*);
    void map_ios_to_vgraph(std::vector<io>&, VGraph*);

private:
    //std::queue<io> _free;
    std::list<io> _free;
    std::list<io> _lock;
    std::list<io> _self_lock; // lock list for self shared op (e.g. split)
    std::unordered_map<io, std::vector<io>, HashIO> _self_lock_next_tree; // helper structure for shared op (e.g. split --> next edges)
};

/**
 *  \brief Dependency scheduler for analysing the execution of ops in graph
 */
class MemoryScheduler : public Scheduler {
public:
    MemoryScheduler() {}
    virtual ~MemoryScheduler() {}

    /// launch operator and push op to execution queue
    virtual void launch(node&) final;

    /// set fix io
    void set_fix_io(std::vector<io>&);
    
    /// ...TODO
    //
private:
    IOBlockResource _io_block_res;
    check_self_shared _need_self_shared;
};


} /* namespace graph */

} /* namespace anakin */

#endif /* ANAKIN_ALGO_H */
