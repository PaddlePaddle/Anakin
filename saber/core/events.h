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

#ifndef ANAKIN_SABER_CORE_EVENTS_H
#define ANAKIN_SABER_CORE_EVENTS_H

#include "core/target_wrapper.h"

namespace anakin{

namespace saber{

template <typename TargetType>
class Events{
public:
    typedef TargetWrapper<TargetType> API;
    /**
     * \brief create target specific event
     */
    explicit Events(){
        API::create_event(&_event);
    }

    /**
     * \brief destroy event
     */
    ~Events(){
        API::destroy_event(_event);
    }

    /**
     * \brief query the event
     */
    void query() {
        API::query_event(_event);
    }

    /**
     * \brief record event to input stream
     * @param stream    stream where processes happend
     */
    void record(typename API::stream_t stream){
        API::record_event(_event, stream);
    }

    /**
     * \brief synchronize the event, block host process
     */
    void sync_host(){
        API::sync_event(_event);
    }

    /**
     * \brief synchronize event on a specific stream
     * @param stream
     */
    void sync_stream(typename API::stream_t& stream){
        API::sync_stream(_event, stream);
    }

public:
    typename API::event_t _event;
};


template <typename TargetType>
class EventsTree{
public:
    typedef TargetWrapper<TargetType> API;
    EventsTree() : _events(){}
    ~EventsTree(){}

    void set_parent(EventsTree* parent){
        _parent = parent;
    }
    
    void insert_children(EventsTree* child){
        child->set_parent(this);
        _children.push_back(child);
    }
    
    void sync_tree(){
        for (int i = 0; i < _children.size(); ++i) {
            _children[i]->sync_tree();
        }
        _events.sync_host();
    }

public:
    Events<TargetType> _events;
    EventsTree* _parent;
    std::vector<EventsTree*> _children;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_CORE_EVENTS_H