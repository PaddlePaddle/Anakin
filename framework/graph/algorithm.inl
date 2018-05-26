namespace anakin {

namespace graph {


/************************************************************************\
 *                  Algorithm functor Has Return
\************************************************************************/
template<typename VertexNameType, 
         typename VertexType, 
         typename WeightType, 
         typename ArcType>
template<typename functor, typename ...ParamTypes>
Algorithm<VertexNameType, VertexType, WeightType, ArcType>& Algorithm<VertexNameType, VertexType, WeightType, ArcType>::_DFS_Edge(Bool2Type<true>, functor& func, ParamTypes&& ...args) {
    LOG(WARNING) << "Not impl yet , which isn't so important in inference analysis";
    return *this;
}

template<typename VertexNameType, 
         typename VertexType, 
         typename WeightType, 
         typename ArcType>
template<typename functor, typename ...ParamTypes>
Algorithm<VertexNameType, VertexType, WeightType, ArcType>& Algorithm<VertexNameType, VertexType, WeightType, ArcType>::_BFS_Edge(Bool2Type<true>, functor& func, ParamTypes&& ...args) {
    std::queue<VertexNameType> que;
    std::vector<VertexNameType> backup;
    auto ins = this->_graph->get_graph_ins();
    CHECK_GT(ins.size(), 0) << " The graph don't have any inputs";
    for(int i = 0; i < ins.size(); i++) {
        VertexNameType vertex_name = ins[i];
        if(std::find(backup.begin(), backup.end(), vertex_name) == backup.end()) {
            que.push(vertex_name);
            backup.push_back(vertex_name);
        }
        /*auto arc_outs = this->_graph->get_out_arcs(vertex_name);
        for(auto& arc : arc_outs) {
            func(arc);
            VertexNameType vertex_name = arc.top();
            if(std::find(backup.begin(), backup.end(), vertex_name) == backup.end()) { // not find
                que.push(vertex_name);
                backup.push_back(vertex_name);
            }
        }*/
    }
    while(!que.empty()) {
        auto& vertex_name = que.front();

        auto& arc_out_its = this->_graph->get_out_arc_its(vertex_name);
        for(auto& arc_it : arc_out_its) {
            auto ret = func(*arc_it, std::forward<ParamTypes>(args)...);
            if(ret == Status::EXIT()) { 
                return *this; 
            }

            VertexNameType vertex_name = arc_it->top();
            if(std::find(backup.begin(), backup.end(), vertex_name) == backup.end()) { // not find
                que.push(vertex_name);
                backup.push_back(vertex_name);
            }
        }
            
        que.pop();
    }
    return *this;
}

template<typename VertexNameType, 
         typename VertexType, 
         typename WeightType, 
         typename ArcType>
template<typename functor, typename ...ParamTypes>
Algorithm<VertexNameType, VertexType, WeightType, ArcType>& Algorithm<VertexNameType, VertexType, WeightType, ArcType>::_DFS(Bool2Type<true>, functor& func, ParamTypes&& ...args) {
    LOG(WARNING) << "Not impl yet , which isn't so important in inference analysis";
    return *this;
}

template<typename VertexNameType, 
         typename VertexType, 
         typename WeightType, 
         typename ArcType>
template<typename functor, typename ...ParamTypes>
Algorithm<VertexNameType, VertexType, WeightType, ArcType>& Algorithm<VertexNameType, VertexType, WeightType, ArcType>::_BFS(Bool2Type<true>, functor& func, ParamTypes&& ...args) {
    std::queue<VertexNameType> que;
    std::vector<VertexNameType> backup;
    auto ins = this->_graph->get_graph_ins();
    CHECK_GT(ins.size(), 0) << " The graph don't have any inputs";
    for(int i = 0; i < ins.size(); i++) {
        VertexNameType vertex_name = ins[i];
        if(std::find(backup.begin(), backup.end(), vertex_name) == backup.end()) {
            que.push(vertex_name);
            backup.push_back(vertex_name);
        }
        // Code below is useful when anakin doesn't define input op
        /*auto arc_outs = this->_graph->get_out_arcs(vertex_name);
        for(auto& arc : arc_outs) {
            VertexNameType vertex_name = arc.top();
            if(std::find(backup.begin(), backup.end(), vertex_name) == backup.end()) { // not find
                que.push(vertex_name);
                backup.push_back(vertex_name);
            }
        }*/
    }
    while(!que.empty()) {
        auto& vertex_name = que.front();
        VertexType& vertex =  (*(this->_graph))[vertex_name];

        auto ret = func(vertex, std::forward<ParamTypes>(args)...);
        if(ret == Status::EXIT()) {
            return *this;
        }

        auto arc_out_its = this->_graph->get_out_arc_its(vertex_name);
        for(auto& arc_it : arc_out_its) {
            VertexNameType vertex_name = arc_it->top();
            if(std::find(backup.begin(), backup.end(), vertex_name) == backup.end()) { // not find
                que.push(vertex_name);
                backup.push_back(vertex_name);
            }
        }
            
        que.pop();
    }
    return *this;
}

/************************************************************************\
 *                  Algorithm functor Hasn't Return
\************************************************************************/

template<typename VertexNameType, 
         typename VertexType, 
         typename WeightType, 
         typename ArcType>
template<typename functor, typename ...ParamTypes>
Algorithm<VertexNameType, VertexType, WeightType, ArcType>& Algorithm<VertexNameType, VertexType, WeightType, ArcType>::_DFS_Edge(Bool2Type<false>, functor& func, ParamTypes&& ...args) {
    LOG(WARNING) << "Not impl yet , which isn't so important in inference analysis";
    return *this;
}

template<typename VertexNameType, 
         typename VertexType, 
         typename WeightType, 
         typename ArcType>
template<typename functor, typename ...ParamTypes>
Algorithm<VertexNameType, VertexType, WeightType, ArcType>& Algorithm<VertexNameType, VertexType, WeightType, ArcType>::_BFS_Edge(Bool2Type<false>, functor& func, ParamTypes&& ...args) {
    std::queue<VertexNameType> que;
    std::vector<VertexNameType> backup;
    auto ins = this->_graph->get_graph_ins();
    CHECK_GT(ins.size(), 0) << " The graph don't have any inputs";
    for(int i = 0; i < ins.size(); i++) {
        VertexNameType vertex_name = ins[i];
        if(std::find(backup.begin(), backup.end(), vertex_name) == backup.end()) {
            que.push(vertex_name);
            backup.push_back(vertex_name);
        }
        /*auto arc_outs = this->_graph->get_out_arcs(vertex_name);
        for(auto& arc : arc_outs) {
            func(arc);
            VertexNameType vertex_name = arc.top();
            if(std::find(backup.begin(), backup.end(), vertex_name) == backup.end()) { // not find
                que.push(vertex_name);
                backup.push_back(vertex_name);
            }
        }*/
    }
    while(!que.empty()) {
        auto& vertex_name = que.front();

        auto& arc_out_its = this->_graph->get_out_arc_its(vertex_name);
        for(auto& arc_it : arc_out_its) {
            func(*arc_it, std::forward<ParamTypes>(args)...);

            VertexNameType vertex_name = arc_it->top();
            if(std::find(backup.begin(), backup.end(), vertex_name) == backup.end()) { // not find
                que.push(vertex_name);
                backup.push_back(vertex_name);
            }
        }
            
        que.pop();
    }
    return *this;
}

template<typename VertexNameType, 
         typename VertexType, 
         typename WeightType, 
         typename ArcType>
template<typename functor, typename ...ParamTypes>
Algorithm<VertexNameType, VertexType, WeightType, ArcType>& Algorithm<VertexNameType, VertexType, WeightType, ArcType>::_DFS(Bool2Type<false>, functor& func, ParamTypes&& ...args) {
    LOG(WARNING) << "Not impl yet , which isn't so important in inference analysis";
    return *this;
}

template<typename VertexNameType, 
         typename VertexType, 
         typename WeightType, 
         typename ArcType>
template<typename functor, typename ...ParamTypes>
Algorithm<VertexNameType, VertexType, WeightType, ArcType>& Algorithm<VertexNameType, VertexType, WeightType, ArcType>::_BFS(Bool2Type<false>, functor& func, ParamTypes&& ...args) {
    std::queue<VertexNameType> que;
    std::vector<VertexNameType> backup;
    auto ins = this->_graph->get_graph_ins();
    CHECK_GT(ins.size(), 0) << " The graph don't have any inputs";
    for(int i = 0; i < ins.size(); i++) {
        VertexNameType vertex_name = ins[i];
        if(std::find(backup.begin(), backup.end(), vertex_name) == backup.end()) {
            que.push(vertex_name);
            backup.push_back(vertex_name);
        }
        // Code below is useful when anakin doesn't define input op
        /*auto arc_outs = this->_graph->get_out_arcs(vertex_name);
        for(auto& arc : arc_outs) {
            VertexNameType vertex_name = arc.top();
            if(std::find(backup.begin(), backup.end(), vertex_name) == backup.end()) { // not find
                que.push(vertex_name);
                backup.push_back(vertex_name);
            }
        }*/
    }
    while(!que.empty()) {
        auto& vertex_name = que.front();
        VertexType& vertex =  (*(this->_graph))[vertex_name];
        func(vertex, std::forward<ParamTypes>(args)...);

        auto arc_out_its = this->_graph->get_out_arc_its(vertex_name);
        for(auto& arc_it : arc_out_its) {
            VertexNameType vertex_name = arc_it->top();
            if(std::find(backup.begin(), backup.end(), vertex_name) == backup.end()) { // not find
                que.push(vertex_name);
                backup.push_back(vertex_name);
            }
        }
            
        que.pop();
    }
    return *this;
}


} /* namespace graph */

} /* namespace anakin */
