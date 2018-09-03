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

#ifndef ANAKIN_ARC_H
#define ANAKIN_ARC_H 

#include <string>
#include <sstream>
#include <list>
#include <unordered_map>
#include "utils/logger/logger.h"

namespace anakin {

namespace graph {

/// \brief class arc.
template<typename VertexNameType, typename WeightType>
class Arc {
public:    
    Arc() {}
    Arc(VertexNameType vertex_1, VertexNameType vertex_2);
    Arc(VertexNameType vertex_1, VertexNameType vertex_2, WeightType weight);
    Arc(const Arc& otherArc);
    virtual ~Arc() {}
	
    /// judge if one arc equal to another
    bool operator==(const Arc<VertexNameType, WeightType>& otherArc) const {
        return (_vertex_1 == otherArc._vertex_1 && _vertex_2 == otherArc._vertex_2);
    }

    Arc& operator=(const Arc& otherArc);

    inline std::string name();

    /// get bottom _vertex_1 
    inline VertexNameType& bottom();
    /// get top_vertex_2
    inline VertexNameType& top();

    /// get weight.
    inline WeightType& weight();
    
private:
    ///<  _vertex_1 stand for bottom vertex
    VertexNameType _vertex_1;  
     ///< _vertex_2 stand for top vertex      
    VertexNameType _vertex_2;   
    /// the weight of arc between _vertex_1 and _vertex_2   
    WeightType _weight;        
};

/// \brief iterator class of arc.in graph
template<typename VertexNameType, typename WeightType, typename ArcType = Arc<VertexNameType, WeightType> >
class Arc_iterator {
public:
    Arc_iterator() {}
    Arc_iterator(const Arc_iterator<VertexNameType, WeightType, ArcType>& rhs):_arc_it(rhs._arc_it) {}
    Arc_iterator(typename std::list<ArcType>::iterator& it):_arc_it(it) {}
    Arc_iterator(typename std::list<ArcType>::iterator&& it):_arc_it(it) {}

    /// copy operation
    Arc_iterator<VertexNameType, WeightType, ArcType>& operator=(const Arc_iterator<VertexNameType, WeightType, ArcType>& rhs) {
        _arc_it = rhs._arc_it;
        return *(this);
    }
    /// Prefix ++
    Arc_iterator<VertexNameType, WeightType, ArcType>& operator++() { _arc_it++; return *(this); }

    /// Postfix ++
    Arc_iterator<VertexNameType, WeightType, ArcType> operator++(int) { 
        Arc_iterator<VertexNameType, WeightType, ArcType> tmp(*this);
        operator++(); 
        return tmp; 
    }

    inline typename std::list<ArcType>::iterator& origin() { return _arc_it; }

    bool operator==(const Arc_iterator<VertexNameType, WeightType, ArcType>& rhs) const { return _arc_it==rhs._arc_it; }
    bool operator!=(const Arc_iterator<VertexNameType, WeightType, ArcType>& rhs) const { return _arc_it!=rhs._arc_it; }
    ArcType& operator*() { return *_arc_it; }
    ArcType* operator->() { return &(*_arc_it); }

private:
    ///< _arc_it stand for the list of arc
    typename std::list<ArcType>::iterator _arc_it;
};

} /* graph */

} /* anakin */

#include "arc.inl"

#endif /* ANAKIN_ARC_H */
