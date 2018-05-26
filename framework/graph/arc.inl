
namespace anakin {

namespace graph {

template<typename VertexNameType, typename WeightType>
Arc<VertexNameType, WeightType>::Arc(VertexNameType vertex_1, VertexNameType vertex_2) {
    this->_vertex_1 = vertex_1;
    this->_vertex_2 = vertex_2;
}

template<typename VertexNameType, typename WeightType>
Arc<VertexNameType, WeightType>::Arc(VertexNameType vertex_1, VertexNameType vertex_2, WeightType weight) {
    this->_vertex_1 = vertex_1;
    this->_vertex_2 = vertex_2;
    this->_weight = weight; 
}

template<typename VertexNameType, typename WeightType>
Arc<VertexNameType, WeightType>::Arc(const Arc& otherArc) {
    this->_vertex_1 = otherArc._vertex_1; 
    this->_vertex_2 = otherArc._vertex_2; 
    this->_weight = otherArc._weight;
}

template<typename VertexNameType, typename WeightType>
Arc<VertexNameType, WeightType>& Arc<VertexNameType, WeightType>::operator=(const Arc& otherArc) {
    this->_vertex_1 = otherArc._vertex_1;
    this->_vertex_2 = otherArc._vertex_2;
    this->_weight = otherArc._weight;
}


template<typename VertexNameType, typename WeightType>
inline VertexNameType& Arc<VertexNameType, WeightType>::bottom() {
    return this->_vertex_1;
}

template<typename VertexNameType, typename WeightType>
inline VertexNameType& Arc<VertexNameType, WeightType>::top() {
    return this->_vertex_2;
}

template<typename VertexNameType, typename WeightType>
inline WeightType& Arc<VertexNameType, WeightType>::weight() {
    return this->_weight;
}

template<typename VertexNameType, typename WeightType>
inline std::string Arc<VertexNameType, WeightType>::name() {
    auto name = std::string(this->_vertex_1) + "_" + std::string(this->_vertex_2);
    return name;
}

} /* namespace graph */

} /* namespace anakin */
