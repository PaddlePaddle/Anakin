#include "framework/model_parser/parser/model_io.h"
#include "framework/core/operator/operator.h"
#include "framework/core/parameter.h"

namespace anakin {

namespace parser {

template<typename Ttype, Precision Ptype>
NodeIO<Ttype, Ptype>& NodeIO<Ttype, Ptype>::operator>>(const NodeProto& node_proto) {
    graph::NodePtr node_p = std::make_shared<graph::Node>();

    auto it_end = _node_name2ptr_map.end();
    auto it_find = _node_name2ptr_map.find(node_proto.name());
    if(it_find == it_end) {
        _node_name2ptr_map[node_proto.name()] = node_p;
    }

    node_p->name() = node_proto.name();
    node_p->need_wait() = node_proto.need_wait();
    node_p->lane() = node_proto.lane();
    switch (node_proto.bit_type()) {
        case INT8: node_p->bit_type() = AK_INT8; break;
        case FLOAT: node_p->bit_type() = AK_FLOAT; break;
        default: node_p->bit_type() = AK_INVALID; break;
    }
    DLOG(INFO) << "read node: " << node_p->name() << \
    " (type: " << node_p->bit_type() << " )";

    auto it = node_proto.attr().begin();
    for (; it != node_proto.attr().end(); ++it) {
        auto& key = it->first;
        auto& value = it->second;

        switch (value.type()) {
        case STR: {
            node_p->set_attr(key, value.s());
        }
        break;

        case INT32: {
            node_p->set_attr(key, value.i());
        }
        break;

        case FLOAT: {
            node_p->set_attr(key, value.f());
        }
        break;

        case DOUBLE: {
            node_p->set_attr(key, value.f());
        }
        break;

        case BOOLEN: {
            node_p->set_attr(key, value.b());
        }
        break;

        case CACHE_LIST: {
            auto& cache_data = value.cache_list();

            switch (cache_data.type()) {
            case FLOAT: {
                PTuple<float> list_data;

                for (int i = 0; i < cache_data.size(); i++) {
                    list_data.push_back(cache_data.f()[i]);
                }

                node_p->set_attr(key, list_data);
            }
            break;

            case BOOLEN: {
                PTuple<bool> list_data;

                for (int i = 0; i < cache_data.size(); i++) {
                    list_data.push_back(cache_data.b()[i]);
                }

                node_p->set_attr(key, list_data);
            }
            break;

            case INT32: {
                PTuple<int> list_data;

                for (int i = 0; i < cache_data.size(); i++) {
                    list_data.push_back(cache_data.i()[i]);
                }

                node_p->set_attr(key, list_data);
            }
            break;

            case STR: {
                PTuple<std::string> list_data;

                for (int i = 0; i < cache_data.size(); i++) {
                    list_data.push_back(cache_data.s()[i]);
                }

                node_p->set_attr(key, list_data);
            }
            break;

            case CACHE_LIST: {
                auto& tmp_cache_data = cache_data.l()[0];

                switch (tmp_cache_data.type()) {
                case INT32: { // now only support int Recursive Structures of list
                    PTuple<PTuple<int>> list_list_data;

                    for (int index = 0; index < cache_data.l().size(); index++) {
                        auto& tmp_inner_cache_data = cache_data.l()[index];
                        list_list_data.push_back(PTuple<int>());

                        for (int i = 0; i < tmp_inner_cache_data.size(); i++) {
                            list_list_data[index].push_back(tmp_inner_cache_data.i()[i]);
                        }
                    }

                    node_p->set_attr(key, list_list_data);
                }
                break;

                default : {
                    LOG(FATAL) << "UnSupport Recursive list data type(DateTypeProto:" << cache_data.type() <<
                               ") in list ";
                }
                }
            }
            break;

            default : {
                LOG(FATAL) << "UnSupport data type(DateTypeProto:" << cache_data.type() << ") in list ";
            }
            break;
            }
        }
        break;

        case TENSOR: {
            auto& tensor = value.tensor();
            if(tensor.shared()) { // cope with shared weights(tensor)
                auto target_node = _node_name2ptr_map[tensor.share_from()]-> template get_attr<PBlock<Ttype> >(key);
                node_p->set_attr(key, target_node);
                // record share info of weights
                node_p->set_share_pair(key, tensor.share_from());
            } else {
                auto& real_shape = tensor.shape();
                auto& valid_shape = tensor.valid_shape();
                CHECK_EQ(real_shape.dim().size(), 4) << "Weights parameter's shape len must equal to 4.";
                auto& data = tensor.data();
                auto& scale = tensor.scale().f();
                std::vector<float> scale_vector;
                for (const float val: scale) {
                    scale_vector.push_back(val);
                }

                switch (data.type()) {
                case FLOAT: { /* At so far, we only support weights saved as float. */
                    saber::Shape saber_shape({1, 1, 1, 1});

                    // get real_shape
                    for (int i = 0; i < 4; i++) {
                        saber_shape[i] = real_shape.dim().value()[i];
                    }

                    auto* block = graph::GraphGlobalMem<Ttype>::Global().template new_block<AK_FLOAT>(saber_shape);
                    // fill data to block
                    float* cpu_data = static_cast<float*>(block->h_tensor().mutable_data());

                    for (int i = 0; i < data.size(); i++) {
                        cpu_data[i] = data.f()[i];
                    }
                    block->d_tensor().set_scale(scale_vector);
                    block->h_tensor().set_scale(scale_vector);

#if defined(    USE_CUDA) || defined(AMD_GPU)
                    // map cpu data to GPU
                    block->d_tensor().set_shape(saber_shape);
                    block->d_tensor().copy_from(block->h_tensor());
#endif
                    if (valid_shape.dim().size() == 0) {
                        // set valid shape (== real shape) for host and device
                        block->d_tensor().set_shape(saber_shape);
                        block->h_tensor().set_shape(saber_shape);
                    } else {
                        saber::Shape saber_valid_shape({1, 1, 1, 1});
                        for (int i=0; i < 4; i++) {
                            saber_valid_shape[i] = valid_shape.dim().value()[i];
                        }
                        // set valid shape for host and device
                        block->d_tensor().set_shape(saber_valid_shape);
                        block->h_tensor().set_shape(saber_valid_shape);
                    }

                    node_p->set_attr(key, *block);
                }
                break;
                case INT8: { /* At so far, we only support weights saved as float. */
                    saber::Shape saber_shape({1, 1, 1, 1});

                    // get real_shape
                    for (int i = 0; i < 4; i++) {
                        saber_shape[i] = real_shape.dim().value()[i];
                    }

                    auto* block = graph::GraphGlobalMem<Ttype>::Global().template new_block<AK_INT8>(saber_shape);
                    // fill data to block
                    char* cpu_data = static_cast<char*>(block->h_tensor().mutable_data());
                    for (int i = 0; i < data.size(); i++) {
                        cpu_data[i] = data.c().data()[i];
                    }
                    block->d_tensor().set_scale(scale_vector);
                    block->h_tensor().set_scale(scale_vector);

#if defined(    USE_CUDA) || defined(AMD_GPU)
                    // map cpu data to GPU
                    block->d_tensor().set_shape(saber_shape);
                    block->d_tensor().copy_from(block->h_tensor());
#endif
                    if (valid_shape.dim().size() == 0) {
                        // set valid shape (== real shape) for host and device
                        block->d_tensor().set_shape(saber_shape);
                        block->h_tensor().set_shape(saber_shape);
                    } else {
                        saber::Shape saber_valid_shape({1, 1, 1, 1});
                        for (int i = 0; i < 4; i++) {
                            saber_valid_shape[i] = valid_shape.dim().value()[i];
                        }
                        // set valid shape for host and device
                        block->d_tensor().set_shape(saber_valid_shape);
                        block->h_tensor().set_shape(saber_valid_shape);
                    }

                    node_p->set_attr(key, *block);
                }
                break;
                default : {
                    LOG(FATAL) << "UnSupport data type(DateTypeProto:" << data.type() << ") in list ";
                }
                break;
                }
            } // not shared
        }
        break;

        default : {
            LOG(FATAL) << "Unknown data type ( DateTypeProto:" << value.type() << ") in message valueType";
        }
        break;
        }
    }

    const auto& op = node_proto.op();
    /* fill node with operator from proto [Deprecated. replace by graph.cpp] */
    node_p->get_op_name() = op.name();
    _que.push(node_p);
    return *this;
}

template<typename Ttype, Precision Ptype>
NodeIO<Ttype, Ptype>& NodeIO<Ttype, Ptype>::operator>>(const
        graph::NodePtr& node_p) {
    _que.push(node_p);
    return *this;
}

template<typename Ttype, Precision Ptype>
Status NodeIO<Ttype, Ptype>::operator<<(graph::Graph<Ttype, Ptype>& graph) {
    while (!this->empty()) {
        auto& node_p = _que.front();
        DLOG(WARNING) << "[NODE] Graph get node: " << node_p->name();
        graph.add_vertex(node_p->name(), node_p);

        if (node_p->get_op_name() != "Output") {
            _que_node_name_in_order.push_back(node_p->name());
        }

        _que.pop();
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status NodeIO<Ttype, Ptype>::operator<<(GraphProto& graph) {
    while (!this->empty()) {
        auto& node_p = _que.front();
        NodeProto* node_proto = graph.add_nodes();
        // set node proto name/lane/need_wait
        node_proto->set_name(node_p->name());
        node_proto->set_lane(node_p->lane());
        node_proto->set_need_wait(node_p->need_wait());
        // set node proto's  op proto
        OpProto* op = node_proto->mutable_op();
        op->set_name(node_p->get_op_name());

        // set node proto's attr
        auto node_proto_attr = node_proto->mutable_attr();
        auto it = node_p->attr().begin();

        for (; it != node_p->attr().end(); ++it) {
            auto& key = it->first;
            auto& value = it->second;

            //(*node_proto_attr)[key] = ;
            if (value.type() == "anakin_string") {
                (*node_proto_attr)[key].set_s(any_cast<std::string>(value));
                (*node_proto_attr)[key].set_type(STR);
            } else if (value.type() == "anakin_int32") {
                (*node_proto_attr)[key].set_i(any_cast<int>(value));
                (*node_proto_attr)[key].set_type(INT32);
            } else if (value.type() == "anakin_float") {
                (*node_proto_attr)[key].set_f(any_cast<float>(value));
                (*node_proto_attr)[key].set_type(FLOAT);
            } else if (value.type() == "anakin_bool") {
                (*node_proto_attr)[key].set_b(any_cast<bool>(value));
                (*node_proto_attr)[key].set_type(BOOLEN);
            } else if (value.type() == "anakin_tuple_string") {
                auto tuple_string = any_cast<PTuple<std::string>>(value);

                for (int i = 0; i < tuple_string.size(); i++) {
                    (*node_proto_attr)[key].mutable_cache_list()->add_s(tuple_string[i]);
                }

                (*node_proto_attr)[key].set_type(CACHE_LIST);
                (*node_proto_attr)[key].mutable_cache_list()->set_type(STR);
                (*node_proto_attr)[key].mutable_cache_list()->set_size(tuple_string.size());
            } else if (value.type() == "anakin_tuple_int") {
                auto tuple_int = any_cast<PTuple<int>>(value);

                for (int i = 0; i < tuple_int.size(); i++) {
                    (*node_proto_attr)[key].mutable_cache_list()->add_i(tuple_int[i]);
                }

                (*node_proto_attr)[key].set_type(CACHE_LIST);
                (*node_proto_attr)[key].mutable_cache_list()->set_type(INT32);
                (*node_proto_attr)[key].mutable_cache_list()->set_size(tuple_int.size());
            } else if (value.type() == "anakin_tuple_tuple_int") { // in case CACHE_LIST in CACHE_LIST
                auto tuple_tuple_int = any_cast<PTuple<PTuple<int>>>(value);

                for (int i = 0; i < tuple_tuple_int.size(); i++) {
                    auto* cach_list = (*node_proto_attr)[key].mutable_cache_list()->add_l();

                    for (int j = 0; j < tuple_tuple_int[i].size(); j++) {
                        cach_list->add_i(tuple_tuple_int[i][j]);
                    }

                    cach_list->set_type(INT32);
                    cach_list->set_size(tuple_tuple_int[i].size());
                }

                (*node_proto_attr)[key].set_type(CACHE_LIST);
                (*node_proto_attr)[key].mutable_cache_list()->set_type(CACHE_LIST);
                (*node_proto_attr)[key].mutable_cache_list()->set_size(tuple_tuple_int.size());
            } else if (value.type() == "anakin_tuple_float") {
                auto tuple_float = any_cast<PTuple<float>>(value);

                for (int i = 0; i < tuple_float.size(); i++) {
                    (*node_proto_attr)[key].mutable_cache_list()->add_f(tuple_float[i]);
                }

                (*node_proto_attr)[key].set_type(CACHE_LIST);
                (*node_proto_attr)[key].mutable_cache_list()->set_type(FLOAT);
                (*node_proto_attr)[key].mutable_cache_list()->set_size(tuple_float.size());
            } else if (value.type() == "anakin_tuple_bool") {
                auto tuple_bool = any_cast<PTuple<bool>>(value);

                for (int i = 0; i < tuple_bool.size(); i++) {
                    (*node_proto_attr)[key].mutable_cache_list()->add_b(tuple_bool[i] == "true" ? true : false);
                }

                (*node_proto_attr)[key].set_type(CACHE_LIST);
                (*node_proto_attr)[key].mutable_cache_list()->set_type(BOOLEN);
                (*node_proto_attr)[key].mutable_cache_list()->set_size(tuple_bool.size());
            } else if (value.type() == "anakin_block") { // default block have float data
                // cope with shared weights
                if (node_p->check_shared(key)) {
                    auto share_target = node_p->get_share_target(key);
                    (*node_proto_attr)[key].mutable_tensor()->set_shared(true);
                    (*node_proto_attr)[key].mutable_tensor()->set_share_from(share_target);
                    (*node_proto_attr)[key].set_type(TENSOR);
                } else {
                    auto block_float = any_cast<PBlock<Ttype>>(value);
                    float* cpu_data = static_cast<float*>(block_float.h_tensor().mutable_data());
                    auto valid_shape = block_float.shape();
                    auto real_shape = block_float.real_shape();

                    if (valid_shape == real_shape) {
                        // set proto tensor shape
                        for (int i = 0; i < valid_shape.dims(); i++) {
                            (*node_proto_attr)[key].mutable_tensor()->mutable_shape()->mutable_dim()->add_value(valid_shape[i]);
                        }

                        (*node_proto_attr)[key].mutable_tensor()->mutable_shape()->mutable_dim()->set_size(
                            valid_shape.size());

                        // set proto tensor data
                        for (int i = 0; i < valid_shape.count(); i++) {
                            (*node_proto_attr)[key].mutable_tensor()->mutable_data()->add_f(cpu_data[i]);
                        }

                        (*node_proto_attr)[key].mutable_tensor()->mutable_data()->set_type(FLOAT);
                        (*node_proto_attr)[key].mutable_tensor()->mutable_data()->set_size(valid_shape.count());
                        (*node_proto_attr)[key].set_type(TENSOR);
                    } else {
                        // set proto tensor valid shape
                        for (int i = 0; i < valid_shape.dims(); i++) {
                            (*node_proto_attr)[key].mutable_tensor()->mutable_valid_shape()->mutable_dim()->add_value(valid_shape[i]);
                        }
                        (*node_proto_attr)[key].mutable_tensor()->mutable_valid_shape()->mutable_dim()->set_size(
                            valid_shape.size());

                        // set proto tensor real shape
                        for (int i = 0; i < real_shape.dims(); i++) {
                            (*node_proto_attr)[key].mutable_tensor()->mutable_shape()->mutable_dim()->add_value(real_shape[i]);
                        }
                        (*node_proto_attr)[key].mutable_tensor()->mutable_shape()->mutable_dim()->set_size(
                            real_shape.size());

                        // set proto tensor data
                        for (int i = 0; i < real_shape.count(); i++) {
                            (*node_proto_attr)[key].mutable_tensor()->mutable_data()->add_f(cpu_data[i]);
                        }

                        (*node_proto_attr)[key].mutable_tensor()->mutable_data()->set_type(FLOAT);
                        (*node_proto_attr)[key].mutable_tensor()->mutable_data()->set_size(real_shape.count());
                        (*node_proto_attr)[key].set_type(TENSOR);
                    }
        }
            } else {
                auto tuple_float = any_cast<PTuple<float>>(value);
                (*node_proto_attr)[key].set_type(CACHE_LIST);
                (*node_proto_attr)[key].mutable_cache_list()->set_type(FLOAT);
                (*node_proto_attr)[key].mutable_cache_list()->set_size(tuple_float.size());

                //LOG(ERROR) << "node: " << node_p->name() << " (" << node_p->get_op_name() << ") \
                    key : " << key << " value_type: " << value.type();
            }
        }

        _que.pop();
    }

    return Status::OK();
}

#ifdef USE_CUDA
template class NodeIO<NV, Precision::FP32>;
template class NodeIO<NV, Precision::FP16>;
template class NodeIO<NV, Precision::INT8>;
#endif

#ifdef AMD_GPU
template class NodeIO<AMD, Precision::FP32>;
template class NodeIO<AMD, Precision::FP16>;
template class NodeIO<AMD, Precision::INT8>;
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
template class NodeIO<X86, Precision::FP32>;
template class NodeIO<X86, Precision::FP16>;
template class NodeIO<X86, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
template class NodeIO<ARM, Precision::FP32>;
#endif

#ifdef ANAKIN_TYPE_FP16
template class NodeIO<ARM, Precision::FP16>;
#endif

#ifdef ANAKIN_TYPE_INT8
template class NodeIO<ARM, Precision::INT8>;
#endif

#endif

} /* parser */

} /* anakin */
