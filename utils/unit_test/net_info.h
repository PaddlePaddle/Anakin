#include "framework/model_parser/parser/model_io.h"
#include "framework/model_parser/parser/parser.h"
#include "framework/core/operator/operator.h"
#include "test/framework/net/net_test.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <fcntl.h>

#if defined(USE_CUDA)
using Target = NV;
using Target_H = X86;
#elif defined(USE_X86_PLACE)
using Target = X86;
using Target_H = X86;
#elif defined(USE_ARM_PLACE)
using Target = ARM;
using Target_H = ARM;
#elif defined(AMD_GPU)
using Target = AMD;
using Target_H = AMDHX86;
#endif

using namespace google::protobuf;

struct TensorConf {
    Shape valid_shape;
};

struct FuncConf {
    std::string name;
    std::string type;
    std::vector<TensorConf> ins;
    std::vector<TensorConf> outs;
    std::shared_ptr<Node> node_ptr;

    AttrInfo& attr() {
        return node_ptr->attr();
    }
};

template<typename Ttype> 
Status fill_tensor_proto(const Tensor<Ttype>* tensor_p, TensorProto* tensor_proto_p) {
    Status ret = Status();
    auto valid_shape = tensor_p->valid_shape();
    LayoutProto layout_proto(LayoutProto(valid_shape.get_layout()));
    for (int i = 0; i < valid_shape.dims(); i++) {
        DLOG(INFO) << "valid_shape: " << valid_shape[i];
        tensor_proto_p->mutable_valid_shape()->mutable_dim()->add_value(valid_shape[i]);
    }
    DLOG(INFO) << "valid_shape size: " << valid_shape.size();
    tensor_proto_p->mutable_valid_shape()->mutable_dim()->set_size(valid_shape.size());
    if (valid_shape.size() == 3) {
        layout_proto = LayoutProto(LAYOUT_NHW);
    } else if (valid_shape.size() == 2) {
        layout_proto = LayoutProto(LAYOUT_NW);
    }
    tensor_proto_p->mutable_valid_shape()->set_layout(layout_proto);
    return ret;
}

template<typename Ttype, Precision Ptype> 
Status fill_func_proto(const OperatorFunc<Ttype, Ptype>& func, FuncProto* func_proto_p) {
    Status ret = Status();
    DLOG(WARNING) << "=======";
    DLOG(INFO) << "Function name: " << func.name;
    DLOG(INFO) << "Function type: " << func.op_name;
    const auto& tensors_in_p = func.ins;
    const auto& tensors_out_p = func.outs;
    DLOG(WARNING) << "In tensor Information";
    for (const auto tensor_in_p: tensors_in_p) {
        auto tensor_in_proto_p = func_proto_p->add_tensor_ins();
        fill_tensor_proto(tensor_in_p, tensor_in_proto_p);
    }
    DLOG(WARNING) << "Out tensor Information";
    for (const auto tensor_out_p: tensors_out_p) {
        auto tensor_out_proto_p = func_proto_p->add_tensor_outs();
        fill_tensor_proto(tensor_out_p, tensor_out_proto_p);
    }
    func_proto_p->set_name(func.name);
    func_proto_p->set_type(func.op_name);
    return ret;
}

template<typename Ttype, Precision Ptype> 
Status fill_net_proto(const Net<Ttype, Ptype>* net, \
    graph::Graph<Ttype, Ptype>* graph_p, NetProto& net_proto) {
    Status ret = Status();
    auto graph_proto_p = net_proto.mutable_graph();
    parser::generate_graph_proto<Ttype, Ptype>(graph_p, *graph_proto_p);
    const auto funcs = net->get_exec_funcs();
    for (const auto& func: funcs) {
        auto func_proto_p = net_proto.add_funcs();
        fill_func_proto(func, func_proto_p);
    }
    return ret;
}

Status read_proto_from_text(const char* filename, Message* proto) {
    Status ret = Status();
    int fd = open(filename, O_RDONLY);
    CHECK_NE(fd, -1) << "File not found: " << filename;
    io::FileInputStream* input = new io::FileInputStream(fd);
    TextFormat::Parse(input, proto);
    delete input;
    close(fd);
    return ret;
}

Shape get_shape_v(TensorProto* tensor_proto_p) {
    std::vector<int> vec;
    auto shape_ptr = tensor_proto_p->mutable_valid_shape();
    auto dim_p = shape_ptr->mutable_dim();
    for (int i = 0; i < dim_p->value_size(); i++) {
        vec.push_back(dim_p->value(i));
    }
    LayoutType layout = static_cast<LayoutType>(shape_ptr->layout());
    Shape valid_shape(vec, layout);
    return valid_shape;
}

Status load_func_proto(FuncConf& func, FuncProto* func_proto_p) {
    Status ret = Status();
    func.name = func_proto_p->name();
    func.type = func_proto_p->type();
    for (int i = 0; i < func_proto_p->tensor_ins_size(); i++) {
        TensorConf tensor;
        auto tensor_proto_p = func_proto_p->mutable_tensor_ins(i);
        tensor.valid_shape = get_shape_v(tensor_proto_p);
        func.ins.push_back(tensor);
    }
    for (int i = 0; i < func_proto_p->tensor_outs_size(); i++) {
        TensorConf tensor;
        auto tensor_proto_p = func_proto_p->mutable_tensor_outs(i);
        tensor.valid_shape = get_shape_v(tensor_proto_p);
        func.outs.push_back(tensor);
    }
    auto node_proto = func_proto_p->node_info();
    parser::NodeIO<Target, Precision::FP32> node_io;
    node_io >> node_proto;
    func.node_ptr = node_io.pop_node_ptr();
    return ret;
}

Status load_funcs_from_proto(NetProto* net_proto_p, std::vector<FuncConf>& funcs) {
    Status ret = Status();
    for (int i = 0; i < net_proto_p->funcs_size(); i++) {
        FuncConf func;
        load_func_proto(func, net_proto_p->mutable_funcs(i));
        funcs.push_back(func);
    }
    return ret;
}

Status load_funcs_from_text(const char* filename, std::vector<FuncConf>& funcs) {
    Status ret = Status();
    NetProto net_proto;
    read_proto_from_text(filename, &net_proto);
    load_funcs_from_proto(&net_proto, funcs);
    return ret;
}

template<typename Ttype, Precision Ptype> 
OperatorBase* create_operator(FuncConf& func) {
    auto op_ptr = OpFactory<Target, Ptype>::Global()[func.type];
    static_cast<Operator<Ttype, Ptype>*>(op_ptr)->_helper->BindParam(func.node_ptr);
    static_cast<Operator<Ttype, Ptype>*>(op_ptr)->_helper->InitParam();
    return op_ptr;
}
