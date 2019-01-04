#include "test/framework/net/debug_info.h"
#include "framework/operators/softmax.h"
#include <assert.h>

std::string g_load_path;

Status read_cmd(int argc, char** argv, std::string& load_path) {
    Status ret = Status();
    if (argc < 2) {
        LOG(FATAL) << "Usage: ./bin load_path";
    }
    if (argc > 1) {
        load_path = std::string(argv[1]);
    }
    return ret;
}

int main(int argc, char** argv) {
    read_cmd(argc, argv, g_load_path);
    std::vector<FuncConf> funcs;
    load_funcs_from_text(g_load_path.c_str(), funcs);
    for (auto& func: funcs) {
        assert(func.type == "Softmax");
        auto op_ptr = create_operator<Target, Precision::FP32>(func);
        auto helper_ptr = static_cast<ops::Softmax<Target, Precision::FP32>*>(op_ptr)->_helper;
        auto param_softmax = static_cast<ops::SoftmaxHelper<Target, Precision::FP32>*>(helper_ptr)->_param_softmax;
        LOG(INFO) << "param_softmax.axis: " << param_softmax.axis;
    }
    return 0;
}

