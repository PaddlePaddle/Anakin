
#include <string>
#include "saber/funcs/timer.h"
#include <chrono>
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <dirent.h> 
#include <sys/stat.h> 
#include <sys/types.h> 
#include <unistd.h>  
#include <fcntl.h>
#include <map>


class EngineBase {
 public:
  // Build the model and do some preparation, for example, in TensorRT, run
  // createInferBuilder, buildCudaEngine.
  virtual void Build(const std::string& model_file, int batch_size = 1) = 0;
  virtual void Build(const std::string& model_file, const std::vector<int>& shape) = 0;
  // Execute the engine, that will run the inference network.
  virtual void Execute() = 0;

  virtual ~EngineBase() {}
};  // class EngineBase

template <typename Ttype, DataType Dtype, Precision Ptype>
class AnakinEngine : public EngineBase {
public:
  typedef typename anakin::saber::DataTrait<Dtype>::dtype Dtype_t;
  typedef anakin::saber::TargetWrapper<X86> X86_API;
  typedef anakin::saber::TargetWrapper<Ttype> NV_API;
  AnakinEngine(){}

  ~AnakinEngine(){};

  void Build(const std::string& model_file, int batch_size = 1) override
  {
    _graph.load(model_file);
    _graph.ResetBatchSize("input_0", batch_size);
    _graph.Optimize();
    _net_executer.init(_graph);
  };

  void Build(const std::string& model_file, const std::vector<int>& shape) override
  {
    _graph.load(model_file);
    _graph.Reshape("input_0", shape);
    _graph.Optimize();
    _net_executer.init(_graph);
  };

  void Execute() override
  {
    _net_executer.prediction();    
  };

  // Fill an input from CPU memory with name and size.
  void SetInputFromCPU(const std::string name, Dtype_t* data, size_t size)
  {
    auto input_tensor = _net_executer.get_in(name);
    anakin::Tensor<Ttype, Dtype> tmp_tensor(data, anakin::saber::X86(), X86_API::get_device_id(), input_tensor->valid_shape());
    *input_tensor = tmp_tensor;
  };

  // accessed directly. Fill an input from GPU memory with name and size.
  void SetInputFromGPU(const std::string& name, Dtype_t* data, size_t size)
  {
    auto input_tensor = _net_executer.get_in(name);
    CHECK_EQ(size, input_tensor->valid_size());
    anakin::Tensor<Ttype, Dtype> tmp_tensor(data, anakin::saber::NV(), NV_API::get_device_id(), input_tensor->valid_shape());
    *input_tensor = tmp_tensor;
  };
  // Get an output called name, the output of tensorrt is in GPU, so this method
  // will just return the output's GPU memory address.
  anakin::Tensor<Ttype, Dtype>* GetOutputInGPU(const std::string& name)
  {
    return _net_executer.get_out(name);
  }

private:
    anakin::graph::Graph<Ttype, Dtype, Ptype> _graph;
    anakin::Net<Ttype, Dtype, Ptype> _net_executer;
};  // class TensorRTEngine
template 
class AnakinEngine<anakin::NV, anakin::saber::AK_FLOAT, anakin::Precision::FP32>;


