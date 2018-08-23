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

#ifndef ANAKIN_SABER_LITE_NET_NET_LITE_H
#define ANAKIN_SABER_LITE_NET_NET_LITE_H

#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"
#include "saber/lite/net/saber_factory_lite.h"

namespace anakin{

namespace saber{

namespace lite {

//template <TargetType Ttype>
class Net {
public:
    /**
     * \brief empty constructor
     */
    Net(PowerMode mode = SABER_POWER_HIGH, int threads = 1);

    /**
     * \brief clear and destroy
     */
    ~Net();

    /**
     * \brief set the mode to execute the model
     * @param mode: SABER_POWER_HIGH for big core, SABER_POWER_LOW for small core, SABER_POWER_FULL for all cores
     * @param threads: number of threads or cores to execute the model
     * @return if success, return SaberSuccess
     */
    SaberStatus set_run_mode(PowerMode mode, int threads);

    /**
     * \brief we could not read the cache size of cpus, the default L1 data cache size is 32Kb,
     * the default L2 cache size is 2Mb, to achive better performance, set your cpu cache size here
     * @param l1_cache: l1 data cache size
     * @param l2_cache: l2 cache size
     * @return return SaberSuccess if successed
     */
    SaberStatus set_device_cache(size_t l1_cache, size_t l2_cache);

    /**
     * \brief load model from modelpath
     * @param lite_model_path:  merged lite model path
     * @return return SaberSuccess if success
     */
    SaberStatus load_model(const char* lite_model_path);

    /**
     * \brief load model info and  weights separately from modelpath
     * @param info_path: lite model info path
     * @param weights_path: lite model weights path
     * @return return SaberSuccess if success
     */
    SaberStatus load_model(const char* info_path, const char* weights_path);

    /**
     * \brief load merged model weights from memory
     * @param merged_memory:  model from memory
     * @return return SaberSuccess if success
     */
    SaberStatus load_model(const void* merged_memory, size_t mem_size);

    /**
     * \brief load model info and  weights separately from memory
     * @param merged_memory:  model from memory
     * @return return SaberSuccess if success
     */
    SaberStatus load_model(const void* info_memory, size_t info_size, const void* weights_memory, size_t weights_size);

    /**
     * \brief unload network structure and weight data
     */
    //void clear();

    /**
     * \brief get input tensor
     * @return
     */
    std::vector<Tensor<CPU, AK_FLOAT>*> get_input();

    /**
     * \brief get input specified by name
     * @param name: input name
     * @return input tensor if exits, else return nullptr
     */
    Tensor<CPU, AK_FLOAT>* get_input(std::string name);

    /**
     * \brief get output tensor
     * @return
     */
    std::vector<Tensor<CPU, AK_FLOAT>*> get_output();

    /**
     * \brief get output tensor specified by name
     * @param name: output name
     * @return output tensor if exits, else return nullptr
     */
    Tensor<CPU, AK_FLOAT>* get_output(std::string name);

    /**
     * \brief do inference here
     * @return return SaberSuccess if successed
     */
    SaberStatus prediction();

private:
    //! choose big cluster or small cluster or all
    PowerMode _mode{SABER_POWER_HIGH};
    //! number of theads to run the model
    int _threads{1};
    //! runtime context to run the model
    Context* _ctx{nullptr};
    //! input shapes, check if changed
    std::vector<Shape> _last_input_shapes;
    //! container which holds the input tensors of each op
    std::vector<std::vector<Tensor<CPU, AK_FLOAT>*>> _tensor_ins;
    //! container which holds the output tensors of each op
    std::vector<std::vector<Tensor<CPU, AK_FLOAT>*>> _tensor_outs;
    //! container which holds all tensors
    std::map<std::string, Tensor<CPU, AK_FLOAT>*> _tensors;
    //! op instances
    std::vector<OpBase*> _ops;
    //! op param instances
    //std::vector<std::shared_ptr<ParamBase>> _op_params;
    //! weights data
    float* _weights{nullptr};
    //! net inputs name
    std::vector<std::string> _ins;
    //! net outputs name
    std::vector<std::string> _outs;
    /**
     * \brief initialized the net after load model, or inputs shapes have been changed
     * @return
     */
    SaberStatus init();

    /**
     * \brief load model info from file
     * @param memory
     * @return return SaberSuccess if successed
     */
    SaberStatus load_model_info(std::istream& stream);
    SaberStatus load_model_weights(std::istream& stream, size_t size);
    //SaberStatus load_model(FILE* fp);
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif // ANAKIN_SABER_LITE_NET_NET_LITE_H
