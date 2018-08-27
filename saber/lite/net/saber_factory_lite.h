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

#ifndef ANAKIN_SABER_LITE_NET_OP_FACTORY_LITE_H
#define ANAKIN_SABER_LITE_NET_OP_FACTORY_LITE_H

#include "saber/lite/funcs/op_base.h"

namespace anakin{

namespace saber{

namespace lite{

//template <typename Dtype>
class OpRegistry {
public:
    typedef OpBase* (*Creator)();
    typedef std::map<std::string, Creator> CreatorRegistry;

    static CreatorRegistry& lite_ops() {
        static CreatorRegistry* g_ops = new CreatorRegistry();
        return *g_ops;
    }

    // Adds a op.
    static void add_op(const std::string op_name, Creator op) {
        CreatorRegistry& registry = lite_ops();
        if (registry.count(op_name) > 0) {
            printf("Op type %s already registered.\n", op_name.c_str());
        } else {
            registry[op_name] = op;
        }
    }

    // Get a layer using  a layer typename, such as "Convolution, Covolution_arm or convolution_cudnn".
    static OpBase* create_op(const std::string layer_type) {

        CreatorRegistry& registry = lite_ops();
        LCHECK_EQ(registry.count(layer_type), 1, "Unknown op type");
        return registry[layer_type]();
    }

    static std::vector<std::string> LayerTypeList() {
        CreatorRegistry& registry = lite_ops();
        std::vector<std::string> layer_types;
        for (typename CreatorRegistry::iterator iter = registry.begin();
             iter != registry.end(); ++iter) {
            layer_types.push_back(iter->first);
        }
        return layer_types;
    }

private:
    // Layer registry should never be instantiated - everything is done with its
    // static variables.
    OpRegistry() {}

    static std::string LayerTypeListString() {
        std::vector<std::string> layer_types = LayerTypeList();
        std::string layer_types_str;
        for (std::vector<std::string>::iterator iter = layer_types.begin();
             iter != layer_types.end(); ++iter) {
            if (iter != layer_types.begin()) {
                layer_types_str += ", ";
            }
            layer_types_str += *iter;
        }
        return layer_types_str;
    }
    // check layer type is exit or not
    static bool has_type(std::string op_name) {
        CreatorRegistry& registry = lite_ops();
        return  registry.count(op_name) == 1;
    }
};


//template <typename Dtype>
class LayerRegisterer {
public:
    LayerRegisterer(const std::string op_name,
                    OpBase* (*creator)()) {
        OpRegistry::add_op(op_name, creator);
    }
};

#define REGISTER_LAYER_CREATOR(op_name, creator)                    \
  static LayerRegisterer g_creator_f_##op_name(#op_name, creator);

#define REGISTER_LAYER_CLASS(op_name)                               \
  OpBase* Creator_##op_name##Layer() {                              \
    return new op_name;                                             \
  }                                                                 \
  REGISTER_LAYER_CREATOR(op_name, Creator_##op_name##Layer)

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_NET_OP_FACTORY_LITE_H
