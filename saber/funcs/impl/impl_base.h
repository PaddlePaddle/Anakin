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

#ifndef ANAKIN_SABER_FUNCS_IMPL_BASE_IMPL_H
#define ANAKIN_SABER_FUNCS_IMPL_BASE_IMPL_H

#include "saber/core/context.h"
#include "saber/core/tensor.h"
#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG)
#include "saber/funcs/timer.h"
#endif

namespace anakin {
namespace saber {

template <typename TargetType,
        DataType DataType,
        typename Param>
class ImplBase {
public:

    ImplBase() {}
    virtual ~ImplBase() {}

    virtual SaberStatus init(const std::vector<Tensor<TargetType>* >& inputs,
              std::vector<Tensor<TargetType> *>& outputs,
              Param &param, Context<TargetType > &ctx) {
      return SaberUnImplError;
    }

    virtual SaberStatus create(const std::vector<Tensor<TargetType>* >& inputs,
                std::vector<Tensor<TargetType> *>& outputs,
                Param &param, Context<TargetType> &ctx) {
      return SaberUnImplError;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<TargetType>* >& inputs,
                  std::vector<Tensor<TargetType> *>& outputs,
                  Param &param) {
      return SaberUnImplError;
    }
    void set_op_name(const char* name){_op_name = name;}
    const char* get_op_name() { return _op_name.c_str();}

protected:
    Param* _param;
    Context<TargetType>* _ctx;
    std::string _op_name;
#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG)
    saber::SaberTimer<TargetType> _timer;
    saber::SaberTimer<TargetType> _trans_timer;
#endif
};
#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG)
struct GOPS{
    float ts;
    float ops;
    GOPS operator+(const GOPS& right) {
        GOPS out;
        out.ts = this->ts + right.ts;
        out.ops = this->ops + right.ops;
        return out;
    }
};

class OpTimer {
public:
    static std::map<std::string, GOPS>& ops() {
        static std::map<std::string, GOPS>* _timer = new std::map<std::string, GOPS>();
        return *_timer;
    }
    // Adds a timer type.
    static void add_timer(const std::string& type, GOPS ts) {
        std::map<std::string, GOPS>& _timer = ops();
        if (_timer.count(type) < 1) {
            _timer[type] = ts;
        } else {
            GOPS tn = _timer[type] + ts;
            _timer[type] = tn;
        }
    }

    static void clear_timer() {
        std::map<std::string, GOPS>& _timer = ops();
        _timer.clear();
    }

    static GOPS get_timer(const std::string type) {
        std::map<std::string, GOPS>& _timer = ops();
        if (_timer.count(type) < 1) {
            LOG(ERROR) << "unknow type: " << type.c_str();
            return {0.f, 0.f};
        }
        return _timer[type];
    }

    static void print_timer() {
        std::map<std::string, GOPS>& _timer = ops();
        GOPS to = get_timer("total");
        if (to.ts <= 0.f) {
            to.ts = 1.f;
        }
        for (auto& it : _timer) {
            printf("op: %s, timer: %f, GOPS: %f, percent: %f%%\n",
                it.first.c_str(), it.second.ts, 1e-6f * it.second.ops / it.second.ts, 100.f * it.second.ts / to.ts);
        }
    }
    template <typename TargetType>
    static void print_timer(Context<TargetType> const& ctx) {

        float cpu_freq_cur = ctx.get_mode() == SABER_POWER_HIGH \
            ? Env<TargetType>::cur_env()[0]._info._max_frequence : \
            Env<TargetType>::cur_env()[0]._info._min_frequence;
        float cpu_ca_theory = cpu_freq_cur * 8.0f / 1000;
        int th_num = ctx.get_threads();
        float cpus_ops = th_num * cpu_ca_theory;

        std::map<std::string, GOPS>& _timer = ops();
        GOPS to = get_timer("total");
        if (to.ts <= 0.f) {
            to.ts = 1.f;
        }
        for (auto& it : _timer) {
            printf("op: %s, timer: %f, GOPS: %f, percent: %f%%, cpu potential: %f%%\n",
                it.first.c_str(), it.second.ts, 1e-6f * it.second.ops / it.second.ts, 100.f * it.second.ts / to.ts,
                1e-6f * it.second.ops / it.second.ts / cpus_ops * 100);
        }
    }

private:
    OpTimer() {}
};

#endif
}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_BASE_IMPL_H
