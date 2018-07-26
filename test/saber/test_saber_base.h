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

#ifndef ANAKIN_TEST_SABER_BASE_H
#define ANAKIN_TEST_SABER_BASE_H


#include "utils/unit_test/aktest.h"
#include "utils/logger/logger.h"
#include "saber/funcs/base.h"
#include "saber/core/tensor.h"
#include "saber/core/shape.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "tensor_op.h"
#include <vector>
#include <string>

using namespace anakin::test;
namespace anakin{
    
namespace saber{
    
    template <typename TargetType_D,typename TargetType_H,DataType Dtype,
        template <typename T,DataType D> class Op,
        template <typename T> class Param>
    class TestSaberBase{
    public:
        typedef Param<TargetType_D> Param_t;
        typedef Op<TargetType_D,Dtype> Op_t;
        typedef Tensor<TargetType_H> TensorH;
        typedef Tensor<TargetType_D> TensorD;
        typedef std::vector<TensorD*> Input_dt;
        typedef std::vector<TensorD*> Output_dt;
        typedef std::vector<TensorH*> Input_ht;
        typedef std::vector<TensorH*> Output_ht;
        typedef void (*CpuFunc_t)(const Input_ht&,Output_ht&,Param_t& param);
        
        
        TestSaberBase(int num=1) : _op_input_num(num) {
        }
        
        TestSaberBase(Param_t& param,int num=1) : _op_input_num(num) {
            _params.push_back(param);
        }
        
        void add_param(Param_t& param){
            _params.push_back(param);
        }
        void set_param(Param_t& param){
            _params.clear();
            _params.push_back(param);
        }
        
        void add_inputs_shape(Shape new_shape){
            
            std::vector<TensorD*> in_d;
            std::vector<TensorH*> in_h;
            std::vector<TensorD*> out_d;
            std::vector<TensorH*> out_h;
            std::vector<TensorH*> in_hd;
            
            for(int i=0;i<_op_input_num;++i){
                TensorD *d_id=new TensorD(new_shape);
                TensorH *d_ih=new TensorH(new_shape);
                TensorD *d_od=new TensorD(new_shape);
                TensorH *d_oh=new TensorH(new_shape);
                TensorH *d_ihd=new TensorH(new_shape);
                in_d.push_back(d_id);
                in_h.push_back(d_ih);
                out_d.push_back(d_od);
                out_h.push_back(d_oh);
                in_hd.push_back(d_ihd);
            }
            _inputs_dev.push_back(in_d);
            _inputs_host.push_back(in_h);
            _outputs_dev.push_back(out_d);
            _outputs_host.push_back(out_h);
            _outputs_hd.push_back(in_hd);
            _input_shapes.push_back(new_shape);
            
            
        }
        void clear_datas(){
            _inputs_dev.clear();
            _inputs_host.clear();
            _outputs_dev.clear();
            _outputs_host.clear();
            _outputs_hd.clear();
            _input_shapes.clear();
        }
        void set_input_shape(Shape new_shape,TestDataType type=RANDOM,double value=1){
            clear_datas();
            
            add_inputs_shape(new_shape);
            _input_type=type;
            _special_value=value;
            
        }
        void auto_gen_inputs(){
            int n,c,h,w;
            for(int n:{1,2}){
                for(int c:{32,64}){
                    for(int h:{64,256}){
                        for(int w:{64,256}){
                            add_inputs_shape(Shape({n,c,h,w}));
                        }
                    }
                }
                    
            }
        }
        void fill_inputs(double minv,double maxv){
            int input_size=_inputs_dev.size();
            CHECK_EQ(input_size,_inputs_host.size())<<"dev and host inputs num must be equal";
            if(_input_type==RANDOM){
                CHECK_EQ(input_size,1)<<"special input num must be 1";
                for(int i=0;i<_inputs_dev.size();++i){
                    for(int j=0;j<_op_input_num;++j){
                        fill_tensor_rand(*_inputs_dev[i][j], minv, maxv);
                        _inputs_host[i][j]->copy_from(*_inputs_dev[i][j]);
                    }
                }
            }
            else{
                for(int i=0;i<_inputs_dev.size();++i){
                    for(int j=0;j<_op_input_num;++j){
                        fill_tensor_const(*_inputs_dev[i][j],_special_value);
                        _inputs_host[i][j]->copy_from(*_inputs_dev[i][j]);
                    }
                }
            }
            //print_tensor(*_inputs_host[0][0]);
                
        }
        void add_custom_input(Input_dt& input){
            CHECK_EQ(input.size(),_op_input_num)<<"input must equal op_input_num";
            clear_datas();
            Shape sh=input[0]->shape();
            add_inputs_shape(sh);
            for(int i=0;i<_op_input_num;++i)
            {
                _inputs_dev[0][i]->copy_from(*input[i]);
                _inputs_host[0][i]->copy_from(*input[i]);
            }
            _input_type=CUSTOM;
            
        }
        void compute_outputs_shape(int param_index=0){
            CHECK_GT(_params.size(),0)<<"no available param";
            CHECK_GT(_inputs_dev.size(),0)<<"no available inputs";
            CHECK_GE(param_index,0)<<"param index must be positive";
            CHECK_EQ(_inputs_dev.size(),_outputs_dev.size())<<"inputs and outputs must have same num";
            CHECK_LT(param_index,_params.size())<<"param_index out of range";
            for(int i=0;i<_inputs_dev.size();++i){
                SABER_CHECK(_baseOp.compute_output_shape(_inputs_dev[i],
                                                    _outputs_dev[i],_params[param_index]));
            }
            for(int i=0;i<_outputs_dev.size();++i)
            {
                for(int j=0;j<_op_input_num;++j)
                {
                    Shape sh=_outputs_dev[i][j]->valid_shape();
                    _outputs_host[i][j]->re_alloc(sh,Dtype);
                    _outputs_hd[i][j]->re_alloc(sh,Dtype);
                }
            }
        }
        void get_op_result(SaberImplStrategy strategy,ImplEnum implenum,int param_index=0){
            CHECK_GE(param_index,0)<<"param index must be positive";
            CHECK_LT(param_index,_params.size())<<"param index out of range";
            
            Context<TargetType_D> ctx(0,1,1);
            for(int input_index=0;input_index<_inputs_dev.size();++input_index){
                    _baseOp.init(_inputs_dev[input_index],_outputs_dev[input_index],
                             _params[param_index],strategy,implenum,ctx);
                for(int iter=0;iter<100;++iter){
                    _baseOp(_inputs_dev[input_index],_outputs_dev[input_index],
                        _params[param_index],ctx);
                    //cudaDeviceSynchronize();
                    typename TensorD::API::stream_t stream=ctx.get_compute_stream();
                    //always 0？
                    _outputs_dev[input_index][0]->record_event(stream);
                    _outputs_dev[input_index][0]->sync();//
                }
                
                for(int j=0;j<_op_input_num;++j){
                    _outputs_hd[input_index][j]->copy_from(*_outputs_dev[input_index][j]);
                }
                //print_tensor(*_outputs_hd[0][0]);
            }
            
        }
        virtual void get_cpu_result(CpuFunc_t CpuFunc,int param_index=0){
            CHECK_EQ(_inputs_dev.size(),_outputs_dev.size())<<"input and output number must be equal";
            CHECK_EQ(_outputs_hd.size(),_outputs_dev.size())<<"input and output number must be equal";
            for(int i=0;i<_inputs_dev.size();++i){
                CpuFunc(_inputs_host[i],_outputs_host[i],_params[param_index]);//depend on cpu func form?
                
            }
        }
        virtual void result_check_accuracy(double succ_ratio=0.00001){
            CHECK_EQ(_outputs_host.size(),_outputs_hd.size())<<"output size in dev and cpu must be equal";
            int check_size=_outputs_host.size();
            std::vector<double> max_diff(check_size,0);
            std::vector<double> max_ratio(check_size,0);
            Shape sh=_inputs_host[0][0]->shape();
            for(int i=0;i<_outputs_host.size();++i)
            {
                for(int j=0;j<_op_input_num;++j){
                    tensor_cmp_host<float>((const float*)_outputs_hd[i][j]->data(), (const float*)_outputs_host[i][j]->data(),
                                    _outputs_hd[i][j]->valid_size(), max_ratio[i], max_diff[i]);
                    LOG(INFO)<<"input_shape:("<<sh.num()<<","<<sh.channel()<<","<<sh.height()<<","<<sh.width()<<")";
                    LOG(INFO)<<"max_ratio:"<<max_ratio[i];
                    if(max_ratio[i]<=succ_ratio)
                        LOG(INFO)<<"Test Passed!";
                    else
                        LOG(FATAL)<<"Test Failed!!";
                        //LOG(ERROR)<<"Test Failed!!";
                }
                
            }
            
            
        }
        void set_rand_limit(double minv,double maxv){
            _max_value=maxv;
            _min_value=minv;
        }
        void run_test(CpuFunc_t CpuFunc,double succ_ratio=0.00001){
            if(_input_type==SPECIAL)
                    fill_inputs(_special_value,_special_value);
            if(_input_type==RANDOM)
                    fill_inputs(_min_value,_max_value);
            //LOG(INFO)<<"compute shape";
            compute_outputs_shape();
            Env<TargetType_D>::env_init();
            Env<TargetType_H>::env_init();
            //LOG(INFO)<<"init fini";
            
            std::vector<std::string> runtype{"STATIC","RUNTIME","SPECIFY"};
            std::vector<std::string> impltype{"SABER"," VENDER"};
            for(auto strate:{SPECIFY,RUNTIME,STATIC}){
                for(auto implenum:{SABER_IMPL,SABER_IMPL}){
                    LOG(INFO)<<"TESTING: strategy:"<<runtype[strate-1]<<",impltype:"<<impltype[(int)implenum];
                    get_op_result(strate,implenum);
                    get_cpu_result(CpuFunc);
                    result_check_accuracy(succ_ratio);
                }
            }
            
        }
        virtual void result_check_speed(){
        }
    private:
        int _op_input_num;
        Op_t _baseOp;
        TestDataType _input_type;
        double _special_value;
        double _max_value{255.0};
        double _min_value{-255.0};
        std::vector<Input_ht> _inputs_host;
        std::vector<Input_dt> _inputs_dev;
        std::vector<Output_dt> _outputs_dev;
        std::vector<Output_ht> _outputs_host;
        std::vector<Output_ht> _outputs_hd;
        std::vector<Shape> _input_shapes;
        std::vector<Param_t> _params;
    
    };//testsaberbase
    
}//namespace saber
}//namespace anakin

#endif //ANAKIN_TEST_SABER_BASE_H
