/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

#ifndef ANAKIN_FRAMEWORK_LITE_CODE_GENERATE_CPP_H
#define ANAKIN_FRAMEWORK_LITE_CODE_GENERATE_CPP_H

#include "framework/lite/op_map.h"
#include "framework/lite/code_gen_base.h"

namespace anakin {

namespace lite {

/**  
 *  \brief class to generate cpp files.
 *
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class GenCPP : public CodeGenBase<Ttype, Dtype, Ptype> {
public:
	explicit GenCPP(std::string model_name, std::string model_dir = ".") {
		_cpp_file_name = model_dir + '/' + model_name + ".cpp";
		_h_file_name = model_dir + '/' + model_name + ".h";
		_model_file_name = model_dir + '/' + model_name + ".lite.bin";
		_model_opt_file_name = model_dir + '/' + model_name + ".lite.info";
		_weight_opt_file = model_dir + '/' + model_name + ".tmp";
		_weights.open(_model_file_name);
		_opt_weights.open(_weight_opt_file);
		_opt_param_write.open(_model_opt_file_name);
 		_code_name = model_name;
		_g_weights_ptr_name = _code_name+"_weights_ptr";
	}
	~GenCPP()=default;

	/// generate all cpp files
	virtual void gen_files(const bool debug_mode) {
		gen_header();
		gen_source(debug_mode);
	}

private:
	void gen_license();
	void gen_header_start();
	void gen_header_end();
	void gen_source_start();
	void gen_source_end();

	/**
	 * \brief generator optimized model for lite executer
	 */
	void gen_opt_model();

	/**
	 * \brief generate tensors for edges
	 */
	void gen_tensors();
	
	/**
	 * \brief initialize tensors for edges
	 */
	void tensors_init();

	/**
	 * \brief generate model's inputs and outputs
	 */
	void gen_model_ios();

	/**
	 * \brief initialize model's inputs and outputs
	 */
	void model_ios_init();

	/**
	 * \brief generate operations for model
	 */
	virtual void gen_ops();

	/**
	 * \brief generate initial impl api for model
	 */
	void gen_init_impl();

	/**
	 * \brief generate running api impl for model
	 */
	void gen_run_impl(const bool debug_mode);


	/**
	 * \brief  generate api for model
	 */
	void gen_head_api();

	/**
	 * \brief generate head api implement
	 */
	void gen_head_api_impl();

	/**
	 * \biref generata header file
	 */
	void gen_header();

	/**
	 * \biref generata source file
	 */
	void gen_source(const bool debug_mode);

private:
	std::string _cpp_file_name;
	std::string _h_file_name;
	std::string _model_file_name;
	std::string _model_opt_file_name;
	std::string _code_name;
	std::string _g_weights_ptr_name;
	std::string _weight_opt_file;
	CodeWritter _code;
	CodeWritter _opt_param_write;
	WeightsWritter _weights;
	WeightsWritter _opt_weights;

};

} /* namespace lite */

} /* namespace anakin */

#endif
