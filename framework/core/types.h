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

#ifndef ANAKIN_TYPES_H
#define ANAKIN_TYPES_H 

namespace anakin {

/**
 *  \brief Anakin running precision
 *   some target plantform maybe doesn't support some precision. 
 */
enum class Precision : int {
    INT4 = -10,
    INT8 = -2,
    FP16 = -1,
    FP32 = 0,
    FP64
};

/** 
 *  \brief Operator run type of operator executor.
 */
enum class OpRunType : int {
    SYNC,           ///< the net exec synchronous (for GPU, means single-stream)
    ASYNC           ///< ASYNC the net exec asynchronous (for GPU, means mutli-stream)
};

/**
 *  \brief service run pattern
 */
enum class ServiceRunPattern: int {
    SYNC,
    ASYNC
};

/** 
 *  \brief Inner return type used by Status type.
 */
enum class RetType {
    SUC,            ///< succeess
    ERR,            ///< error
    IMME_EXIT,      ///< need immediately exit
};

/** 
 *  \brief Request type for input of operator in inference.(NOT used yet)
 *
 *   Normally operator doesn't need, except cases like NLP data or Batch Norm requires.
 */
enum class EnumReqType {
    OFFSET,         ///< request hold offset info for inputs. which is used in sequence data.
    NONE            ///< hold none data for inputs.
};


} /* namespace anakin */

#endif
