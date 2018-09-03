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

#ifndef ANAKIN_LLVM_BASE_H
#define ANAKIN_LLVM_BASE_H

#include <vector>
#include "utils/logger/logger.h"

namespace anakin {

namespace graph {

    /**
    * \brief enum OpType define
    * using number is to prensent the algorithm operation
    */
    enum OpType {
        NONE = -20,       ///< -20 stand for Null operation
        ADD = -7,         ///< -7 stand for ADD operation
        SUB = -6,         ///< -6 stand for SUB operation
        MUL = -5,         ///< -5 stand for MUL operation
        DIV = -4,         ///< -4 stand for DIV operation
        MOD = -3,         ///< -3 stand for MOD operation
        MIN = -2,         ///< -2 stand for MIN operation
        MAX = -1,         ///< -1 stand for MAX operation
        CONV = 1,         ///< 1 stand for CONV operation
        DCONV = 2,        ///< 2 stand for DCON operation
        RELU = 3,         ///< 3 stand for RELU operation
        FULL_CONNECT = 4, ///< 4 stand for FULL_CONNECT operation
        POOLING = 5,      ///< 5 stand for POOLING operation
        EMBEDDING = 6,    ///< 6 stand for EMBEDDING operation
        RNN = 7,          ///< 7 stand for RNN operation
        BATCHNORM = 8,    ///< 8 stand for BATCHNORM operation
        TOPK = 9,         ///< 9 stand for TOPK operation
        REVERSE = 10      ///< 10 stand for REVERSE operation
    };
    /**
    * \brief enum OpProperty define
    * using number is to prensent the propery operation
    */
    enum OpProperty {
        assemble = 0, ///< 0 stand for assemble
        scatter = 1, ///< 1 stand for scatter
        mutli_io ///< 2 stand for mutli_io
    };


} /* namespace graph */

} /* namespace anakin */

#endif
