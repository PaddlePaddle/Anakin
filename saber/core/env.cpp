#include "env.h"

namespace anakin {

    namespace saber {

#ifdef USE_BM

        template<>
        void Env<BM>::env_init(int max_stream){
            //TODO: decide what to put here
            LOG(INFO) << "env init for BM";
        }

#endif


    }
}
