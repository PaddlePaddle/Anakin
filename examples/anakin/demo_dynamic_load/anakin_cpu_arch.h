#ifndef ANAKIN_GET_CPU_ARCH_H
#define ANAKIN_GET_CPU_ARCH_H
#include <cpuid.h>
#include <string>
typedef std::string* (*cpu_arch_func)();
class CPUID_Helper {
    typedef union {
        struct {
            unsigned stepping   : 4;
            unsigned model      : 4;
            unsigned family     : 4;
            unsigned type       : 2;
            unsigned _reserved0 : 2;
            unsigned model_ext  : 4;
            unsigned family_ext : 8;
            unsigned _reserved1 : 4;
        };

        uint32_t value;

    } cpu_signature_t;
public:
    CPUID_Helper() {
        uint32_t ebx, ecx, edx;
        get_max_ext_leaf();
        __get_cpuid(0x1, &_cpu_signature.value, &ebx, &ecx, &edx);
    }
    std::string get_cpu_arch() {


        // print the stepping id, model and family of the cpu
//        printf("Stepping ID: %X\n", _cpu_signature.stepping);
//        printf("Model: %X\n", _cpu_signature.model);
//        printf("Family: %X\n", _cpu_signature.family);

        uint32_t model_num = _cpu_signature.model;

        // check the extended model number
        if (_cpu_signature.family == 0x6 || _cpu_signature.family == 0xF) {
            model_num |= (_cpu_signature.model_ext << 4);

//            printf("Extended Model: %X\n", model_num);
        }

        return microarch_info(model_num);
    }

private:

    void get_max_ext_leaf() {
        uint32_t ebx, ecx, edx;

        // get the maximum cpuid extended leaf
        __get_cpuid(0x80000000, &_max_ext_leaf, &ebx, &ecx, &edx);
    }

    std::string microarch_info(uint32_t model_num) {
        // https://software.intel.com/en-us/articles/intel-architecture-and-processor-identification-with-cpuid-model-and-family-numbers
        // https://en.wikipedia.org/wiki/List_of_Intel_CPU_microarchitectures
        // http://instlatx64.atw.hu/
        //
        // TODO: Older microarchitectures identification

        switch (model_num) {
            // atom microarchitectures
            case 0x1C:
            case 0x26:
                return "atom";//"Atom - 45 nm";

            case 0x36:
                return "atom";//"Atom - 32 nm";

                // mainline microarchitectures
            case 0x03:
            case 0x04:
                return "prescott";//"Prescott - 90 nm";

            case 0x06:
                return "presler";//"Presler - 65 nm";

            case 0x0D:
                return "dothan";//"Dothan - 90 nm";

            case 0x0F:
            case 0x16:
                return "merom";//"Merom - 65 nm";

            case 0x17:
            case 0x1D:
                return "penryn";//"Penryn - 45 nm";

            case 0x1A:
            case 0x1E:
            case 0x2E:
                return "nehalem";//"Nehalem - 45 nm";

            case 0x25:
            case 0x2C:
            case 0x2F:
                return "westmere";//"Westmere - 32 nm";

            case 0x2A:
            case 0x2D:
                return "sandybridge";//"SandyBridge - 32 nm";

            case 0x3A:
            case 0x3E:
                return "ivybridge";//"IvyBridge - 22 nm";

            case 0x3C:
            case 0x3F:
                return "haswell";//"Haswell - 22 nm";

            case 0x3D:
            case 0x4F:
                return "broadwell";//"Broadwell - 14 nm";

            case 0x55:
            case 0x5E:
                return "knl";//"Skylake - 14 nm";

            case 0x8E:
            case 0x9E:
                return "kabylake";//KabyLake - 14 nm";

            default:
                return "<Unknow>";
        }
    }
private:
    unsigned int _max_ext_leaf = 0;
    cpu_signature_t _cpu_signature;
};
#endif //ANAKIN_GET_CPU_ARCH_H
