# Compile Anakin for Intel SGX
Currently, only Linux is supported. You can either use Ubuntu or Cent OS, with
the versions supported by Intel SGX Linux driver. Check out the latest versions
of SGX software stack [here](https://01.org/intel-software-guard-extensions/downloads). 

## Steps

Follow these steps to build and run Anakin in an SGX secure enclave.

  1. Check out if your CPU and motherboard support SGX. Boot into your BIOS
     and see if there is an option controlling the availability of SGX. If
there is such an option, turn it on.
  2. Download and Install Intel SGX SDK and driver. The software packages and
     documentation can be found at [Intel Open
Source](https://01.org/intel-software-guard-extensions/downloads).
  3. Download and Install Intel MKL (not MKL-ML or MKL-DNN). You will need
     MKL 2019 Update 3. Older versions of MKL may cause problems like memory
leak. 
  4. Run the [SGX build script](../../tools/sgx_build.sh).
  5. If the build succeeds, you will find an executable called `anakin_app`
     under the `sgx_build/sgx` directory. The executable provides basic
interfaces to help you quickly deploy a model and run some inference tasks.
However, if you really need to use Anakin for SGX in production, you have to
customize the ECALL/OCALL interfaces your self. the corresponding code can be
found at [here](../../sgx).

## Support

SGX can be a complicated concept to understand for beginners. Feel free to
submit any issues if you are interested in extra security but new to SGX. In
case you are a systems developer and are knowledgeable about Intel chip
technology, you may find this [paper](https://eprint.iacr.org/2016/086.pdf)
helpful.

## Disclaimer

Anakin for SGX is still experimental and under active development. It is not
extensively tested as on other platforms. Some operators and models may not be
supported. Also, due to the limitations of the hardware, you will likely suffer
from some performance degradation. You can report considerably slow cases to us
to help improve Anakin for SGX.
