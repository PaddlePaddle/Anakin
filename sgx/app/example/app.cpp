#include <cstdio>
#include <ctime>
#include "enclave_u.h"
#include "sgx_urts.h"

/* Initialize the enclave:
 *   Step 1: try to retrieve the launch token saved by last transaction
 *   Step 2: call sgx_create_enclave to initialize an enclave instance
 *   Step 3: save the launch token if it is updated
 */
int initialize_enclave(sgx_enclave_id_t* eid, const char *token_path, const char *enclave_name) {
    sgx_launch_token_t token = {0};
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    int updated = 0;

    /* Step 1: try to retrieve the launch token saved by last transaction
     *         if there is no token, then create a new one.
     */
    /* try to get the token saved in $HOME */
    FILE* fp = fopen(token_path, "rb");
    if (fp == nullptr && (fp = fopen(token_path, "wb+")) == NULL) {
        printf("Warning: Failed to create/open the launch token file \"%s\".\n", token_path);
    }

    if (fp != nullptr) {
        /* read the token from saved file */
        size_t read_num = fread(token, 1, sizeof(sgx_launch_token_t), fp);
        if (read_num != 0 && read_num != sizeof(sgx_launch_token_t)) {
            /* if token is invalid, clear the buffer */
            memset(&token, 0x0, sizeof(sgx_launch_token_t));
            printf("Warning: Invalid launch token read from \"%s\".\n", token_path);
        }
    }

    /* Step 2: call sgx_create_enclave to initialize an enclave instance */
    ret = sgx_create_enclave(enclave_name, SGX_DEBUG_FLAG, &token, &updated, eid, nullptr);
    if (ret != SGX_SUCCESS) {
        if (fp != nullptr) fclose(fp);
        return -1;
    }

    /* Step 3: save the launch token if it is updated */
    if (updated == false || fp == nullptr) {
        /* if the token is not updated, or file handler is invalid, do not perform saving */
        if (fp != nullptr) fclose(fp);
        return 0;
    }

    /* reopen the file with write capablity */
    fp = freopen(token_path, "wb", fp);
    if (fp == nullptr) return 0;
    size_t write_num = fwrite(token, 1, sizeof(sgx_launch_token_t), fp);
    if (write_num != sizeof(sgx_launch_token_t))
        printf("Warning: Failed to save launch token to \"%s\".\n", token_path);
    fclose(fp);
    return 0;
}

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;

#define SGX_INPUT_MAX (1024U * 1024U * 1U)
uint8_t sgx_input[SGX_INPUT_MAX];

#define SGX_OUTPUT_MAX (1024U * 1024U * 1U)
uint8_t sgx_output[SGX_OUTPUT_MAX];

int main(int argc, char const *argv[]) {
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "usage: %s model_name [input_file]\n", argv[0]);
        return 1;
    }

    size_t input_size = 0;
    if (argc == 3) {
        FILE *input_file = fopen(argv[2], "rb");

        if (!input_file) {
            fprintf(stderr, "error: cannot open input file %s\n", argv[2]);
            return 1;
        }

        fseek(input_file, 0, SEEK_END);
        long int fend = ftell(input_file);
        fseek(input_file, 0, SEEK_SET);

        if (fend > sizeof(sgx_input)) {
            fprintf(stderr, "error: oversized input\n");
            return 1;
        }

        if (fend <= 0) {
            fprintf(stderr, "error: cannot read input file\n");
            return 1;
        }

        input_size = fend;
        if (input_size != fread(sgx_input, 1, input_size, input_file)) {
            fprintf(stderr, "error: cannot read input file\n");
            return 1;
        }

        fclose(input_file);
    }

    if (initialize_enclave(&global_eid, "anakin_enclave.token", "anakin_enclave.signed") < 0) {
        printf("Fail to initialize enclave.\n");
        return 1;
    }

    int ecall_retcode = -1;
    sgx_status_t status = setup_model(global_eid, &ecall_retcode, argv[1]);

    if (status != SGX_SUCCESS) {
        fprintf(stderr, "error: SGX ecall 'setup_model' failed.\n");
        return 1;
    }

    if (ecall_retcode) {
        fprintf(stderr, "error: invalid anakin model.\n");
        return 1;
    }

    clock_t begin = clock();

    size_t result_size = 0;
    ecall_retcode = -1;

    status = infer(global_eid, &ecall_retcode, input_size, sgx_input,
                   sizeof(sgx_output), sgx_output, &result_size);

    if (status != SGX_SUCCESS) {
        fprintf(stderr, "error: SGX ecall 'infer' failed.\n");
        return 1;
    } else if (ecall_retcode) {
        fprintf(stderr, "error: invalid inference parameters.\n");
    }

    clock_t end = clock();

    fprintf(stderr, "%lf seconds elapsed during inference\n", (double)(end - begin) / CLOCKS_PER_SEC);

    auto f = reinterpret_cast<float *>(sgx_output);
    auto n = result_size / sizeof(float);
    for (int i = 0; i < n; ++i) {
        printf("%f\n", f[i]);
    }

    return 0;
}
