#include <sgx_error.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "stdio.h"
#include "enclave_t.h"

struct _FILE {
    uintptr_t untrusted;
    size_t bytes_left;
    unsigned char *buffer;
    unsigned char *curp;
};

FILE *stdout = NULL;
FILE *stdin = NULL;
FILE *stderr = NULL;

#define SGX_PRINTF_BUFSIZE    4096
#define SGX_FILE_IO_BUFSIZE   4096

int printf(const char *fmt, ...) {
    char buf[SGX_PRINTF_BUFSIZE];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, SGX_PRINTF_BUFSIZE, fmt, ap);
    va_end(ap);
    return ocall_print_string(buf);
}

int fprintf(FILE *f, const char *format, ...) {
    return 0;
}

int putchar(int character) {
    char buf[2] = { character, '\0' };
    return ocall_print_string(buf);
}

#define FILE_MODE_READ  0
#define FILE_MODE_WRITE 1
#define FILE_MODE_ERROR 2

FILE *fopen(const char *name, const char *mode) {
    int fmode = FILE_MODE_ERROR;

    if (strncmp(mode, "rb", 3) == 0)
        fmode = FILE_MODE_READ;
    else if (strncmp(mode, "wb", 3) == 0)
        fmode = FILE_MODE_WRITE;
    else
        return NULL;

    uintptr_t f = 0;
    sgx_status_t ec = ocall_fopen(&f, name, mode);

    if (ec != SGX_SUCCESS)
        return NULL;

    FILE *ret = malloc(sizeof(FILE));

    ret->untrusted = f;
    ret->buffer = malloc(SGX_FILE_IO_BUFSIZE);
    ret->curp = ret->buffer;

    if (fmode == FILE_MODE_READ)
        ret->bytes_left = 0;
    else
        ret->bytes_left = SGX_FILE_IO_BUFSIZE;

    return ret;
}

size_t fwrite(const void *buf, size_t size, size_t count, FILE *f) {
    size_t bytes_written = 0;
    sgx_status_t ec = ocall_fwrite(&bytes_written, buf, size, count, f->untrusted);

    if (ec != SGX_SUCCESS)
        return 0;

    return bytes_written;
}

size_t fread(void *buf, size_t size, size_t count, FILE *f) {
    const size_t total = size * count;
    size_t left = total;
    unsigned char *_buf = buf;

    do {
        size_t round = f->bytes_left < left ? f->bytes_left : left;
        if (round != 0) {
            memcpy(_buf, f->curp, round);
            f->curp += round;
            _buf += round;
            left -= round;
            f->bytes_left -= round;
        }

        if (f->bytes_left == 0) {
            f->curp = f->buffer;

            sgx_status_t ec;

            ec = ocall_fread(&f->bytes_left, f->buffer,
                             1, SGX_FILE_IO_BUFSIZE, f->untrusted);

            if (ec != SGX_SUCCESS) {
                return total - left;
            }

            if (f->bytes_left == 0)
                break;
        }
    } while (left > 0);

    return total - left;
}

int fseek(FILE *f, long int offset, int origin) {
    int ret = -1;
    sgx_status_t ec = ocall_fseek(&ret, f->untrusted, offset, origin);

    if (ec != SGX_SUCCESS)
        return -1;

    return ret;
};

long int ftell(FILE *f) {
    long int ret = -1L;
    sgx_status_t ec = ocall_ftell(&ret, f->untrusted);

    if (ec != SGX_SUCCESS)
        return -1L;

    return ret;
};

size_t fsize(FILE *f) {
    size_t size = 0;
    sgx_status_t ec = ocall_fsize(&size, f->untrusted);

    if (ec != SGX_SUCCESS)
        return 0;

    return size;
}

int fclose(FILE *f) {
    int r = EOF;

    sgx_status_t ec = ocall_fclose(&r, f->untrusted);

    free(f->buffer);
    free(f);

    if (ec != SGX_SUCCESS)
        return EOF;

    return r;
}
