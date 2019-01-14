#include "bloomfilter/bloomfilter.h"

#include <string.h>
#include <unistd.h>
#include <stdio.h>

#include "bloomfilter/murmur3.h"

#define bit_set(v, n)    ((v)[(n) >> 3] |= (0x1 << (0x7 - ((n) & 0x7))))
#define bit_get(v, n)    ((v)[(n) >> 3] &  (0x1 << (0x7 - ((n) & 0x7))))
#define bit_clr(v, n)    ((v)[(n) >> 3] &=~(0x1 << (0x7 - ((n) & 0x7))))

unsigned int G_BLOOMFILTER_HEADER_SIZE = 32;
unsigned int G_BLOOMFILTER_MAGIC_NUM_OLD = 17062621;
unsigned int G_BLOOMFILTER_MAGIC_NUM_NEW = 17070416;

void
bloomfilter_init(struct bloomfilter *bloomfilter, uint64_t m, uint64_t k)
{
    memset(bloomfilter, 0, sizeof(*bloomfilter));
    bloomfilter->m = m;
    bloomfilter->k = k;
    bloomfilter->magic_num = G_BLOOMFILTER_MAGIC_NUM_NEW;
    bloomfilter->count = 0;
    memset(bloomfilter->bit_vector, 0, bloomfilter->m >> 3);
}

int bloomfilter_check(struct bloomfilter* filter){
    if( filter->magic_num == G_BLOOMFILTER_MAGIC_NUM_NEW){
        return 1;
    }else{
        fprintf(stderr, "error magic_num %d\n", filter->magic_num);
        return 0;
    }
}

int
bloomfilter_load_32bits(struct bloomfilter **bloomfilter, FILE *fp) {
    if(fp == NULL) {
        return 0;
    }
    unsigned char bytes[4];
    struct bloomfilter* t;
    fread(bytes, 4, 1, fp);
    uint32_t magic_num = char_to_little_endian_32bits(bytes);
    if(magic_num != G_BLOOMFILTER_MAGIC_NUM_OLD) {
        return 0;
    }
    fread(bytes, 4, 1, fp);
    uint32_t m = char_to_little_endian_32bits(bytes);
    if(m % 8 != 0) {
        return 0;
    }
    fread(bytes, 4, 1, fp);
    uint32_t k = char_to_little_endian_32bits(bytes);

    fread(bytes, 4, 1, fp);
    uint32_t count = char_to_little_endian_32bits(bytes);
    t = (struct bloomfilter*)malloc(sizeof(struct bloomfilter)+(m>>3));
    memset(t, 0, sizeof(struct bloomfilter) + (m >> 3));
    t->m = m;
    t->k = k;
    t->magic_num = magic_num;
    t->count = count;
    fseek(fp, G_BLOOMFILTER_HEADER_SIZE - 16, SEEK_CUR);
    fread(t->bit_vector, m >> 3, 1, fp);
    fseek(fp, 0, SEEK_END); // seek to end of file
    unsigned int filesize = ftell(fp);
    if (filesize != m / 8 + G_BLOOMFILTER_HEADER_SIZE) {
        free(t);
        return 0;
    }
    *bloomfilter = t;
    return 1;
}

int
bloomfilter_load(struct bloomfilter **bloomfilter, const void *path)
{
    struct bloomfilter* t;
    unsigned char bytes[8];
    FILE * file = fopen(path, "rb");
    if (file != NULL) {
        if(bloomfilter_load_32bits(bloomfilter, file) > 0) {
            fclose(file);
            return 1;
        }
        //back to beginning of file
        fseek(file, 0, SEEK_SET);
        fread(bytes, 8, 1, file);
        uint64_t magic_num = char_to_little_endian_64bits(bytes);
        if(magic_num  != G_BLOOMFILTER_MAGIC_NUM_NEW) {
            fclose(file);
            return 0;
        }
        fread(bytes, 8, 1, file);
        uint64_t m = char_to_little_endian_64bits(bytes);
        if(m % 8 != 0) {
            fclose(file);
            return 0;
        }
        fread(bytes, 8, 1, file);
        uint64_t k = char_to_little_endian_64bits(bytes);

        fread(bytes, 8, 1, file);
        uint64_t count = char_to_little_endian_64bits(bytes);

        t = (struct bloomfilter*)malloc(sizeof(struct bloomfilter)+(m>>3));
        memset(t, 0, sizeof(struct bloomfilter) + (m >> 3));
        t->m = m;
        t->k = k;
        t->magic_num = magic_num;
        t->count = count;
        fread(t->bit_vector, m >> 3, 1, file);
        fseek(file, 0, SEEK_END); // seek to end of file
        unsigned int filesize = ftell(file);
        fclose(file);
        if(filesize != m / 8 + G_BLOOMFILTER_HEADER_SIZE) {
            free(t);
            return 0;
        }
        *bloomfilter = t;
        return 1;
    }
    fprintf(stderr, "file %s not exist\n", path);
    return 0;
}

int
bloomfilter_set(struct bloomfilter *bloomfilter, const void *key, size_t len)
{
    if(bloomfilter_get(bloomfilter, key, len) > 0) {
        return 0;
    }
    uint32_t i;
    uint64_t  result[2];
    for (i = 0; i < bloomfilter->k; i++) {
        murmurhash3_x64_128(key, len, i, &result);
        result[0] %= bloomfilter->m;
        result[1] %= bloomfilter->m;
        bit_set(bloomfilter->bit_vector, result[0]);
        bit_set(bloomfilter->bit_vector, result[1]);
    }
    bloomfilter->count++;
    return 1;
}

int
bloomfilter_set_nocheck(struct bloomfilter *bloomfilter, const void *key, size_t len)
{
    uint32_t i;
    uint64_t  result[2];
    for (i = 0; i < bloomfilter->k; i++) {
        murmurhash3_x64_128(key, len, i, &result);
        result[0] %= bloomfilter->m;
        result[1] %= bloomfilter->m;
        bit_set(bloomfilter->bit_vector, result[0]);
        bit_set(bloomfilter->bit_vector, result[1]);
    }
    bloomfilter->count++;
    return 1;
}

int
bloomfilter_get(struct bloomfilter *bloomfilter, const void *key, size_t len)
{
    uint32_t i;
    uint64_t  result[2];

    for (i = 0; i < bloomfilter->k; i++) {
        murmurhash3_x64_128(key, len, i, &result);
        result[0] %= bloomfilter->m;
        result[1] %= bloomfilter->m;
        if (!bit_get(bloomfilter->bit_vector, result[0])){
            return 0;
        }
        if (!bit_get(bloomfilter->bit_vector, result[1])){
            return 0;
        }
    }
    return 1;
}

int
bloomfilter_get_hash(struct bloomfilter *bloomfilter, const void *key, size_t len, char *dst)
{
    uint32_t i;
    uint64_t  result[2];
    char hash[255] = "";
    char valstr[32];
    for (i = 0; i < bloomfilter->k; i++) {
        murmurhash3_x64_128(key, len, i, &result);
        sprintf(valstr, "%lld,", result[0]);
        strcat(hash, valstr);
        sprintf(valstr, "%lld,", result[1]);
        strcat(hash, valstr);
    }
    strcpy(dst, hash);
    return 1;
}

int
bloomfilter_dump(struct bloomfilter *bloomfilter, const void *path)
{
    FILE * file = fopen(path, "wb");
    if (file != NULL) {
        fwrite(&bloomfilter->magic_num, sizeof(bloomfilter->magic_num), 1, file);
        fwrite(&bloomfilter->m, sizeof(bloomfilter->m), 1, file);
        fwrite(&bloomfilter->k, sizeof(bloomfilter->k), 1, file);
        fwrite(&bloomfilter->count, sizeof(bloomfilter->count), 1, file);
        fwrite(bloomfilter->bit_vector, (bloomfilter->m >> 3), 1, file);
        fclose(file);
        return 1;
    }
    return 0;
}

/**
 * works either big-endian or little-endian architectures
 */
uint32_t
char_to_little_endian_32bits(unsigned char *bytes) {
    return bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
}

/**
 * works either big-endian or little-endian architectures
 */
uint64_t
char_to_little_endian_64bits(unsigned char *bytes) {
    uint64_t bytes_ull[8];
    int i;
    for(i = 0; i < 8; i++) {
        bytes_ull[i] = bytes[i];
    }
    return bytes_ull[0] | (bytes_ull[1] << 8) | (bytes_ull[2] << 16) | (bytes_ull[3] << 24) | 
            (bytes_ull[4] << 32) | (bytes_ull[5] << 40) | (bytes_ull[6] << 48) | (bytes_ull[7] << 56);
}
